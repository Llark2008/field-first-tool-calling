"""Audit, freeze, certify and execute a fixed 75-update mechanism pilot."""
import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path

from .ablation_data import audit_candidate,read_rows
from .controls import source_guard
from .evaluate_formal import summarize,failure_type
from .formal_data import sha
from .generation_scoring import score_generation
from .live_progress import Heartbeat,update_state
from .mechanism_analysis import ARMS,STEPS,REFERENCES,analyze
from .resource_monitor import ResourceMonitor,resource_violation
from .run import runtime_root,configure_runtime
from .run_clarification_ablation import (read_json,write_json,verify_file_map,verify_checkpoint_bundle,
    validate_evaluation,eval_meta,finalize_resources,record,token_audit,verify as verify_parent)
from .train_ablation import certify_control,validate_config
from .train_formal import exposure_schedule,SEED,configure_mlx_memory
from .train_mechanism import recipe,verify_checkpoint,train_variant


def source_hashes(source):
    return {str(p.relative_to(source)):sha(p) for folder in ['toolcall_lab','tests'] for p in sorted((source/folder).glob('*.py'))}


def verify_reused(frozen):
    verify_file_map(Path(frozen['parent_run']),frozen['parent_files'])


def reuse_specs(parent):
    old=Path(parent['old_run'])
    specs=[dict(name=n,old_name=n,prompt=n.removeprefix('base_') if n.startswith('base_') else 'v2',
                adapter=str(old/'train-A/step-0600') if n=='A75' else None) for n in REFERENCES]
    return specs+[dict(name=f'M0-{s}',old_name=f'R0-{s}',prompt='v2',adapter=str(old/f'train-B/step-{s:04d}')) for s in STEPS]


def rescore(cases,rows,path,repair_missing_metrics=False):
    if len(rows)!=len(cases) or [r['case_id'] for r in rows]!=[c['id'] for c in cases]:
        raise ValueError('Rescoring requires complete aligned rows')
    for c,r in zip(cases,rows):
        score=score_generation(c,r['output'],r['finish_reason'])
        if score!=r['score']:raise ValueError('Rescoring mismatch: '+r['case_id'])
        valid=score['schema_valid'] and not score['truncated'] and score.get('action',{}).get('action')=='call'
        guard=source_guard(c['request'],c['tools'],score['action']) if valid else dict(blocked=False,ungrounded_fields=[])
        if guard!=r['guard'] or failure_type(r)!=r['failure_type']:raise ValueError('Guard/failure mismatch')
    aggregate=summarize(rows);metrics_path=Path(path).with_suffix('.metrics.json')
    if repair_missing_metrics and not metrics_path.exists():write_json(metrics_path,aggregate)
    if aggregate!=read_json(metrics_path):raise ValueError('Saved aggregate mismatch')
    return dict(file=str(path),rows=len(rows),sha256=sha(path),generation_seconds=sum(r['seconds'] for r in rows))


def audit_inputs(parent,source):
    parent=Path(parent);p=read_json(parent/'frozen.json');verify_parent(parent,source,p)
    if read_json(parent/'completed.json').get('fingerprint')!=p['fingerprint']:raise ValueError('Parent not complete')
    data=Path(p['dataset']);structural=audit_candidate(data);tokens=token_audit(data,Path(p['snapshot']))
    if tokens!=p['token_audit']:raise ValueError('Historical token audit changed')
    cases=read_rows(data/'dev.jsonl');reused=[]
    for spec in reuse_specs(p):
        adapter=Path(spec['adapter']) if spec['adapter'] else None
        path=parent/'eval'/('dev-'+spec['old_name']+'.jsonl')
        rows=validate_evaluation(path,cases,eval_meta(p,spec['old_name'],'dev',spec['prompt'],adapter,cases))
        reused.append(dict(name=spec['name'],**rescore(cases,rows,path)))
    from mlx_lm.utils import load_tokenizer
    from .training_spike import encode_row
    rows=read_rows(data/'D0-sft.jsonl');tokenizer=load_tokenizer(Path(p['snapshot']))
    schedule=exposure_schedule(1200,1,SEED);exposure=[rows[i]['id'] for i in schedule]
    consumed=[rows[i] for i in schedule[:600]];enc=[encode_row(r,tokenizer,1217) for r in consumed]
    return dict(status='passed',structural=structural,token_audit=tokens,reused=reused,
        consumed=dict(rows=600,optimizer_updates=75,actions=dict(Counter(json.loads(r['messages'][-1]['content'])['action'] for r in consumed)),
                      input_tokens=sum(o for t,o in enc),answer_tokens=sum(len(t)-o for t,o in enc),
                      ids_sha256=hashlib.sha256(json.dumps(exposure[:600]).encode()).hexdigest(),
                      schedule_sha256=hashlib.sha256(json.dumps(exposure).encode()).hexdigest()))


def freeze(root,parent,source):
    update_state(root,stage='audit',status='running',message='核验原始数据、旧输出、模型身份、监督 token 与审查记录。')
    audit=audit_inputs(parent,source);p=read_json(parent/'frozen.json');sources=source_hashes(source)
    review=read_json(root/'implementation-review.json')
    if review.get('status')!='passed' or review.get('source_hashes')!=sources:raise ValueError('Missing/stale implementation review')
    parent_files={n:sha(parent/n) for n in ['frozen.json','completed.json']}
    for spec in reuse_specs(p):
        for suffix in ['jsonl','meta.json','metrics.json']:
            name='eval/dev-'+spec['old_name']+'.'+suffix;parent_files[name]=sha(parent/name)
    write_json(root/'input-audit.json',audit)
    frozen=dict(parent_run=str(parent),parent_files=parent_files,dataset=p['dataset'],snapshot=p['snapshot'],old_run=p['old_run'],
        historical_fingerprint=read_json(Path(p['old_run'])/'frozen.json')['fingerprint'],old_files=p['old_files'],model_files=p['model_files'],
        dataset_files={n:e['sha256'] for n,e in p['manifest']['files'].items()},manifest_sha256=sha(Path(p['dataset'])/'manifest.json'),
        sources=sources,versions={d.metadata['Name']:d.version for d in distributions()},
        protocol_files={str(f.relative_to(source)):sha(f) for f in [source/'docs/superpowers/specs/2026-11-14-training-mechanism-design.md',source/'docs/superpowers/plans/2026-11-14-training-mechanism.md']},
        input_audit_sha256=sha(root/'input-audit.json'),implementation_review_sha256=sha(root/'implementation-review.json'),
        token_audit=audit['token_audit'],consumed=audit['consumed'],reuse=reuse_specs(p),
        protocol=dict(arms={'M0':['sample',False],'M1':['token',False],'M2':['sample',True]},learning_rate=1e-4,microsteps=600,
                      checkpoints=list(STEPS),primary_microstep=600,seed=SEED,max_output_tokens=192,temperature=0,thinking=False,
                      joint_goal='All capability gates AND Q-M0 >= 4 AND F < M0 at 75 updates'))
    frozen['fingerprint']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    if (root/'frozen.json').exists() and read_json(root/'frozen.json')!=frozen:raise ValueError('Cannot overwrite frozen mechanism run')
    write_json(root/'frozen.json',frozen)
    for relative in [*sources,*frozen['protocol_files']]:
        dst=root/'source-snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,dst)
    update_state(root,stage='freeze',status='pending',audit=dict(status='passed',consumed=audit['consumed'],reused_rows=840),message='审计与冻结完成，准备原配方数值复现。')
    return frozen


def verify(root,source,frozen,full=True):
    check=dict(frozen);expected=check.pop('fingerprint')
    if hashlib.sha256(json.dumps(check,sort_keys=True).encode()).hexdigest()!=expected:raise ValueError('Fingerprint corrupt')
    if source_hashes(source)!=frozen['sources']:raise ValueError('Source changed after freeze')
    verify_file_map(source,frozen['protocol_files']);verify_reused(frozen)
    if {d.metadata['Name']:d.version for d in distributions()}!=frozen['versions']:raise ValueError('Dependencies changed')
    data=Path(frozen['dataset']);verify_file_map(data,frozen['dataset_files'])
    if sha(data/'manifest.json')!=frozen['manifest_sha256']:raise ValueError('Manifest changed')
    for name in ['input-audit','implementation-review']:
        if sha(root/(name+'.json'))!=frozen[name.replace('-','_')+'_sha256']:raise ValueError('Audit changed')
    if full:
        verify_file_map(Path(frozen['snapshot']),frozen['model_files'])
        verify_file_map(Path(frozen['old_run']),frozen['old_files'])


def verify_training(root,frozen,arm):
    path=root/('control-certification' if arm=='M0' else 'train-'+arm);limit=200 if arm=='M0' else 600
    config=recipe(arm,limit,frozen['fingerprint'],sha(Path(frozen['dataset'])/'D0-sft.jsonl'))
    validate_config(read_json(path/'config.json'),config)
    for step in config['checkpoints']:verify_checkpoint(path/f'step-{step:04d}',config,step)
    summary=read_json(path/'summary.json')
    expected=dict(status='complete',arm=arm,microsteps=limit,optimizer_updates=limit//8,source_fingerprint=frozen['fingerprint'],
                  normalization=config['normalization'],bias_correction=config['bias_correction'],learning_rate=1e-4)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Training summary identity mismatch')
    ids=[r['id'] for r in read_rows(Path(frozen['dataset'])/'D0-sft.jsonl')]
    exposure=[ids[i] for i in exposure_schedule(1200,1,SEED)]
    if read_json(path/'exposure.json')!=exposure:raise ValueError('Training exposure changed')
    if summary['consumed_ids_sha256']!=hashlib.sha256(json.dumps(exposure[:limit]).encode()).hexdigest():raise ValueError('Consumed IDs differ')
    events=read_rows(path/'events.jsonl')
    if [e['microstep'] for e in events]!=list(range(8,limit+1,8)):raise ValueError('Missing/duplicate training events')
    if arm!='M0' and (summary['total_answer_tokens'],summary['total_input_tokens'])!=(frozen['consumed']['answer_tokens'],frozen['consumed']['input_tokens']):raise ValueError('Consumed token audit differs')
    return summary


def certify(root,frozen,resources):
    out=root/'control-certification'
    train_variant(Path(frozen['snapshot']),Path(frozen['dataset'])/'D0-sft.jsonl',out,arm='M0',limit=200,
                  fingerprint=frozen['fingerprint'],progress_root=root,resources=resources)
    verify_training(root,frozen,'M0');report=certify_control(out,Path(frozen['old_run'])/'train-B')
    report.update(fingerprint=frozen['fingerprint'],exposure_sha256=sha(out/'exposure.json'))
    write_json(root/'control-equivalence.json',report);return report


def reused_rows(frozen,cases):
    verify_reused(frozen);parent=Path(frozen['parent_run']);p=read_json(parent/'frozen.json');refs={};outputs={}
    for spec in frozen['reuse']:
        adapter=Path(spec['adapter']) if spec['adapter'] else None
        rows=validate_evaluation(parent/'eval'/('dev-'+spec['old_name']+'.jsonl'),cases,
                                 eval_meta(p,spec['old_name'],'dev',spec['prompt'],adapter,cases))
        if spec['name'].startswith('M0-'):outputs['M0',int(spec['name'].split('-')[1])]=rows
        else:refs[spec['name']]=rows
    return outputs,refs


def load_available(root,frozen):
    cases=read_rows(Path(frozen['dataset'])/'dev.jsonl');outputs,refs=reused_rows(frozen,cases);partial=[]
    certificate=root/'control-equivalence.json'
    certified=certificate.exists() and read_json(certificate).get('status')=='passed'
    if certified:
        if read_json(certificate).get('fingerprint')!=frozen['fingerprint']:raise ValueError('Stale control certificate')
        verify_training(root,frozen,'M0');certify_control(root/'control-certification',Path(frozen['old_run'])/'train-B')
    else:outputs={}
    for arm in ['M1','M2']:
        for s in STEPS:
            name=f'{arm}-{s}';path=root/'eval'/('dev-'+name+'.jsonl')
            if not path.exists():continue
            config=recipe(arm,600,frozen['fingerprint'],sha(Path(frozen['dataset'])/'D0-sft.jsonl'))
            adapter=root/f'train-{arm}/step-{s:04d}';verify_checkpoint(adapter,config,s)
            rows=validate_evaluation(path,cases,eval_meta(frozen,name,'dev','v2',adapter,cases),partial=True)
            if len(rows)==len(cases):
                rescore(cases,rows,path,repair_missing_metrics=True);outputs[arm,s]=rows
            else:partial.append(dict(name=name,completed_rows=len(rows),required_rows=len(cases)))
    return cases,outputs,refs,partial


def evaluate(root,source,frozen,resources,cases,arm,step):
    import mlx.core as mx
    from mlx_lm import load
    from .evaluate_formal import evaluate_model
    verify(root,source,frozen,full=False);verify_training(root,frozen,arm);resources.check()
    adapter=root/f'train-{arm}/step-{step:04d}';name=f'{arm}-{step}'
    path=root/'eval'/('dev-'+name+'.jsonl');meta=eval_meta(frozen,name,'dev','v2',adapter,cases)
    if path.exists():
        rows=validate_evaluation(path,cases,meta,partial=True)
        if len(rows)==len(cases):rescore(cases,rows,path,repair_missing_metrics=True);return rows
    update_state(root,stage='development',status='running',detail=name,step=0,total_steps=120,message='逐题自由生成，不重试、不修复答案。')
    model,tokenizer=load(frozen['snapshot'],adapter_path=str(adapter))
    try:
        resources.check();evaluate_model(model,tokenizer,cases,path,meta,root,'development',resources);resources.check()
    finally:
        del model,tokenizer;mx.clear_cache()
    verify_training(root,frozen,arm);rows=validate_evaluation(path,cases,meta);rescore(cases,rows,path);return rows


def finish_report(root,source,frozen,resources=None,complete=True,reason=None):
    from .report_mechanism import write_report,examples
    if resources:finalize_resources(resources)
    verify(root,source,frozen)
    cases,outputs,refs,partial=load_available(root,frozen);analysis=analyze(cases,outputs,refs,complete=complete)
    if complete:
        if read_json(root/'control-equivalence.json')['status']!='passed':raise ValueError('Control not certified')
        verify_training(root,frozen,'M0');certify_control(root/'control-certification',Path(frozen['old_run'])/'train-B')
        for a in ['M1','M2']:verify_training(root,frozen,a)
    checked=[]
    for spec in frozen['reuse']:
        if spec['name'].startswith('M0-') and ('M0',int(spec['name'].split('-')[1])) not in outputs:continue
        rows=outputs['M0',int(spec['name'].split('-')[1])] if spec['name'].startswith('M0-') else refs[spec['name']]
        checked.append(dict(origin='reused',**rescore(cases,rows,Path(frozen['parent_run'])/'eval'/('dev-'+spec['old_name']+'.jsonl'))))
    for (arm,s),rows in outputs.items():
        if arm!='M0':checked.append(dict(origin='new',**rescore(cases,rows,root/'eval'/f'dev-{arm}-{s}.jsonl')))
    counts={origin:sum(x['rows'] for x in checked if x['origin']==origin) for origin in ['reused','new']}
    if complete and counts!={'reused':840,'new':720}:raise ValueError('Wrong final evaluation budget')
    summaries=[read_json(p) for p in sorted(root.glob('resource-summary-*.json'))]
    if resources:summaries.append(resources.summary())
    if complete and (not summaries or summaries[-1]['error']):raise ValueError('No clean final resource summary')
    analysis.update(reason=reason,fingerprint=frozen['fingerprint'],partial_evaluations=partial,
        cases=examples(cases,{a:outputs[a,600] for a in ARMS if (a,600) in outputs}),token_audit=frozen['token_audit'],consumed=frozen['consumed'],
        training={a:verify_training(root,frozen,a) for a in ['M1','M2'] if (root/f'train-{a}/summary.json').exists()},
        certification=read_json(root/'control-equivalence.json') if (root/'control-equivalence.json').exists() else dict(status='pending'),
        resource_summaries=summaries,verification=dict(status='passed' if complete else 'completed_rows_verified',evaluations=checked,counts=counts,
            total_generations=sum(counts.values()),new_generation_seconds=sum(x['generation_seconds'] for x in checked if x['origin']=='new')))
    payload=write_report(analysis,root,source)
    if complete:
        write_json(root/'completed.json',dict(status='complete',fingerprint=frozen['fingerprint'],finished_at=time.time(),report_sha256=sha(root/'report-data.json')))
        update_state(root,stage='complete',status='complete',finished_at=time.time(),results_ready=True,message='机制对照完成，结果已重评分；请查看固定75更新比较及能力门槛。')
    else:
        (root/'completed.json').unlink(missing_ok=True)
        update_state(root,status='failed',results_ready=True,message='实验未完成，仅展示已完成配置：'+str(reason))
    return payload


def run(root,source,frozen,resources):
    import mlx.core as mx
    configure_mlx_memory(mx);certify(root,frozen,resources)
    cases=read_rows(Path(frozen['dataset'])/'dev.jsonl');outputs,refs=reused_rows(frozen,cases)
    for name,rows in refs.items():record(root,cases,rows,name+'（复用）')
    for (arm,step),rows in outputs.items():record(root,cases,rows,f'{arm} · {step//8} 更新（复用）')
    for arm in ['M1','M2']:
        verify(root,source,frozen)
        train_variant(Path(frozen['snapshot']),Path(frozen['dataset'])/'D0-sft.jsonl',root/('train-'+arm),arm=arm,limit=600,
                      fingerprint=frozen['fingerprint'],progress_root=root,resources=resources)
        verify_training(root,frozen,arm)
        for step in STEPS:record(root,cases,evaluate(root,source,frozen,resources,cases,arm,step),f'{arm} · {step//8} 更新')
    update_state(root,stage='reporting',step=0,total_steps=0,message='重评分全部1560条输出，核对来源和75更新的联合目标。')
    finish_report(root,source,frozen,resources)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--stage',choices=['freeze','run','report'],required=True);args=p.parse_args()
    source=Path(__file__).resolve().parents[1];runtime=runtime_root(source,os.environ.get('TOOLCALL_RUNTIME_DIR'));configure_runtime(runtime)
    root=Path(args.run_root).resolve();parent=Path(args.parent_run).resolve()
    if not root.is_relative_to(runtime/'runs') or not parent.is_relative_to(runtime/'runs') or root==parent:p.error('Use separate external runtime paths')
    root.mkdir(parents=True,exist_ok=True)
    initially_complete=(root/'completed.json').exists()
    import fcntl
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(root/'worker.pid').write_text(str(os.getpid()))
    beat=Heartbeat(root);beat.start();resources=None
    try:
        if args.stage=='freeze':frozen=freeze(root,parent,source)
        else:
            frozen=read_json(root/'frozen.json')
            if frozen['parent_run']!=str(parent):raise ValueError('Different parent requested')
            verify(root,source,frozen)
            if args.stage=='run':
                if initially_complete:
                    print(json.dumps(dict(stage=args.stage,status='already_complete',fingerprint=frozen['fingerprint'])),flush=True)
                    return
                resources=ResourceMonitor(root/f'resources-{int(time.time())}.jsonl');resources.start()
                baseline=root/'resource-baseline.json'
                if baseline.exists():
                    initial=read_json(baseline);current=resources.samples[0];resources.samples.insert(0,initial)
                    violation=resource_violation(current,initial['swap_used_bytes'])
                    if violation:resources.error=violation
                else:write_json(baseline,resources.samples[0])
                resources.check();run(root,source,frozen,resources)
            else:finish_report(root,source,frozen,complete=(root/'completed.json').exists(),reason='Requested report before completion')
        print(json.dumps(dict(stage=args.stage,status='stage_finished',fingerprint=frozen['fingerprint'])),flush=True)
    except BaseException as exc:
        if initially_complete:raise
        write_json(root/'last-error.json',dict(error=repr(exc),stage=args.stage,time=time.time()))
        if resources:
            if resources.thread.is_alive():resources.stop()
            write_json(root/f'resource-summary-{int(time.time())}.json',resources.summary())
        update_state(root,status='failed',message='流程停止：'+repr(exc))
        if 'frozen' in locals() and args.stage!='freeze':
            try:finish_report(root,source,frozen,complete=False,reason=repr(exc))
            except Exception as error:write_json(root/'partial-report-error.json',dict(error=repr(error)))
        raise
    finally:
        beat.stop()
        if resources:
            if resources.thread.is_alive():resources.stop()
            write_json(root/f'resource-summary-{int(time.time())}.json',resources.summary())
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
