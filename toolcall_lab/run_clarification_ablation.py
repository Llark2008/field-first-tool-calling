"""Frozen local four-cell development experiment; never opens a new final test."""
import argparse
import hashlib
import json
import os
import shutil
import time
from importlib.metadata import distributions,version
from pathlib import Path

from .ablation_data import audit_candidate,read_rows
from .analyze_clarification_ablation import ARMS,REFERENCES,analyze_pilot,analyze_available,metrics_for,regression_decision,historical_complete_counts
from .formal_data import sha
from .live_progress import Heartbeat,read_state,update_state
from .resource_monitor import ResourceMonitor,resource_violation
from .run import configure_runtime,runtime_root
from .train_formal import MODEL_REVISION,configure_mlx_memory,exposure_schedule
from .train_ablation import CHECKPOINTS,train_variant,certify_control


def read_json(p):return json.loads(Path(p).read_text())


def write_json(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)


def verify_file_map(root,hashes):
    for name,digest in hashes.items():
        if sha(Path(root)/name)!=digest:raise ValueError('Frozen file changed: '+name)


def verify_historical_model(snapshot,historical):
    verify_file_map(snapshot,historical['model_files'])


def verify_checkpoint_bundle(path,fingerprint,microstep):
    state=read_json(Path(path)/'state.json')
    if (state['fingerprint'],state['microstep'],state['optimizer_updates'])!=(fingerprint,microstep,microstep//8):raise ValueError('Historical checkpoint identity/cursor mismatch')
    if set(state['hashes'])!={'adapters.safetensors','optimizer.safetensors','rng.safetensors'}:raise ValueError('Historical checkpoint bundle incomplete')
    verify_file_map(path,state['hashes'])


def finalize_resources(resources):
    resources.stop()
    resources.check()


def validate_evaluation(path,cases,metadata,partial=False):
    path=Path(path)
    if read_json(path.with_suffix('.meta.json'))!=metadata:raise ValueError('Evaluation metadata mismatch')
    rows=read_rows(path)
    if len(rows)>len(cases) or (not partial and len(rows)!=len(cases)):raise ValueError('Incomplete/extra evaluation rows')
    if [r['case_id'] for r in rows]!=[c['id'] for c in cases[:len(rows)]]:raise ValueError('Evaluation ID alignment mismatch')
    for r,c in zip(rows,cases):
        for k in ['gold','condition']:
            if r[k]!=c[k]:raise ValueError('Evaluation labels differ')
        if r.get('template_group')!=c.get('template_group',c.get('group_id')):raise ValueError('Evaluation families differ')
    return rows


def token_audit(dataset,snapshot):
    from mlx_lm.utils import load_tokenizer
    from .training_spike import encode_row
    from .controls import build_control_messages
    tokenizer=load_tokenizer(snapshot);encoded={k:[encode_row(r,tokenizer,1217) for r in read_rows(dataset/(k+'-sft.jsonl'))] for k in ['D0','D1']}
    if [t[o:] for t,o in encoded['D0']]!=[t[o:] for t,o in encoded['D1']]:raise ValueError('Supervised targets changed')
    result=dict(status='passed',sequence_cap=1217,identical_supervised_targets=1200,training={},development={})
    for k,enc in encoded.items():result['training'][k]=dict(n=len(enc),max_tokens=max(len(t) for t,o in enc),total_input_tokens=sum(o for t,o in enc),total_answer_tokens=sum(len(t)-o for t,o in enc))
    for prompt in ['v2','no_examples','checklist']:
        lengths=[len(tokenizer.apply_chat_template(build_control_messages(c,prompt),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)) for c in read_rows(dataset/'dev.jsonl')]
        if max(lengths)>2048:raise ValueError('Development input too long')
        result['development'][prompt]=dict(n=len(lengths),max_input_tokens=max(lengths))
    return result


def freeze(root,dataset,snapshot,source,old):
    audit=audit_candidate(dataset);historical=read_json(old/'frozen.json')
    verify_file_map(source/'toolcall_lab',historical['sources'])
    verify_historical_model(snapshot,historical)
    for n,v in historical['versions'].items():
        if version(n)!=v:raise ValueError('Historical dependency changed: '+n)
    review=read_json(root/'audit-review.json');implementation=read_json(root/'implementation-review.json')
    if review.get('status')!='passed' or review.get('dataset_manifest_sha256')!=sha(dataset/'manifest.json'):raise ValueError('Missing/stale semantic audit')
    if implementation.get('status')!='passed':raise ValueError('Implementation review missing')
    tokens=token_audit(dataset,snapshot);write_json(root/'token-audit.json',tokens)
    sources={str(p.relative_to(source)):sha(p) for folder in ['toolcall_lab','tests'] for p in (source/folder).glob('*.py')}
    if implementation.get('source_hashes')!=sources:raise ValueError('Implementation review is stale')
    old_files={'frozen.json':sha(old/'frozen.json')}
    for arm,steps in [('B',CHECKPOINTS),('A',(600,))]:
        for step in steps:
            cp=old/f'train-{arm}/step-{step:04d}'
            verify_checkpoint_bundle(cp,historical['fingerprint'],step)
            for p in cp.iterdir():
                if p.is_file():old_files[str(p.relative_to(old))]=sha(p)
    old_files['train-B/config.json']=sha(old/'train-B/config.json')
    old_files['train-B/events.jsonl']=sha(old/'train-B/events.jsonl')
    frozen=dict(dataset=str(dataset),old_run=str(old),snapshot=str(snapshot),manifest=read_json(dataset/'manifest.json'),
        sources=sources,model_files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},old_files=old_files,
        versions={d.metadata['Name']:d.version for d in distributions()},historical_versions=historical['versions'],
        audit_review_sha256=sha(root/'audit-review.json'),implementation_review_sha256=sha(root/'implementation-review.json'),
        token_audit=tokens,token_audit_sha256=sha(root/'token-audit.json'),
        protocol_files={str(p.relative_to(source)):sha(p) for p in [source/'docs/superpowers/specs/2026-11-14-clarification-ablation-design.md',source/'docs/superpowers/plans/2026-11-14-clarification-ablation.md']},
        protocol=dict(rates={'R0':1e-4,'R1':5e-5,'R2':1e-4,'R3':5e-5},data={'R0':'D0','R1':'D0','R2':'D1','R3':'D1'},
            microsteps=1200,checkpoints=list(CHECKPOINTS),primary_microstep=600,seed=20261113,optimizer='MLX Adam bias_correction=False',
            max_output_tokens=192,temperature=0,thinking=False,selection='fixed engineering gates; no runner-up regression search'))
    frozen['fingerprint']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    if (root/'frozen.json').exists() and read_json(root/'frozen.json')!=frozen:raise ValueError('Cannot overwrite frozen experiment')
    write_json(root/'frozen.json',frozen)
    for relative in sources:
        dst=root/'source-snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,dst)
    update_state(root,stage='freeze',status='pending',audit=dict(structural=audit,semantic=review['status'],token=tokens),message='数据、模型、代码、依赖和比较规则已冻结；准备历史对照复现核验。')
    return frozen


def verify(root,source,frozen,full=True):
    check=dict(frozen);expected=check.pop('fingerprint')
    if hashlib.sha256(json.dumps(check,sort_keys=True).encode()).hexdigest()!=expected:raise ValueError('Fingerprint corrupt')
    verify_file_map(source,frozen['sources']);verify_file_map(source,frozen['protocol_files'])
    current={d.metadata['Name']:d.version for d in distributions()}
    if current!=frozen['versions']:raise ValueError('Frozen runtime dependencies changed')
    verify_file_map(Path(frozen['dataset']),{n:e['sha256'] for n,e in frozen['manifest']['files'].items()})
    if sha(Path(frozen['dataset'])/'manifest.json')!=read_json(root/'audit-review.json')['dataset_manifest_sha256']:raise ValueError('Manifest changed')
    for name in ['audit-review','implementation-review','token-audit']:
        if sha(root/(name+'.json'))!=frozen[name.replace('-','_')+'_sha256']:raise ValueError('Frozen audit changed')
    if full:
        verify_file_map(Path(frozen['snapshot']),frozen['model_files']);verify_file_map(Path(frozen['old_run']),frozen['old_files'])


def certify(root,frozen,resources):
    dataset=Path(frozen['dataset']);old=Path(frozen['old_run']);out=root/'control-certification'
    if (root/'control-equivalence.json').exists():
        report=read_json(root/'control-equivalence.json')
        if report['fingerprint']!=frozen['fingerprint'] or report['status']!='passed':raise ValueError('Invalid control certificate')
        certify_control(out,old/'train-B');return report
    train_variant(Path(frozen['snapshot']),dataset/'D0-sft.jsonl',out,learning_rate=1e-4,max_microsteps=200,
                  checkpoints=(200,),fingerprint=frozen['fingerprint'],progress_root=root,resources=resources)
    report=certify_control(out,old/'train-B')
    ids=[r['id'] for r in read_rows(dataset/'D0-sft.jsonl')];expected=[ids[i] for i in exposure_schedule(1200,1,20261113)]
    if read_json(out/'exposure.json')!=expected:raise ValueError('Control schedule differs')
    historical_dataset=Path(read_json(old/'frozen.json')['dataset'])
    if sha(dataset/'D0-sft.jsonl')!=sha(historical_dataset/'B-sft.jsonl'):raise ValueError('Control training input differs')
    report.update(fingerprint=frozen['fingerprint'],exposure_sha256=sha(out/'exposure.json'),historical_frozen_sha256=sha(old/'frozen.json'))
    write_json(root/'control-equivalence.json',report);return report


def adapter_for(root,frozen,arm,step):
    return Path(frozen['old_run'])/f'train-B/step-{step:04d}' if arm=='R0' else root/f'train-{arm}/step-{step:04d}'


def eval_meta(frozen,name,split,prompt,adapter,cases):
    return dict(name=name,split=split,prompt=prompt,dataset_fingerprint=frozen['fingerprint'],
        cases_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        adapter_sha256=sha(adapter/'adapters.safetensors') if adapter else None,
        max_output_tokens=192,temperature=0,thinking=False)


def evaluate(root,source,frozen,resources,cases,name,split='dev',prompt='v2',adapter=None,stage='development'):
    import mlx.core as mx
    from mlx_lm import load
    from .evaluate_formal import evaluate_model
    verify(root,source,frozen,full=False);resources.check()
    output=root/'eval'/f'{split}-{name}.jsonl';meta=eval_meta(frozen,name,split,prompt,adapter,cases)
    if output.exists():
        rows=validate_evaluation(output,cases,meta,partial=True)
        if len(rows)==len(cases):return rows
    update_state(root,stage=stage,status='running',detail=name,step=0,total_steps=len(cases),message='加载本配置，逐条自由生成；无重试或答案修复。')
    model,tokenizer=load(str(frozen['snapshot']),adapter_path=str(adapter) if adapter else None)
    try:
        resources.check();evaluate_model(model,tokenizer,cases,output,meta,root,stage,resources)
        resources.check()
    finally:
        del model,tokenizer;mx.clear_cache()
    return validate_evaluation(output,cases,meta)


def record(root,cases,rows,name):
    m=metrics_for(cases,rows);old=[c for c in read_state(root).get('checkpoints',[]) if c['name']!=name]
    old.append(dict(name=name,**{k:f"{m[k]['count']}/{m[k]['n']}" for k in ['C','Q','F','J']},status='已评测'))
    update_state(root,checkpoints=old)


def load_all(root,frozen):
    cases=read_rows(Path(frozen['dataset'])/'dev.jsonl');refs={};outputs={}
    for name in REFERENCES:
        prompt=name.removeprefix('base_') if name.startswith('base_') else 'v2'
        adapter=Path(frozen['old_run'])/'train-A/step-0600' if name=='A75' else None
        refs[name]=validate_evaluation(root/'eval'/f'dev-{name}.jsonl',cases,eval_meta(frozen,name,'dev',prompt,adapter,cases))
    for arm in ARMS:
        for step in CHECKPOINTS:
            name=f'{arm}-{step}';adapter=adapter_for(root,frozen,arm,step)
            outputs[arm,step]=validate_evaluation(root/'eval'/f'dev-{name}.jsonl',cases,eval_meta(frozen,name,'dev','v2',adapter,cases))
    return cases,outputs,refs


def run(root,source,frozen,resources):
    import mlx.core as mx
    configure_mlx_memory(mx);certify(root,frozen,resources)
    cases=read_rows(Path(frozen['dataset'])/'dev.jsonl')
    for name in REFERENCES:
        prompt=name.removeprefix('base_') if name.startswith('base_') else 'v2'
        adapter=Path(frozen['old_run'])/'train-A/step-0600' if name=='A75' else None
        rows=evaluate(root,source,frozen,resources,cases,name,prompt=prompt,adapter=adapter,stage='references');record(root,cases,rows,name)
    for arm in ARMS:
        if arm!='R0':
            verify(root,source,frozen,full=False)
            train_variant(Path(frozen['snapshot']),Path(frozen['dataset'])/(frozen['protocol']['data'][arm]+'-sft.jsonl'),root/('train-'+arm),
                learning_rate=frozen['protocol']['rates'][arm],max_microsteps=1200,checkpoints=CHECKPOINTS,
                fingerprint=frozen['fingerprint'],progress_root=root,resources=resources)
        for step in CHECKPOINTS:
            rows=evaluate(root,source,frozen,resources,cases,f'{arm}-{step}',adapter=adapter_for(root,frozen,arm,step));record(root,cases,rows,f'{arm} · {step//8} 更新')
    cases,outputs,refs=load_all(root,frozen)
    update_state(root,stage='selection',step=0,total_steps=0,message='按冻结能力门槛选择候选；主因素比较固定为 75 次更新。')
    analysis=analyze_pilot(cases,outputs,refs);write_json(root/'selection.json',analysis['selection'])
    selected=analysis['selection']['selected']
    if selected and selected['arm']!='R0':
        from .expanded import expanded_cases
        from .evaluate_formal import summarize
        historical=expanded_cases();arm=selected['arm'];step=selected['microstep'];hist={}
        for name in [arm,'R0']:
            hist[name]=evaluate(root,source,frozen,resources,historical,f'{name}-{step}',split='historical',adapter=adapter_for(root,frozen,name,step),stage='regression')
        summaries={k:summarize(v) for k,v in hist.items()}
        counts=historical_complete_counts(summaries[arm])
        regression=dict(status='evaluated',candidate=arm,microstep=step,metrics=summaries,decision=regression_decision(counts))
    else:regression=dict(status='not_run',reason='No eligible new candidate' if not selected else 'Historical R0 selected')
    write_json(root/'regression.json',regression)
    finish_report(root,source,frozen,resources)


def finish_report(root,source,frozen,resources=None):
    from .report_clarification_ablation import write_pilot_report,verify_outputs
    if resources:finalize_resources(resources)
    cases,outputs,refs=load_all(root,frozen);analysis=analyze_pilot(cases,outputs,refs)
    analysis['regression']=read_json(root/'regression.json')
    analysis['fingerprint']=frozen['fingerprint']
    analysis['training']={a:read_json(root/f'train-{a}/summary.json') for a in ['R1','R2','R3']}
    analysis['certification']=read_json(root/'control-equivalence.json')
    analysis['resource_summaries']=[read_json(p) for p in sorted(root.glob('resource-summary-*.json'))]
    if resources:analysis['resource_summaries'].append(resources.summary())
    update_state(root,stage='reporting',step=0,total_steps=0,message='重新评分原始输出并生成四组对比、曲线与案例。')
    verification=verify_outputs(root,frozen,cases,outputs,refs)
    analysis['verification']=verification
    payload=write_pilot_report(analysis,root,source)
    write_json(root/'completed.json',dict(status='complete',fingerprint=frozen['fingerprint'],finished_at=time.time()))
    update_state(root,stage='complete',status='complete',finished_at=time.time(),results_ready=True,message='本轮开发实验完成；因素对比、能力门槛与具体案例已整理。')
    return payload


def finish_partial_report(root,source,frozen,reason):
    from .report_clarification_ablation import write_pilot_report,verify_outputs
    verify(root,source,frozen,full=False)
    cases=read_rows(Path(frozen['dataset'])/'dev.jsonl');outputs={};refs={};partial=[]
    for arm,step in [(a,s) for a in ARMS for s in CHECKPOINTS]+[(n,None) for n in REFERENCES]:
        name=f'{arm}-{step}' if step else arm
        prompt=arm.removeprefix('base_') if arm.startswith('base_') else 'v2'
        adapter=adapter_for(root,frozen,arm,step) if step else Path(frozen['old_run'])/'train-A/step-0600' if arm=='A75' else None
        path=root/'eval'/f'dev-{name}.jsonl'
        if not path.exists():continue
        rows=validate_evaluation(path,cases,eval_meta(frozen,name,'dev',prompt,adapter,cases),partial=True)
        if len(rows)==len(cases):
            if step:outputs[arm,step]=rows
            else:refs[arm]=rows
        else:partial.append(dict(name=name,completed_rows=len(rows),required_rows=len(cases)))
    analysis=analyze_available(cases,outputs,refs)
    analysis.update(reason=reason,fingerprint=frozen['fingerprint'],partial_evaluations=partial,
        regression=dict(status='not_run',reason='Experiment incomplete; no promotion'),
        training={a:read_json(root/f'train-{a}/summary.json') for a in ['R1','R2','R3'] if (root/f'train-{a}/summary.json').exists()},
        certification=read_json(root/'control-equivalence.json') if (root/'control-equivalence.json').exists() else {'status':'not_completed'},
        resource_summaries=[read_json(p) for p in sorted(root.glob('resource-summary-*.json'))])
    analysis['verification']=verify_outputs(root,frozen,cases,outputs,refs,partial=True)
    payload=write_pilot_report(analysis,root,source,loaded=(cases,outputs,refs))
    (root/'completed.json').unlink(missing_ok=True)
    update_state(root,status='failed',results_ready=True,message='实验未完成：'+reason+'；已生成仅含已完成配置的描述性报告。')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--stage',choices=['freeze','certify','run','report'],required=True);args=p.parse_args()
    source=Path(__file__).resolve().parents[1];runtime=runtime_root(source,os.environ.get('TOOLCALL_RUNTIME_DIR'));configure_runtime(runtime)
    root=Path(args.run_root).resolve();dataset=Path(args.dataset).resolve()
    if not root.is_relative_to(runtime/'runs') or not dataset.is_relative_to(runtime/'datasets'):p.error('Use external runtime paths')
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(root/'worker.pid').write_text(str(os.getpid()))
    beat=Heartbeat(root);beat.start();resources=None
    try:
        if args.stage=='freeze':
            snapshot=runtime/'cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots'/MODEL_REVISION
            frozen=freeze(root,dataset,snapshot,source,runtime/'runs/formal-20261113')
        else:
            frozen=read_json(root/'frozen.json')
            if frozen['dataset']!=str(dataset):raise ValueError('Different dataset requested')
            verify(root,source,frozen)
            if args.stage in ('certify','run'):
                resources=ResourceMonitor(root/f'resources-{args.stage}-{int(time.time())}.jsonl');resources.start()
                baseline=root/'resource-baseline.json'
                if baseline.exists():
                    initial=read_json(baseline);current=resources.samples[0];resources.samples.insert(0,initial)
                    violation=resource_violation(current,initial['swap_used_bytes'])
                    if violation:resources.error=violation
                else:write_json(baseline,resources.samples[0])
                resources.check()
            if args.stage=='certify':
                certify(root,frozen,resources);finalize_resources(resources);update_state(root,stage='certify',status='pending',message='历史对照核验通过，准备正式四组对照。')
            elif args.stage=='run':run(root,source,frozen,resources)
            else:
                if (root/'regression.json').exists():finish_report(root,source,frozen)
                else:finish_partial_report(root,source,frozen,'Requested report before experiment completion')
        print(json.dumps(dict(stage=args.stage,status='stage_finished',fingerprint=frozen['fingerprint'])),flush=True)
    except BaseException as exc:
        write_json(root/'last-error.json',dict(error=repr(exc),stage=args.stage,time=time.time(),resources=resources.summary() if resources else None))
        update_state(root,status='failed',message='流程停止：'+repr(exc))
        if resources:
            if resources.thread.is_alive():resources.stop()
            write_json(root/f'resource-summary-{args.stage}-{int(time.time())}.json',resources.summary())
        if 'frozen' in locals() and args.stage!='freeze':
            try:finish_partial_report(root,source,frozen,repr(exc))
            except Exception as report_error:write_json(root/'partial-report-error.json',dict(error=repr(report_error),original_error=repr(exc)))
        raise
    finally:
        beat.stop()
        if resources:
            if resources.thread.is_alive():resources.stop()
            write_json(root/f'resource-summary-{args.stage}-{int(time.time())}.json',resources.summary())
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
