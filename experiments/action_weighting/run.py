"""Audit and execute the predeclared W20 pilot; runtime is outside sync."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from toolcall_lab.ablation_data import read_rows
from toolcall_lab.analyze_clarification_ablation import metrics_for
from toolcall_lab.evaluate_formal import evaluate_model
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import Heartbeat,update_state
from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
from toolcall_lab.run_clarification_ablation import (read_json as read,write_json,verify_file_map,
    eval_meta,validate_evaluation,verify_checkpoint_bundle)
from toolcall_lab.run_mechanism import verify as verify_parent,reused_rows,rescore,verify_training as verify_old_training
from toolcall_lab.train_ablation import certify_control
from toolcall_lab.train_formal import SEED,exposure_schedule,configure_mlx_memory
from toolcall_lab.training_spike import encode_row
from scripts.run_train_replay import select_seen,teacher_forced,metrics as replay_metrics,MONITOR_HTML
from .core import recipe,verify_checkpoint,action_vocabulary,decision_index,train
from . import report

SOURCE=Path(__file__).resolve().parents[2]
RUNTIME=Path.home()/'.local/share/local-tool-calling'
PARENT=RUNTIME/'runs/mechanism-20261114'
REPLAY=RUNTIME/'runs/train-replay-20261115'
ROOT=RUNTIME/'runs/action-weighting-20261115'


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def sources():
    files=list((SOURCE/'experiments/action_weighting').rglob('*.py'))
    files += [SOURCE/'docs/superpowers'/p for p in ['specs/2026-11-15-action-weighting-design.md','plans/2026-11-15-action-weighting.md']]
    files += [SOURCE/'scripts/run_train_replay.py',SOURCE/'reports/train-replay-protocol.md']
    return {str(p.relative_to(SOURCE)):sha(p) for p in sorted(files)}


def audit(root):
    from mlx_lm.utils import load_tokenizer
    update_state(root,stage='audit',status='running',step=0,total_steps=1200,message='逐条验证动作监督位置和历史对照身份。')
    p=read(PARENT/'frozen.json'); verify_parent(PARENT,SOURCE,p)
    if read(PARENT/'completed.json')['report_sha256']!=sha(PARENT/'report-data.json'): raise ValueError('Parent report changed')
    verify_old_training(PARENT,p,'M0')
    certificate=certify_control(PARENT/'control-certification',Path(p['old_run'])/'train-B')
    if certificate['status']!='passed': raise ValueError('Historical control not equivalent')
    cases,sft,replay_audit=select_seen(p)
    old_audit=read(REPLAY/'audit.json')
    if old_audit['audit']!=replay_audit or read_rows(REPLAY/'cases.jsonl')!=cases: raise ValueError('Replay selection changed')
    if old_audit['parent_fingerprint']!=p['fingerprint'] or old_audit['script_sha256']!=sha(SOURCE/'scripts/run_train_replay.py') or old_audit['protocol_sha256']!=sha(SOURCE/'reports/train-replay-protocol.md'): raise ValueError('Replay source identity changed')
    if read(REPLAY/'completed.json')['results_sha256']!=sha(REPLAY/'results.json'): raise ValueError('Replay results changed')
    verify_file_map(REPLAY,read(REPLAY/'results.json')['artifacts'])
    m0=Path(p['old_run'])/'train-B/step-0600'
    if old_audit['adapters']['M0']!={'path':str(m0),'sha256':sha(m0/'adapters.safetensors')}: raise ValueError('Replay control mismatch')
    for step in [200,400,600]: verify_checkpoint_bundle(Path(p['old_run'])/f'train-B/step-{step:04d}',p['historical_fingerprint'],step)
    dev=read_rows(Path(p['dataset'])/'dev.jsonl'); outputs,_=reused_rows(p,dev)
    for (_,step),rows in outputs.items(): rescore(dev,rows,Path(p['parent_run'])/f'eval/dev-R0-{step}.jsonl')
    replay_free=validate_evaluation(REPLAY/'free-M0.jsonl',cases,eval_meta(p,'M0','seen_train_120','v2',m0,cases))
    rescore(cases,replay_free,REPLAY/'free-M0.jsonl')
    tf=read_rows(REPLAY/'teacher-M0.jsonl')
    if replay_metrics(cases,replay_free,tf)!=read(REPLAY/'metrics-M0.json'): raise ValueError('Replay metrics changed')
    tokenizer=load_tokenizer(Path(p['snapshot'])); prefix,ids=action_vocabulary(tokenizer)
    rows=read_rows(Path(p['dataset'])/'D0-sft.jsonl'); encoded=[]; positions=[]
    for r in rows:
        t,o=encode_row(r,tokenizer,1217); action=json.loads(r['messages'][-1]['content'])['action']
        i=decision_index(t,o,prefix,ids,action)
        if .20<=1/(len(t)-o): raise ValueError('Decision weight does not exceed original share')
        positions.append(dict(id=r['id'],action=action,target_index=i,answer_index=i-o,tokens=len(t)-o))
        encoded.append((t,o))
    schedule=exposure_schedule(1200,1,SEED)[:600]
    consumed=dict(rows=len(schedule),input_tokens=sum(encoded[i][1] for i in schedule),answer_tokens=sum(len(encoded[i][0])-encoded[i][1] for i in schedule),
        actions=dict(Counter(positions[i]['action'] for i in schedule)),ids_sha256=hashlib.sha256(json.dumps([rows[i]['id'] for i in schedule]).encode()).hexdigest())
    for key in consumed:
        if consumed[key]!=p['consumed'][key]: raise ValueError('Consumed exposure audit mismatch: '+key)
    if len(rows)!=1200 or consumed['input_tokens']!=610408 or consumed['answer_tokens']!=14362: raise ValueError('Unexpected data budget')
    result=dict(status='passed',parent_fingerprint=p['fingerprint'],decisions=positions,consumed=consumed,
        replay=replay_audit,prefix_tokens=prefix,action_token_ids=ids,control_certificate=certificate)
    write_json(root/'input-audit.json',result)
    return p,result


def freeze(root):
    review=read(root/'implementation-review.json')
    if review.get('status')!='passed' or review.get('sources')!=sources(): raise ValueError('Missing/stale implementation review')
    p,a=audit(root)
    reused={str(path):sha(path) for path in [PARENT/'frozen.json',PARENT/'completed.json',PARENT/'report-data.json',
        REPLAY/'audit.json',REPLAY/'cases.jsonl',REPLAY/'completed.json',REPLAY/'results.json',REPLAY/'free-M0.jsonl',
        REPLAY/'free-M0.meta.json',REPLAY/'free-M0.metrics.json',REPLAY/'teacher-M0.jsonl',REPLAY/'metrics-M0.json']}
    f=dict(parent_fingerprint=p['fingerprint'],dataset=p['dataset'],snapshot=p['snapshot'],sources=sources(),reused=reused,
        audit_sha256=sha(root/'input-audit.json'),review_sha256=sha(root/'implementation-review.json'),action_weight=.20)
    f['fingerprint']=digest(f)
    if (root/'frozen.json').exists() and read(root/'frozen.json')!=f: raise ValueError('Cannot overwrite frozen run')
    write_json(root/'frozen.json',f)
    for relative in f['sources']:
        target=root/'source-snapshot'/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(SOURCE/relative,target)
    update_state(root,stage='ready',status='pending',audit={k:v for k,v in a.items() if k!='decisions'},message='1200条动作位置审计通过，等待训练。')
    return f


def verify(root,f,full=True):
    check=dict(f); expected=check.pop('fingerprint')
    if digest(check)!=expected or f['sources']!=sources(): raise ValueError('Frozen identity/source changed')
    verify_file_map(Path('/'),f['reused'])
    if sha(root/'input-audit.json')!=f['audit_sha256'] or sha(root/'implementation-review.json')!=f['review_sha256']: raise ValueError('Audit/review changed')
    p=read(PARENT/'frozen.json')
    if p['fingerprint']!=f['parent_fingerprint']: raise ValueError('Parent fingerprint changed')
    verify_parent(PARENT,SOURCE,p,full=full)
    return p


def check_training(root,f):
    out=root/'train-W20'; config=recipe(f['fingerprint'],sha(Path(f['dataset'])/'D0-sft.jsonl'))
    if read(out/'config.json')!=config: raise ValueError('Recipe mismatch')
    for step in [200,400,600]: verify_checkpoint(out/f'step-{step:04d}',config,step)
    result=read(out/'summary.json'); audit_data=read(root/'input-audit.json')['consumed']
    for key,value in dict(status='complete',arm='W20',microsteps=600,optimizer_updates=75,source_fingerprint=f['fingerprint'],
        total_input_tokens=610408,total_answer_tokens=14362,consumed_ids_sha256=audit_data['ids_sha256'],events=75).items():
        if result.get(key)!=value: raise ValueError('Training summary mismatch: '+key)
    events=read_rows(out/'events.jsonl')
    if [e['microstep'] for e in events]!=list(range(8,601,8)): raise ValueError('Training event grid mismatch')
    if sum(e['supervised_tokens'] for e in events)!=14362: raise ValueError('Training supervised count mismatch')
    if any(abs(e['objective_loss']-(.2*e['decision_loss']+.8*e['rest_loss']))>1e-5 for e in events): raise ValueError('Recorded objective mismatch')
    return result


def evaluate(root,f,cases,step,resources,replay_sft=None):
    import mlx.core as mx
    from mlx_lm import load
    verify(root,f,full=False); check_training(root,f)
    name=f'W20-{step}'; split='dev' if replay_sft is None else 'seen_train_120'
    path=root/(f'dev-{name}.jsonl' if replay_sft is None else 'free-W20.jsonl')
    adapter=root/f'train-W20/step-{step:04d}'; meta=eval_meta(f,name,split,'v2',adapter,cases)
    if path.exists():
        rows=validate_evaluation(path,cases,meta,partial=True)
        if len(rows)==len(cases) and replay_sft is None: rescore(cases,rows,path,repair_missing_metrics=True); return rows
    update_state(root,stage='development' if replay_sft is None else 'train_replay',status='running',detail=name,step=0,total_steps=120,message='加载固定检查点，逐题自由生成。')
    resources.check(); model,tokenizer=load(f['snapshot'],adapter_path=str(adapter))
    try:
        resources.check(); evaluate_model(model,tokenizer,cases,path,meta,root,'generation',resources)
        rows=validate_evaluation(path,cases,meta); rescore(cases,rows,path,repair_missing_metrics=True)
        if replay_sft is not None:
            tfmeta=dict(meta,mode='teacher_forced',sft_sha256=digest(replay_sft),source_sha256=sha(SOURCE/'scripts/run_train_replay.py'))
            tfpath=root/'teacher-W20.jsonl'; metapath=root/'teacher-W20.meta.json'
            if metapath.exists() and read(metapath)!=tfmeta: raise ValueError('Teacher-forced resume identity mismatch')
            if tfpath.exists() and not metapath.exists(): raise ValueError('Teacher outputs lack identity')
            write_json(metapath,tfmeta)
            teacher_forced(model,tokenizer,replay_sft,cases,tfpath,root,name,resources)
        resources.check(); return rows
    finally:
        del model,tokenizer; mx.clear_cache()


def replay_summary(cases,free,tf):
    m=replay_metrics(cases,free,tf); decisions=[]; objectives=[]
    for row in tf:
        n=row['supervised_tokens']; details=row['token_details']
        if len(details)!=n or abs(sum(t['loss'] for t in details)-row['loss_sum'])>1e-6: raise ValueError('Teacher loss bookkeeping mismatch')
        if row['correct_tokens']!=sum(t['correct'] for t in details): raise ValueError('Teacher accuracy bookkeeping mismatch')
        d=details[row['action_positions'][0]]['loss']; decisions.append(d)
        objectives.append(.2*d+.8*(row['loss_sum']-d)/(n-1))
    m.update(first_decision_loss=sum(decisions)/len(decisions),weighted_objective=sum(objectives)/len(objectives),first_decision=report.decision_metrics(tf))
    return m


def build_report(root,f,complete=False):
    p=verify(root,f,full=False); dev=read_rows(Path(f['dataset'])/'dev.jsonl'); old,_=reused_rows(p,dev)
    grid={f'C0-{s}':metrics_for(dev,old['M0',s]) for s in [200,400,600]}; outputs={}
    for s in [200,400,600]:
        path=root/f'dev-W20-{s}.jsonl'
        if not path.exists(): continue
        adapter=root/f'train-W20/step-{s:04d}'
        verify_checkpoint(adapter,recipe(f['fingerprint'],sha(Path(f['dataset'])/'D0-sft.jsonl')),s)
        rows=validate_evaluation(path,dev,eval_meta(f,f'W20-{s}','dev','v2',adapter,dev),partial=True)
        if len(rows)==120:
            rescore(dev,rows,path,repair_missing_metrics=True); grid[f'W20-{s}']=metrics_for(dev,rows); outputs[s]=rows
    cases,sft,_=select_seen(p)
    replay={'C0':replay_summary(cases,read_rows(REPLAY/'free-M0.jsonl'),read_rows(REPLAY/'teacher-M0.jsonl'))}
    if (root/'teacher-W20.jsonl').exists() and len(read_rows(root/'teacher-W20.jsonl'))==120:
        adapter=root/'train-W20/step-0600'; meta=eval_meta(f,'W20-600','seen_train_120','v2',adapter,cases)
        if read(root/'teacher-W20.meta.json')!=dict(meta,mode='teacher_forced',sft_sha256=digest(sft),source_sha256=sha(SOURCE/'scripts/run_train_replay.py')): raise ValueError('Teacher-forced metadata mismatch')
        free=validate_evaluation(root/'free-W20.jsonl',cases,meta); rescore(cases,free,root/'free-W20.jsonl')
        replay['W20']=replay_summary(cases,free,read_rows(root/'teacher-W20.jsonl'))
    training=check_training(root,f) if (root/'train-W20/summary.json').exists() else None
    resources=[read(path) for path in sorted(root.glob('resource-summary-*.json'))]
    if complete: report.require_completion(training,resources)
    events={'C0':[e for e in read_rows(Path(p['old_run'])/'train-B/events.jsonl') if e['optimizer_updates']<=75],
        'W20':read_rows(root/'train-W20/events.jsonl') if (root/'train-W20/events.jsonl').exists() else []}
    curves=report.curve_series(events)
    v=report.verdict(grid,len(replay)==2) if complete else None
    if complete and v is None: raise ValueError('Incomplete result grid')
    examples=[]
    if 600 in outputs:
        for c,a,b in zip(dev,old['M0',600],outputs[600]):
            if a['score']['success']!=b['score']['success']:
                examples.append(dict(id=c['id'],request=c['request'],gold=c['gold'],C0=a['output'],W20=b['output'],C0_correct=a['score']['success'],W20_correct=b['score']['success']))
    data=dict(status='complete' if complete else 'partial',fingerprint=f['fingerprint'],development=grid,replay=replay,verdict=v,examples=examples,
        limitations=['One fixed weight, seed and exposure budget; no claim about all weighting settings.','Development set is repeatedly used; sealed final test untouched.','Train replay is a fixed paired/tool-balanced subset, not full training-population accuracy.'],
        training=training,resources=resources,training_events=events,curves=curves,
        artifacts={p.name:sha(p) for p in sorted(root.glob('*.jsonl'))})
    report.write(data,root,SOURCE)
    checks=[dict(name=k,complete=f'{m["C"]["count"]}/60',clarify=f'{m["Q"]["count"]}/48',unsupported=f'{m["U"]["count"]}/12') for k,m in grid.items()]
    update_state(root,results_ready=True,checkpoints=checks)
    return data


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['serve','freeze','run','report'],required=True); args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='serve':
        from toolcall_lab import live_progress
        live_progress.HTML=MONITOR_HTML.replace('训练样本回放','动作加权实验').replace('已见训练样本回放','动作位置加权 W20').replace('固定120题 · 原模型与M0/M1/M2最新75更新检查点 · 自由生成与teacher forcing · 只评测，不训练','固定20%动作权重 · 75更新 · 三检查点开发评测 + 训练回放').replace('已完成模型','已完成开发评测')
        live_progress.HTML=live_progress.HTML.replace('</html>', '''<section><h2>训练损失（每8题平均）</h2><p>绿：原始loss · 橙：20%加权目标；不同目标不能按大小互相排名。</p><svg id="losschart" viewBox="0 0 900 270" role="img" aria-label="实时训练损失"></svg></section><script>
async function lossplot(){try{let s=await(await fetch('/api/state',{cache:'no-store'})).json(),h=s.loss_history||[],svg=document.getElementById('losschart');svg.replaceChildren();if(!h.length)return;let max=Math.max(...h.flatMap(p=>[p.loss,p.raw_loss]))*1.05||1;for(let [key,color] of [['raw_loss','#7ee0b6'],['loss','#ffb75d']]){let line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',h.map(p=>(45+p.step/75*810)+','+(230-p[key]/max*210)).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',color);line.setAttribute('stroke-width','2');svg.append(line)}for(let [x,y,t] of [[5,20,max.toFixed(2)],[20,230,'0'],[45,260,'0'],[800,260,'75更新']]){let label=document.createElementNS('http://www.w3.org/2000/svg','text');label.setAttribute('x',x);label.setAttribute('y',y);label.setAttribute('fill','#a7bbc6');label.textContent=t;svg.append(label)}}catch(e){}}lossplot();setInterval(lossplot,5000);</script></html>''')
        sys.argv=[sys.argv[0],'--root',str(ROOT),'--port','49407']; live_progress.main(); return
    import fcntl
    lock=(ROOT/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    beat=Heartbeat(ROOT); beat.start(); resources=None
    try:
        if args.stage=='freeze': freeze(ROOT); return
        f=read(ROOT/'frozen.json'); p=verify(ROOT,f)
        if args.stage=='report':
            if (ROOT/'completed.json').exists(): report.verify_marker(read(ROOT/'completed.json'),f['fingerprint'],sha(ROOT/'report-data.json'))
            build_report(ROOT,f,(ROOT/'completed.json').exists()); return
        if (ROOT/'completed.json').exists():
            report.verify_marker(read(ROOT/'completed.json'),f['fingerprint'],sha(ROOT/'report-data.json'))
            print('Already complete',flush=True); return
        resources=ResourceMonitor(ROOT/f'resources-{int(time.time())}.jsonl'); resources.start()
        baseline=ROOT/'resource-baseline.json'
        if baseline.exists():
            initial=read(baseline); current=resources.samples[0]; resources.samples.insert(0,initial)
            problem=resource_violation(current,initial['swap_used_bytes'])
            if problem: resources.error=problem
        else: write_json(baseline,resources.samples[0])
        resources.check()
        import mlx.core as mx
        configure_mlx_memory(mx); build_report(ROOT,f)
        train(Path(f['snapshot']),Path(f['dataset'])/'D0-sft.jsonl',ROOT/'train-W20',fingerprint=f['fingerprint'],progress_root=ROOT,resources=resources)
        check_training(ROOT,f)
        dev=read_rows(Path(f['dataset'])/'dev.jsonl')
        for step in [200,400,600]: evaluate(ROOT,f,dev,step,resources); build_report(ROOT,f)
        cases,sft,_=select_seen(p); evaluate(ROOT,f,cases,600,resources,replay_sft=sft)
        resources.stop(); resources.check(); write_json(ROOT/f'resource-summary-{int(time.time())}.json',resources.summary()); resources=None
        verify(ROOT,f); data=build_report(ROOT,f,True)
        write_json(ROOT/'completed.json',dict(status='complete',fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json'),finished_at=time.time()))
        update_state(ROOT,stage='complete',status='complete',step=120,total_steps=120,finished_at=time.time(),message='动作加权实验与重评分已完成。',results_ready=True)
        print(json.dumps(data['verdict']),flush=True)
    except BaseException as error:
        update_state(ROOT,status='failed',message=repr(error)); write_json(ROOT/'last-error.json',dict(error=repr(error),time=time.time()))
        raise
    finally:
        if resources:
            resources.stop(); write_json(ROOT/f'resource-summary-{int(time.time())}.json',resources.summary())
        beat.stop(); fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__': main()
