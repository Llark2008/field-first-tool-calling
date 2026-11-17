"""Audit/freeze/run two learning rates over exactly three original-data epochs."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state,Heartbeat
from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
from experiments.fields_sft.resources import verify_sessions,save_summary
from experiments.fields_sft.data import audit_case,encoded_target,messages,digest,compact
from experiments.fields_sft.evaluate import validate_rows,validate_teacher,metadata
from experiments.fields_sft.train import check_training
from experiments.field_action_diagnostics.run import read,write,immutable
from .core import schedule,RATES
from .progress import ROOT
from .train import train_epoch
from .evaluate import evaluate_epoch,meta,paths
from . import report

SOURCE=Path(__file__).resolve().parents[2]
PARENT=ROOT.parent/'fields-sft-20261115'
DIAG=ROOT.parent/'field-action-diagnostics-20261116'
DOCS=[SOURCE/'docs/superpowers'/p for p in ['specs/2026-11-16-fields-epochs-design.md','plans/2026-11-16-fields-epochs.md']]

def sources():
    packages=['toolcall_lab','experiments/fields_epochs','experiments/fields_sft','experiments/field_action_diagnostics','experiments/fields_first_prompt','experiments/action_weighting']
    files=[p for d in packages for p in (SOURCE/d).rglob('*.py')]+DOCS+[SOURCE/'scripts/run_train_replay.py',SOURCE/'requirements-macos-arm64.txt']
    return {str(p.relative_to(SOURCE)):sha(p) for p in sorted(set(files))}

def versions():return {n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','tokenizers','numpy','jsonschema']}

def write_rows(p,rows):
    content=''.join(compact(r)+'\n' for r in rows)
    if p.exists() and p.read_text()!=content:raise ValueError('Existing raw artifact changed')
    if not p.exists():p.write_text(content)

def verify_signed_parent(root):
    frozen=read(root/'frozen.json');marker=read(root/'completed.json')
    if marker['fingerprint']!=frozen['fingerprint'] or marker['report_sha256']!=sha(root/'report-data.json'):
        raise ValueError('Parent signed report changed')
    data=read(root/'report-data.json');artifacts=data.get('artifacts',data.get('raw_sha256'))
    if not artifacts:raise ValueError('Parent artifact signatures missing')
    for name,value in artifacts.items():
        if sha(root/name)!=value:raise ValueError('Parent signed artifact changed: '+name)

def audit():
    from mlx_lm.utils import load_tokenizer
    from experiments.fields_sft.run import verify as verify_parent
    from experiments.field_action_diagnostics.run import verify as verify_diag
    update_state(ROOT,stage='audit',status='running',detail='原数据和续训状态审计',step=0,total_steps=840,message='验证600条目标完全不变、每轮数据曝光、固定评测和继承的优化器状态。')
    parent=read(PARENT/'frozen.json');verify_parent(parent);check_training(PARENT,parent,'F');verify_diag(read(DIAG/'frozen.json'))
    for p in [PARENT,DIAG]:verify_signed_parent(p)
    cases=read_rows(PARENT/'train-cases.jsonl');sft=read_rows(PARENT/'sft-F.jsonl');tok=load_tokenizer(Path(parent['snapshot']))
    if len(cases)!=600 or [c['id'] for c in cases]!=[r['id'] for r in sft]:raise ValueError('Training IDs changed')
    exposure=schedule([c['id'] for c in cases]);immutable(ROOT/'exposure.json',exposure);answer=input_n=0
    for c,r in zip(cases,sft):
        audit_case(c);row,ids,offset,groups=encoded_target(c,'F',tok,parent['cap'])
        if row!=r:raise ValueError('Original field-first training target changed')
        answer+=len(ids)-offset;input_n+=offset
    evalcases={};targets={};prompts={};maxlen=0;hashes={}
    for split in ['dev','seen']:
        cs=read_rows(PARENT/f'{split}-cases.jsonl');evalcases[split]=cs;targets[split]={};prompts[split]={};hashes[split]=digest(cs)
        baseline=read_rows(PARENT/f'{split}-F-600.jsonl');validate_rows(cs,baseline,'F')
        if read(PARENT/f'{split}-F-600.meta.json')!=metadata(PARENT,parent,cs,'F',600,split):raise ValueError('Reused free output metadata changed')
        for c in cs:
            audit_case(c);row,ids,offset,groups=encoded_target(c,'F',tok,2048);maxlen=max(maxlen,len(ids))
            targets[split][c['id']]=dict(tokens=ids,offset=offset,groups=groups);prompts[split][c['id']]=digest(messages(c,'F'))
        if split=='seen':validate_teacher(cs,read_rows(PARENT/'teacher-F.jsonl'),'F',tok,2048)
    trainids=set(exposure)
    if not {c['id'] for c in evalcases['seen']}<=trainids or {c['id'] for c in evalcases['dev']}&trainids:raise ValueError('Train/eval membership mismatch')
    immutable(ROOT/'teacher-targets.json',targets)
    a=dict(status='passed',unique_training=600,epochs=3,exposures=1800,actions=dict(Counter(c['gold']['action'] for c in cases)),
        answer_tokens_per_epoch=answer,input_tokens_per_epoch=input_n,train_cap=parent['cap'],teacher_max_sequence=maxlen,
        seen=120,dev=120,new_training_epochs=5,reused_training_epochs=1,new_free_generations=1200,new_teacher_rows=1320,
        preserved_optimizer=True,bias_correction=False,order='original600 repeated3times',sealed_test_used=False)
    if answer!=parent['audit']['tokens']['F']['answer_tokens'] or input_n!=parent['audit']['tokens']['F']['input_tokens']:raise ValueError('Training token counts changed')
    immutable(ROOT/'audit.json',a)
    initial=[dict(arm='A',epoch=1,split=s,**report.cell_scores(cs,read_rows(PARENT/f'{s}-F-600.jsonl'))) for s,cs in evalcases.items()]
    update_state(ROOT,step=840,status='pending',audit=a,cells=initial,loss_A=read_rows(PARENT/'train-F/events.jsonl'),loss_B=[],message='审计完成；保留A第1轮已验证结果，等待代码审查和冻结。')
    print(compact(a),flush=True)
    return parent,a,prompts,hashes

def freeze():
    parent,a,prompts,hashes=audit()
    names=['frozen.json','completed.json','report-data.json','sft-F.jsonl','train-cases.jsonl','dev-cases.jsonl','seen-cases.jsonl','teacher-F.jsonl','teacher-F.meta.json',
           'dev-F-600.jsonl','seen-F-600.jsonl','dev-F-600.meta.json','seen-F-600.meta.json','train-F/config.json','train-F/events.jsonl','train-F/summary.json']
    inputs=[PARENT/n for n in names]+list((PARENT/'train-F/step-0600').iterdir())+[DIAG/'frozen.json',DIAG/'completed.json',DIAG/'report-data.json']
    inputs += [p for p in Path(parent['snapshot']).iterdir() if p.is_file()]
    f=dict(parent=str(PARENT),parent_fingerprint=parent['fingerprint'],snapshot=parent['snapshot'],cap=parent['cap'],teacher_cap=2048,
        sources=sources(),inputs={str(p):sha(p) for p in inputs},local_files={n:sha(ROOT/n) for n in ['audit.json','teacher-targets.json','exposure.json']},
        versions=versions(),audit=a,prompt_hashes=prompts,eval_cases_sha256=hashes,generation=parent['generation'])
    f['fingerprint']=digest(f);immutable(ROOT/'frozen.json',f)
    for relative in f['sources']:
        p=ROOT/'source-snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((SOURCE/relative).read_bytes())
    seed_reuse(f);report.build(ROOT,f,SOURCE)
    print('Frozen '+f['fingerprint'],flush=True)

def seed_reuse(f):
    for split in ['dev','seen']:
        old=read_rows(PARENT/f'{split}-F-600.jsonl');p=paths(ROOT,'A',1,split,'free')
        immutable(p.with_suffix('.meta.json'),meta(ROOT,f,'A',1,split,'free'))
        write_rows(p,[dict(case_id=r['case_id'],prompt_sha256=f['prompt_hashes'][split][r['case_id']],**{k:r[k] for k in ['output','finish_reason','seconds','prompt_tokens','generation_tokens']}) for r in old])
    p=paths(ROOT,'A',1,'seen','teacher');immutable(p.with_suffix('.meta.json'),meta(ROOT,f,'A',1,'seen','teacher'));write_rows(p,read_rows(PARENT/'teacher-F.jsonl'))

def verify(f):
    obj=dict(f);fp=obj.pop('fingerprint')
    if digest(obj)!=fp or sources()!=f['sources'] or versions()!=f['versions']:raise ValueError('Frozen source/config/environment changed')
    for p,h in f['inputs'].items():
        if sha(Path(p))!=h:raise ValueError('Historical input/model changed: '+p)
    for name,h in f['local_files'].items():
        if sha(ROOT/name)!=h:raise ValueError('Local frozen input changed: '+name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['audit','freeze','run','verify']);args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat=Heartbeat(ROOT);beat.start();resources=None
    try:
        if args.stage=='audit':audit();return
        if args.stage=='freeze':freeze();return
        f=read(ROOT/'frozen.json');verify(f)
        if (ROOT/'completed.json').exists():
            marker=read(ROOT/'completed.json')
            if marker['fingerprint']!=f['fingerprint'] or marker['report_sha256']!=sha(ROOT/'report-data.json'):raise ValueError('Completed report signature changed')
            report.build(ROOT,f,SOURCE,True,read(ROOT/'report-data.json'));print('Complete model/input/raw/teacher/resource/metric verification passed',flush=True);return
        if args.stage=='verify':raise ValueError('Run incomplete; no completion claim')
        verify_sessions(ROOT);resources=ResourceMonitor(ROOT/f'resources-{time.time_ns()}.jsonl');resources.start();bp=ROOT/'resource-baseline.json'
        if bp.exists():
            initial=read(bp);problem=resource_violation(resources.samples[0],initial['swap_used_bytes']);resources.samples.insert(0,initial)
            if problem:resources.error=problem
        else:write(bp,resources.samples[0])
        resources.check();seed_reuse(f)
        callback=lambda:report.build(ROOT,f,SOURCE)
        for epoch in [1,2,3]:
            for arm in RATES:
                verify(f);train_epoch(ROOT,f,arm,epoch,resources);callback()
                evaluate_epoch(ROOT,f,arm,epoch,resources,callback);callback()
        resources.stop();resources.check();save_summary(ROOT,resources);resources=None
        verify(f);result=report.build(ROOT,f,SOURCE,True)
        write(ROOT/'completed.json',dict(fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json'),finished_at=time.time()))
        update_state(ROOT,stage='complete',status='complete',detail='三轮对照完成',step=12,total_steps=12,finished_at=time.time(),results_ready=True,message='两组各225次更新，固定训练/开发每轮评测完成；模型与转换器分开评分。')
        print(compact({k:{m:v[m] for m in ['model','D']} for k,v in result['cells'].items()}),flush=True)
    except BaseException as exc:update_state(ROOT,status='failed',message=repr(exc));raise
    finally:
        if resources:resources.stop();save_summary(ROOT,resources)
        beat.stop();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
