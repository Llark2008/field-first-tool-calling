"""Audit, freeze, run and verify the paired fields-first SFT pilot."""
import argparse
from collections import Counter
import math
from pathlib import Path
import time
from toolcall_lab.ablation_data import read_rows,audit_candidate
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import Heartbeat,update_state
from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
from toolcall_lab.run_clarification_ablation import read_json as read,write_json,verify_file_map
from experiments.fields_first_prompt.run import ROOT as BASE,verify as verify_base,checked_rows as old_checked
from experiments.action_weighting.run import verify as verify_weighting,ROOT as WEIGHTING
from experiments.action_weighting.report import verify_marker
from .data import digest,compact,audit_case,select_cases,messages,target,encoded_target
from .scoring import score
from .train import train,check_training
from .evaluate import evaluate
from .progress import ROOT
from . import report
from .resources import verify_sessions,save_summary

SOURCE=Path(__file__).resolve().parents[2]
DOCS=[SOURCE/'docs/superpowers'/p for p in ['specs/2026-11-15-fields-sft-design.md','plans/2026-11-15-fields-sft.md']]

def sources():return {str(p.relative_to(SOURCE)):sha(p) for p in sorted([*list((SOURCE/'experiments/fields_sft').rglob('*.py')),*DOCS])}

def write_rows(path,rows):
    content=''.join(compact(r)+'\n' for r in rows)
    if path.exists() and path.read_text()!=content:raise ValueError('Cannot overwrite changed dataset/output: '+str(path))
    path.write_text(content)

def audit():
    from mlx_lm.utils import load_tokenizer
    update_state(ROOT,stage='audit',status='running',step=0,total_steps=1320,message='核验D1数据来源、配对标签、两段答案监督和原模型基线。')
    base=read(BASE/'frozen.json');verify_base(base);verify_weighting(WEIGHTING,read(WEIGHTING/'frozen.json'))
    verify_marker(read(BASE/'completed.json'),base['fingerprint'],sha(BASE/'report-data.json'))
    dataset=Path(base['dataset']);candidate=audit_candidate(dataset)
    rows=read_rows(dataset/'D1.jsonl');dev=read_rows(dataset/'dev.jsonl')
    for c in rows+dev:audit_case(c)
    selected=select_cases(rows);seen=select_cases(selected,pairs_per_tool=8,extras_per_tool=2,unsupported=12)
    if len(selected)!=600 or Counter(c['gold']['action'] for c in selected)!=dict(call=300,clarify=240,unsupported=60):raise ValueError('Exposure quota mismatch')
    if len(seen)!=120 or len({r['id'] for r in seen}&{r['id'] for r in selected})!=120:raise ValueError('Replay must be seen')
    if {c['id'] for c in rows}&{c['id'] for c in dev}:raise ValueError('Train/dev leakage')
    write_rows(ROOT/'train-cases.jsonl',selected);write_rows(ROOT/'seen-cases.jsonl',seen);write_rows(ROOT/'dev-cases.jsonl',dev)
    tok=load_tokenizer(Path(base['snapshot']));tokens={};annotations={};baselines={};maxlen=0
    for arm,old in [('J','I'),('F','V')]:
        encoded=[encoded_target(c,arm,tok) for c in selected];maxlen=max(maxlen,max(len(t) for _,t,_,_ in encoded))
        tokens[arm]=dict(input_tokens=sum(o for _,t,o,g in encoded),answer_tokens=sum(len(t)-o for _,t,o,g in encoded),max_sequence=max(len(t) for _,t,o,g in encoded))
        annotations[arm]=[dict(id=c['id'],offset=o,length=len(t),groups=g,token_sha256=digest(t)) for c,(_,t,o,g) in zip(selected,encoded)]
        write_rows(ROOT/f'sft-{arm}.jsonl',[r for r,_,_,_ in encoded])
        if digest([messages(c,arm) for c in dev])!=base['prompt_sha256'][old]:raise ValueError('Base prompt mismatch')
        originals=old_checked(BASE/f'free-{old}.jsonl',dev,base,old);converted=[]
        for c,r in zip(dev,originals):
            new=score(c,r['output'],r['finish_reason'],arm)
            if any(new[k]!=r[k] for k in ['score','guard','failure_type']):raise ValueError('Base primary metric mismatch')
            converted.append(dict(r,**new))
        path=ROOT/f'dev-{arm}-0.jsonl';write_rows(path,converted)
        baselines[arm]=dict(parent_arm=old,parent_output_sha256=sha(BASE/f'free-{old}.jsonl'),sha256=sha(path))
    for c in selected:
        if target(c,'J')[0]!=target(c,'F')[0].split('\n')[1]:raise ValueError('Final labels differ')
        altered=dict(c,gold={'action':'unsupported'},supplied_arguments={'secret':'oracle'},missing_fields=['secret'])
        for arm in ['J','F']:
            if messages(c,arm)!=messages(altered,arm):raise ValueError('Private annotations leaked')
    cap=1+32*math.ceil(maxlen/32)
    if cap>2048:raise ValueError('Sequence budget exceeded')
    a=dict(status='passed',full_annotation_rows=len(rows)+len(dev),selected_rows=600,pairs=240,actions=dict(Counter(c['gold']['action'] for c in selected)),
        by_tool={str(k):dict(Counter(c['condition'] for c in selected if c['target_tool']==k)) for k in sorted({c['target_tool'] for c in selected},key=str)},
        null_fields=dict(Counter(k for c in selected for k in c['missing_fields'])),tokens=tokens,cap=cap,seen_replay=120,
        data_candidate=candidate,eos_after_entire_answer=True,prompt_labels_excluded=True,exposure_sha256=digest([c['id'] for c in selected]))
    write_json(ROOT/'input-audit.json',a);write_json(ROOT/'token-audit.json',annotations)
    update_state(ROOT,stage='audit',status='pending',step=1320,total_steps=1320,audit=a,message='数据与token审计通过，训练前核对实现审查记录。')
    return base,a,baselines

def freeze():
    review=read(ROOT/'implementation-review.json')
    if review.get('status')!='passed' or review.get('sources')!=sources():raise ValueError('Missing or stale whole-change review')
    base,a,baselines=audit();ds=Path(base['dataset'])
    localnames=['train-cases.jsonl','seen-cases.jsonl','dev-cases.jsonl','sft-J.jsonl','sft-F.jsonl','input-audit.json','token-audit.json','implementation-review.json']
    identity=dict(sources=sources(),dataset=str(ds),snapshot=base['snapshot'],parent_fingerprint=base['fingerprint'],
        parent_files={str(p):sha(p) for p in [BASE/'frozen.json',BASE/'completed.json',BASE/'report-data.json',BASE/'free-I.jsonl',BASE/'free-I.meta.json',BASE/'free-V.jsonl',BASE/'free-V.meta.json']},
        dataset_files={name:sha(ds/name) for name in ['D1.jsonl','manifest.json','dev.jsonl','rewrite-audit.json']},
        local_files={name:sha(ROOT/name) for name in localnames},baselines=baselines,audit=a,cap=a['cap'],generation=dict(temperature=0,max_tokens=384,thinking=False))
    identity['fingerprint']=digest(identity)
    if (ROOT/'frozen.json').exists() and read(ROOT/'frozen.json')!=identity:raise ValueError('Cannot overwrite frozen experiment')
    write_json(ROOT/'frozen.json',identity)
    for relative in identity['sources']:
        dest=ROOT/'source-snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SOURCE/relative).read_bytes())
    return identity

def verify(f,full=True):
    obj=dict(f);expected=obj.pop('fingerprint')
    if digest(obj)!=expected or sources()!=f['sources']:raise ValueError('Frozen source identity changed')
    verify_file_map(ROOT,f['local_files']);verify_file_map(Path(f['dataset']),f['dataset_files']);verify_file_map(Path('/'),f['parent_files'])
    base=read(BASE/'frozen.json')
    if base['fingerprint']!=f['parent_fingerprint']:raise ValueError('Parent changed')
    verify_base(base)
    if full:verify_weighting(WEIGHTING,read(WEIGHTING/'frozen.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','freeze','run','report'],required=True);args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat=Heartbeat(ROOT);beat.start();resources=None
    try:
        if args.stage=='audit':audit();return
        if args.stage=='freeze':f=freeze();report.build(ROOT,f,SOURCE);return
        f=read(ROOT/'frozen.json');verify(f)
        if (ROOT/'completed.json').exists():
            verify_marker(read(ROOT/'completed.json'),f['fingerprint'],sha(ROOT/'report-data.json'));verify_file_map(ROOT,read(ROOT/'report-data.json')['artifacts'])
            # Recompute all metrics/targets as well as checking persisted artifact digests.
            report.build(ROOT,f,SOURCE,True);print('Complete and verified',flush=True);return
        if args.stage=='report':report.build(ROOT,f,SOURCE);return
        verify_sessions(ROOT)
        resources=ResourceMonitor(ROOT/f'resources-{int(time.time())}.jsonl');resources.start();bp=ROOT/'resource-baseline.json'
        if bp.exists():
            initial=read(bp);current=resources.samples[0];resources.samples.insert(0,initial)
            problem=resource_violation(current,initial['swap_used_bytes'])
            if problem:resources.error=problem
        else:write_json(bp,resources.samples[0])
        resources.check();report.build(ROOT,f,SOURCE)
        dev=read_rows(ROOT/'dev-cases.jsonl');seen=read_rows(ROOT/'seen-cases.jsonl')
        for arm in ['J','F']:
            verify(f,False);train(ROOT,f,arm,resources);check_training(ROOT,f,arm)
            for step in [200,400,600]:
                evaluate(ROOT,f,dev,arm,step,'dev',resources);report.build(ROOT,f,SOURCE)
            evaluate(ROOT,f,seen,arm,600,'seen',resources);report.build(ROOT,f,SOURCE)
        resources.stop();resources.check();save_summary(ROOT,resources);resources=None
        verify(f);data=report.build(ROOT,f,SOURCE,True)
        write_json(ROOT/'completed.json',dict(status='complete',fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json'),finished_at=time.time()))
        update_state(ROOT,stage='complete',status='complete',finished_at=time.time(),step=120,total_steps=120,message='两组训练、960条新自由生成与240条teacher forcing已完成并重评分。',results_ready=True)
        print(compact(data['verdict']),flush=True)
    except BaseException as exc:
        update_state(ROOT,status='failed',message=repr(exc));raise
    finally:
        if resources:resources.stop();save_summary(ROOT,resources)
        beat.stop();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
