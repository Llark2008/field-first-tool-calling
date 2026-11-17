"""Matched wording-only intervention and frozen independent development pilot."""
import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .ablation_phrases import render_train, render_dev, DEV_UNSUPPORTED
from .formal_data import sha, make_fixture, schema_for, training_row, PROJECTS, ENVS, ALGOS
from .fixtures import reference_result
from .tasks import BASE_TOOLS

QUOTAS=dict(rank_runs=60,summarize_metric=50,compare_algorithms=30,list_runs=20,get_run_config=10,get_metric_series=10)
MASKS={
 'rank_runs':[['order'],['order'],['limit'],['metric'],['order','project'],['limit','environment'],['metric','order'],['metric','limit']],
 'summarize_metric':[['stat'],['stat'],['metric'],['metric'],['stat','environment'],['metric','project'],['stat','project'],['metric','stat']],
 'compare_algorithms':[['project'],['project'],['environment'],['environment'],['metric','project'],['metric','project'],['metric','environment'],['metric','environment']],
 'list_runs':[['project']]*8,'get_run_config':[['run_id']]*8,'get_metric_series':[['run_id']]*8,
}


def read_rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]


def _require(ok,message):
    if not ok: raise ValueError(message)


def rewrite_rows(rows):
    by_group=defaultdict(list)
    for r in rows: by_group[r['intent_group']].append(r)
    selected={}
    for tool,quota in QUOTAS.items():
        strata=defaultdict(list)
        for group,pair in by_group.items():
            if len(pair)!=2 or pair[0]['target_tool']!=tool: continue
            missing=next(r for r in pair if r['condition']=='natural')
            strata[tuple(missing['missing_fields'])].append(group)
        # Largest remainder allocation preserves source missing-pattern proportions.
        total=sum(map(len,strata.values())); allocation={k:quota*len(v)//total for k,v in strata.items()}
        remain=quota-sum(allocation.values())
        for k in sorted(strata,key=lambda k:(-(quota*len(strata[k])%total),k))[:remain]: allocation[k]+=1
        chosen=[]
        for k in sorted(strata):
            chosen.extend(sorted(strata[k],key=lambda g:hashlib.sha256(('ablation-20261114/'+g).encode()).hexdigest())[:allocation[k]])
        chosen.sort(key=lambda g:hashlib.sha256(('family/'+g).encode()).hexdigest())
        for i,g in enumerate(chosen): selected[g]=i%6
    rewritten=copy.deepcopy(rows); audit=[]
    for row in rewritten:
        if row['intent_group'] not in selected: continue
        family=selected[row['intent_group']]
        old=row['request']; row['request']=render_train(row['target_tool'],row['supplied_arguments'],family)
        audit.append(dict(id=row['id'],intent_group=row['intent_group'],target_tool=row['target_tool'],
            expression_family=f"rewrite:{row['target_tool']}:f{family}",old_request=old,new_request=row['request'],
            supplied_arguments=row['supplied_arguments'],missing_fields=row['missing_fields'],gold=row['gold']))
    return rewritten,audit


def build_dev():
    rng=random.Random(20261114); result=[]
    for ti,name in enumerate(BASE_TOOLS):
        rankdeck=[(m,o) for m in ['final_return','peak_return'] for o in ['asc','desc']]*2+ [('final_return','asc'),('peak_return','desc')]
        rng.shuffle(rankdeck)
        for i in range(10):
            family=i//2 if i<8 else i-8; boundary=(i%2==1) if i<8 else i==8
            projects=rng.sample(PROJECTS,3); fixture=make_fixture(rng,projects,8000+ti*10+i)
            catalog=rng.sample(list(fixture['configs']),6); run=rng.choice(catalog)
            p=projects[i%3]; e=ENVS[i%2]; algo=ALGOS[(i//2)%2]; metric=['final_return','peak_return'][(i//2)%2]
            args={
                'list_runs':dict(project=p,**({} if boundary else dict(environment=e,algorithm=algo))),
                'summarize_metric':dict(project=p,environment=e,algorithm=algo,metric=metric,stat=['mean','min','max'][i%3]),
                'rank_runs':dict(project=p,environment=e,metric=rankdeck[i][0],order=rankdeck[i][1],limit=1+i%3),
                'compare_algorithms':dict(project=p,environment=e,metric=metric,algorithms=rng.sample(ALGOS,2)),
                'get_run_config':dict(run_id=run),
                'get_metric_series':dict(run_id=run,metric='episode_return',**({} if boundary else dict(max_points=1+i%3))),
            }[name]
            names=[name]+rng.sample([n for n in BASE_TOOLS if n!=name],2);rng.shuffle(names)
            group=f'ablation-dev-{name}-{i:02d}';gold=dict(action='call',name=name,arguments=args)
            row=dict(id=group+'-complete',group_id=group,intent_group=group,template_group=f'ablation-dev:{name}:f{family}',
                split='dev',target_tool=name,condition='complete',difficulty='boundary' if boundary else 'routine',
                request=render_dev(name,args,family,boundary),
                context='Known project IDs: '+', '.join(projects)+'. Known run IDs: '+', '.join(catalog)+
                    '. Known environments: Hopper-v4, Walker2d-v4. Algorithms: SAC, PPO. Tables: experiments, evaluations. These inventories are not defaults; no user selection has been established.',
                tools=[schema_for(n,projects) for n in names],fixture=fixture,gold=gold,
                expected_result=reference_result(fixture,gold),missing_fields=[],supplied_arguments=args)
            result.append(row)
            if i<8:
                absent=sorted(MASKS[name][i]);remaining={k:v for k,v in args.items() if k not in absent}
                missing=copy.deepcopy(row);missing.update(id=group+'-natural',condition='natural',difficulty='missing',
                    request=render_dev(name,remaining,family,boundary),supplied_arguments=remaining,missing_fields=absent,
                    gold=dict(action='clarify',missing_fields=absent),expected_result=None)
                result.append(missing)
    for i in range(12):
        row=copy.deepcopy(result[i*7]);group=f'ablation-dev-unsupported-{i:02d}'
        row.update(id=group,group_id=group,intent_group=group,template_group=f'ablation-dev:unsupported:f{i%4}',
            condition='unsupported',difficulty='unsupported',target_tool=None,gold=dict(action='unsupported'),
            request=DEV_UNSUPPORTED[i%4],missing_fields=[],supplied_arguments={},expected_result=None)
        result.append(row)
    rng.shuffle(result)
    return result


def audit_rows(d0,d1,dev,rewrites):
    from .evaluation import score_response
    _require(len(d0)==len(d1)==1200,'Wrong training count')
    changed=[]
    for a,b in zip(d0,d1):
        _require({k:v for k,v in a.items() if k!='request'}=={k:v for k,v in b.items() if k!='request'},'Non-request content changed')
        if a['request']!=b['request']:changed.append(b)
    _require(len(changed)==360 and len({r['intent_group'] for r in changed})==180,'Wrong changed-pair count')
    _require(Counter(r['target_tool'] for r in changed if r['condition']=='complete')==QUOTAS,'Rewrite quota mismatch')
    _require(Counter(r['gold']['action'] for r in d1)==dict(call=600,clarify=480,unsupported=120),'Class mixture changed')
    _require({r['id'] for r in changed}=={r['id'] for r in rewrites},'Rewrite sidecar mismatch')
    _require(Counter(r['condition'] for r in dev)==dict(complete=60,natural=48,unsupported=12),'Dev mixture mismatch')
    _require(Counter(r['difficulty'] for r in dev if r['condition']=='complete')==dict(routine=30,boundary=30),'Difficulty mismatch')
    missing=[r for r in dev if r['condition']=='natural']
    _require(Counter(len(r['missing_fields']) for r in missing)=={1:36,2:12},'Omission sizes mismatch')
    _require(sum(bool(set(r['missing_fields'])&{'metric','stat','order','limit'}) for r in missing)==20,'Semantic omissions mismatch')
    families=defaultdict(set)
    for r in rewrites:families[r['expression_family']].add(r['gold']['action'])
    _require(len(families)==36 and all(v=={'call','clarify'} for v in families.values()),'Family encodes action')
    _require(not {r['template_group'] for r in d1}&{r['template_group'] for r in dev},'Cross-split template reuse')
    for rows in [d1,dev]:
        _require(len({r['id'] for r in rows})==len(rows),'Duplicate case IDs')
        for r in rows:
            _require(score_response(r,json.dumps(r['gold']))['success'],'Gold execution failed: '+r['id'])
            if r['gold']['action']=='unsupported':continue
            schema=next(t for t in r['tools'] if t['name']==r['target_tool'])
            required={k for k in schema['parameters']['required'] if 'default' not in schema['parameters']['properties'][k]}
            _require(set(r['missing_fields'])==required-set(r['supplied_arguments']),'Missing slots contradict schema: '+r['id'])
            if r['condition']=='natural':_require(r['gold']['missing_fields']==r['missing_fields'],'Missing gold mismatch')
    return dict(status='passed',train_count=1200,changed_requests=360,changed_pairs=180,unchanged_requests=840,
        pair_quotas=QUOTAS,dev_counts=dict(Counter(r['condition'] for r in dev)),dev_omission_sizes={'single':36,'double':12},
        semantic_dev_omissions=20,train_expression_families=36,dev_expression_families=28,
        review_attribution='Programmatic invariance/schema/executor checks; semantic review is a separate required record.')


def build_candidate(source_dataset:Path,output:Path)->dict:
    from .run import runtime_root
    import os
    source_dataset=Path(source_dataset);output=Path(output).resolve()
    runtime=runtime_root(Path(__file__).resolve().parents[1],os.environ.get('TOOLCALL_RUNTIME_DIR'))
    _require(output.is_relative_to(runtime/'datasets'),'Use external dataset directory')
    old=json.loads((source_dataset/'manifest.json').read_text())
    for name in ['B.jsonl','B-sft.jsonl']:_require(sha(source_dataset/name)==old['files'][name]['sha256'],'Source dataset changed')
    d0=read_rows(source_dataset/'B.jsonl');d1,rewrites=rewrite_rows(d0);dev=build_dev();audit=audit_rows(d0,d1,dev,rewrites)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source_dataset/'B.jsonl',output/'D0.jsonl');shutil.copyfile(source_dataset/'B-sft.jsonl',output/'D0-sft.jsonl')
    for name,rows in [('D1.jsonl',d1),('D1-sft.jsonl',[training_row(r) for r in d1]),('dev.jsonl',dev)]:
        (output/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    (output/'rewrite-audit.json').write_text(json.dumps(rewrites,ensure_ascii=False,indent=2))
    manifest=dict(status='candidate_requires_semantic_and_token_audit',source_dataset=str(source_dataset),
        source_manifest_sha256=sha(source_dataset/'manifest.json'),audit=audit,files={p.name:dict(sha256=sha(p)) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def audit_candidate(root:Path)->dict:
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,entry in manifest['files'].items():_require(sha(root/name)==entry['sha256'],'Candidate file changed: '+name)
    return audit_rows(read_rows(root/'D0.jsonl'),read_rows(root/'D1.jsonl'),read_rows(root/'dev.jsonl'),json.loads((root/'rewrite-audit.json').read_text()))
