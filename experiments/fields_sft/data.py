"""Paired selection and supervised fields, isolated from historical experiments."""
from collections import defaultdict
import hashlib
import json
import random

from experiments.fields_first_prompt.core import messages as old_messages, field_target

SEED = 20261113
SALT = 'fields-sft-20261115'

def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def messages(case, arm):
    if arm not in ['J', 'F']: raise ValueError('Unknown arm')
    return old_messages(case, {'J':'I', 'F':'V'}[arm])

def target(case, arm):
    if arm not in ['J', 'F']: raise ValueError('Unknown arm')
    spans=[]; content=''
    if arm=='F':
        obj=field_target(case)
        schema=next((t for t in case['tools'] if t['name']==obj['tool']),None)
        required=schema['parameters']['required'] if schema else []
        content='{"tool":'+compact(obj['tool'])+',"fields":{'
        for i,(k,v) in enumerate(obj['fields'].items()):
            content+=(',' if i else '')+compact(k)+':'
            start=len(content);content+=compact(v)
            spans.append(dict(name=k,kind='null' if v is None else 'known',required=k in required,start=start,end=len(content)))
        content+='}}\n'
    start=len(content);content+=compact(case['gold'])
    spans.append(dict(kind='action',name='final_action',start=start,end=len(content)))
    return content,spans

def audit_case(case):
    from jsonschema import Draft202012Validator
    from .scoring import equal
    obj=field_target(case); supplied=case['supplied_arguments'];gold=case['gold']
    if obj['tool'] is None:
        if gold!={'action':'unsupported'} or supplied or case['missing_fields']: raise ValueError('Unsupported labels disagree')
        return obj
    schema=next(t['parameters'] for t in case['tools'] if t['name']==obj['tool'])
    props=schema['properties']
    if not set(supplied)<=set(props): raise ValueError('Unknown supplied field')
    for k,v in supplied.items():
        if v is None or not Draft202012Validator(props[k]).is_valid(v): raise ValueError('Invalid supplied value: '+k)
    for k,v in obj['fields'].items():
        if v is not None and not Draft202012Validator(props[k]).is_valid(v): raise ValueError('Invalid field target: '+k)
    missing=sorted(k for k in schema['required'] if obj['fields'][k] is None)
    if missing!=sorted(case['missing_fields']): raise ValueError('Missing labels disagree')
    if missing:
        if gold!={'action':'clarify','missing_fields':case['missing_fields']}: raise ValueError('Clarify labels disagree')
    else:
        expected=dict(supplied)
        for k in schema['required']:
            if k not in expected and 'default' in props[k]:expected[k]=props[k]['default']
        if gold.get('action')!='call' or gold.get('name')!=obj['tool'] or not equal(gold.get('arguments'),expected) or not Draft202012Validator(schema).is_valid(gold.get('arguments')):
            raise ValueError('Call labels disagree or violate schema')
    return obj

def audit_pair(complete, natural):
    from .scoring import equal
    audit_case(complete);audit_case(natural)
    for key in ['group_id','target_tool','tools','context','template_group']:
        if complete[key]!=natural[key]: raise ValueError('Pair context changed: '+key)
    if complete['condition']!='complete' or natural['condition']!='natural': raise ValueError('Wrong pair conditions')
    a,b=complete['supplied_arguments'],natural['supplied_arguments']
    if not set(b)<=set(a) or any(not equal(a[k],v,k) for k,v in b.items()): raise ValueError('Pair known values changed')
    if set(a)-set(b)!=set(natural['missing_fields']): raise ValueError('Pair removes wrong fields')

def _hash(row):
    return digest([SALT,row['id']])

def select_cases(rows, pairs_per_tool=40, extras_per_tool=10, unsupported=60):
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate IDs')
    grouped=defaultdict(list)
    for r in rows:grouped[r['group_id']].append(r)
    pairs=defaultdict(list);extras=defaultdict(list);other=[]
    for group in grouped.values():
        if len(group)==2:
            ordered=sorted(group,key=lambda r:r['condition'])
            audit_pair(*ordered);pairs[ordered[0]['target_tool']].append(ordered)
        elif len(group)==1:
            r=group[0]
            if r['condition']=='unsupported':other.append(r)
            elif r['condition']=='complete':extras[r['target_tool']].append(r)
            else:raise ValueError('Unpaired missing example')
        else:raise ValueError('Wrong group size')
    selected=[]
    for tool in sorted(pairs):
        strata=defaultdict(list)
        for p in pairs[tool]:strata[tuple(sorted(p[1]['missing_fields']))].append(p)
        total=sum(map(len,strata.values()))
        if total<pairs_per_tool:raise ValueError('Insufficient pair quota')
        allocation={k:len(v)*pairs_per_tool//total for k,v in strata.items()}
        remainder=sorted(strata,key=lambda k:(-(len(strata[k])*pairs_per_tool%total),k))
        for k in remainder[:pairs_per_tool-sum(allocation.values())]:allocation[k]+=1
        for k in sorted(strata):
            for p in sorted(strata[k],key=lambda p:_hash(p[0]))[:allocation[k]]:selected.extend(p)
        chosen={r['id'] for r in selected}
        pool=sorted(extras[tool],key=_hash)+sorted([p[0] for p in pairs[tool] if p[0]['id'] not in chosen],key=_hash)
        if len(pool)<extras_per_tool:raise ValueError('Insufficient extra complete quota')
        selected.extend(pool[:extras_per_tool])
    if len(other)<unsupported:raise ValueError('Insufficient unsupported quota')
    selected.extend(sorted(other,key=_hash)[:unsupported])
    random.Random(SEED).shuffle(selected)
    return selected

def encoded_target(case,arm,tokenizer,max_length=2048):
    from toolcall_lab.training_spike import encode_row
    content,spans=target(case,arm)
    row=dict(id=case['id'],messages=messages(case,arm)+[dict(role='assistant',content=content)])
    tokens,offset=encode_row(row,tokenizer,max_length)
    encoding=tokenizer.backend_tokenizer.encode(content,add_special_tokens=False)
    if tokens[offset:offset+len(encoding.ids)]!=encoding.ids:raise ValueError('Answer token boundary mismatch')
    eos=set(tokenizer.eos_token_ids)
    if any(t in eos for t in encoding.ids) or not any(t in eos for t in tokens[offset+len(encoding.ids):]):raise ValueError('EOS must follow entire target')
    groups={k:[] for k in ['known','null','action']}
    for s in spans:
        indices=[offset+i for i,(a,b) in enumerate(encoding.offsets) if a<s['end'] and b>s['start']]
        if not indices:raise ValueError('Empty annotated token span')
        groups[s['kind']].extend(indices)
    groups={k:sorted(set(v)) for k,v in groups.items()}
    return row,tokens,offset,groups
