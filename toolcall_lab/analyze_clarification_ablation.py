"""Predeclared equal-budget contrasts and separately gated operating-point selection."""
import random
from collections import defaultdict
from .evaluate_formal import summarize
from .report_formal import clarification_field_errors
from .train_ablation import CHECKPOINTS

ARMS=('R0','R1','R2','R3')
REFERENCES=('base_v2','base_no_examples','base_checklist','A75')


def aligned_cases(cases,rows):
    cm={c['id']:c for c in cases};rm={r['case_id']:r for r in rows}
    if len(cm)!=len(cases) or len(rm)!=len(rows) or cm.keys()!=rm.keys():raise ValueError('Missing/duplicate/mismatched case IDs')
    for k,c in cm.items():
        r=rm[k]
        if any(r.get(key)!=c.get(key) for key in ['condition','gold','template_group','target_tool','missing_fields']):
            raise ValueError('Output does not match frozen case: '+k)
    return cm,rm


def units(cases,rows,metric):
    cm,rm=aligned_cases(cases,rows);complete={c['intent_group']:c['id'] for c in cases if c['condition']=='complete'}
    result=[]
    for c in cases:
        condition=c['condition'];r=rm[c['id']];s=r['score'];action=s.get('action',{}).get('action')
        if metric in ('C','F') and condition=='complete':value=int(s['success']) if metric=='C' else int(action=='clarify')
        elif metric in ('Q','P') and condition=='natural':value=int(s['success']) if metric=='Q' else int(action=='call')
        elif metric=='J' and condition=='natural':
            if c['intent_group'] not in complete:raise ValueError('Unpaired natural case')
            pair=rm[complete[c['intent_group']]]
            value=int(s['success'] and pair['score']['success'] and not r.get('guard',{}).get('blocked') and not pair.get('guard',{}).get('blocked'))
        elif metric=='U' and condition=='unsupported':value=int(s['success'])
        else:continue
        result.append(dict(id=c['id'],family=c['template_group'],tool=c['target_tool'],value=value))
    return result


def metrics_for(cases,rows):
    cm,rm=aligned_cases(cases,rows)
    result={}
    for k in ('C','Q','F','P','U','J'):
        values=units(cases,rows,k);n=len(values);count=sum(x['value'] for x in values)
        result[k]=dict(count=count,n=n,rate=count/n if n else None)
    for diff in ('boundary','routine'):
        ids=[c['id'] for c in cases if c['condition']=='complete' and c.get('difficulty')==diff]
        count=sum(bool(rm[i]['score']['success']) for i in ids)
        result[diff]=dict(count=count,n=len(ids),rate=count/len(ids) if ids else None)
    result['by_condition']=summarize(rows)
    result['by_tool']={t:summarize([r for r in rows if r['target_tool']==t]) for t in sorted({r['target_tool'] for r in rows if r['target_tool']})}
    result['field_errors']=clarification_field_errors(rows)
    return result


def gate_candidate(m,references,r0):
    floors=dict(complete_floor_count=max(54,max(x['C']['count'] for x in references)-3,r0['C']['count']-3),
        boundary_floor_count=max(x['boundary']['count'] for x in references)-3,
        unsupported_floor_count=max(x['U']['count'] for x in references)-1,false_clarify_ceiling_count=3)
    reasons=[]
    if m['C']['count']<floors['complete_floor_count']:reasons.append('complete_accuracy')
    if m['F']['count']>3:reasons.append('overclarification')
    if m['boundary']['count']<floors['boundary_floor_count']:reasons.append('boundary_complete')
    if m['U']['count']<floors['unsupported_floor_count']:reasons.append('unsupported')
    return dict(eligible=not reasons,reasons=reasons,**floors)


def select_candidate(analysis):
    eligible=[c for c in analysis['candidates'] if c['gate']['eligible']]
    ranked=sorted(eligible,key=lambda c:(-c['metrics']['Q']['count'],-c['metrics']['J']['count'],c['metrics']['F']['count'],
        -c['metrics']['C']['count'],c['microstep'],ARMS.index(c['arm'])))
    return dict(selected=ranked[0] if ranked else None,eligible_count=len(eligible),
        rule='Hard capability gates; Q desc, J desc, F asc, C desc, earlier step, R0/R1/R2/R3.',
        regression_required=bool(ranked and ranked[0]['arm']!='R0'))


def effect_interval(unit_sets,coefficients,replicates=2000):
    """Paired family bootstrap stratified by tool and family size, fixing n."""
    first=unit_sets[0];maps=[{v['id']:v for v in seq} for seq in unit_sets]
    if any(m.keys()!=maps[0].keys() for m in maps):raise ValueError('Effect alignment mismatch')
    grouped=defaultdict(list)
    for u in first:
        if any(m[u['id']]['family']!=u['family'] for m in maps):raise ValueError('Effect family mismatch')
        grouped[u['family']].append(sum(c*m[u['id']]['value'] for c,m in zip(coefficients,maps)))
    strata=defaultdict(list)
    for family,vals in grouped.items():
        tool=next(u['tool'] for u in first if u['family']==family)
        strata[(tool,len(vals))].append(sum(vals))
    n=len(first);delta=sum(sum(v) for v in grouped.values());rng=random.Random(20261114);draws=[]
    for _ in range(replicates):
        draws.append(100*sum(sum(rng.choice(values) for _ in values) for values in strata.values())/n)
    draws.sort()
    return dict(delta_count=delta,n=n,delta_pp=100*delta/n,ci95_pp=[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]],
        families=len(grouped),replicates=replicates,bootstrap='paired family, stratified by tool and family size; fixed denominator')


def analyze_pilot(cases:list[dict],outputs:dict,references:dict,replicates=2000)->dict:
    if set(outputs)!={(a,s) for a in ARMS for s in CHECKPOINTS}:raise ValueError('Incomplete or unexpected four-arm grid')
    if set(references)!=set(REFERENCES):raise ValueError('Four fixed references required')
    refs={n:metrics_for(cases,r) for n,r in references.items()}
    metrics={key:metrics_for(cases,r) for key,r in outputs.items()}
    for m in [*refs.values(),*metrics.values()]:
        if [m[k]['n'] for k in ['C','Q','F','P','U','J','boundary','routine']]!=[60,48,60,48,12,48,30,30]:raise ValueError('Wrong pilot denominators')
    candidates=[]
    for a in ARMS:
        for s in CHECKPOINTS:
            m=metrics[a,s];candidates.append(dict(arm=a,microstep=s,updates=s//8,metrics=m,gate=gate_candidate(m,list(refs.values()),metrics['R0',s])))
    gates={(c['arm'],c['microstep']):c['gate'] for c in candidates}
    effects=effects_at_primary(cases,outputs,gates,replicates)
    result=dict(status='complete',kind='development pilot',primary_updates=75,candidates=candidates,references=refs,effects=effects,
        limitations=['One seed; development selection, not a new held-out final test.','Synthetic authored expressions and shared tool domain.',
                     'Intervals resample expression families, not training seeds.','Historical control wall time cannot identify a causal speed effect.'])
    result['selection']=select_candidate(result)
    return result


def effects_at_primary(cases,outputs,gates,replicates):
    effects={}
    for a,b in [('R0','R1'),('R0','R2'),('R2','R3'),('R1','R3')]:
        e={k:effect_interval([units(cases,outputs[a,600],k),units(cases,outputs[b,600],k)],[-1,1],replicates) for k in ['C','Q','F','J']}
        eligible=gates[b,600]['eligible'];q=e['Q']['delta_count'];f=e['F']['delta_count']
        e['practical_change']=dict(capability_gate_passed=eligible,higher_Q=eligible and q>=4,
            less_F_preserving_Q=eligible and f<=-3 and q>=-2,
            both_directionally_improved_preserving_C=eligible and q>0 and f<0)
        effects[b+'-'+a]=e
    effects['interaction']={k:effect_interval([units(cases,outputs[a,600],k) for a in ARMS],[1,-1,-1,1],replicates) for k in ['C','Q','F','J']}
    return effects


def analyze_available(cases,outputs,references,replicates=2000):
    """Incomplete runs remain descriptive and never select/promote a candidate."""
    expected={(a,s) for a in ARMS for s in CHECKPOINTS}
    if set(outputs)-expected or set(references)-set(REFERENCES):raise ValueError('Unexpected partial grid entries')
    refs={n:metrics_for(cases,r) for n,r in references.items()};metrics={k:metrics_for(cases,r) for k,r in outputs.items()}
    common=[s for s in CHECKPOINTS if all((a,s) in outputs for a in ARMS)];candidates=[]
    for (a,s),m in sorted(metrics.items()):
        gate=gate_candidate(m,list(refs.values()),metrics['R0',s]) if len(refs)==4 and ('R0',s) in metrics else dict(eligible=False,reasons=['missing_reference'],complete_floor_count=None)
        candidates.append(dict(arm=a,microstep=s,updates=s//8,metrics=m,gate=gate))
    gates={(c['arm'],c['microstep']):c['gate'] for c in candidates}
    effects=effects_at_primary(cases,outputs,gates,replicates) if 600 in common else {}
    return dict(status='incomplete',kind='development pilot',primary_updates=75,candidates=candidates,references=refs,effects=effects,
        common_updates=[s//8 for s in common],missing_cells=[f'{a}-{s}' for a,s in sorted(expected-set(outputs))],
        missing_references=sorted(set(REFERENCES)-set(refs)),selection=dict(selected=None,eligible_count=None,regression_required=False,reason='Incomplete experiment; selection withheld'),
        limitations=['Interrupted development pilot; only fully completed common points can be compared.','No selection or promotion from an incomplete grid.','One seed, synthetic tasks, no new final test.'])


def historical_complete_counts(summary):
    return {k:summary['complete_'+k]['success'] for k in ['original','semantic_a','semantic_b']}


def regression_decision(counts):
    floors={'original':38,'semantic_a':39,'semantic_b':39}
    if set(counts)!=set(floors):raise ValueError('Three historical complete subgroups required')
    failures=[k for k in floors if counts[k]<floors[k]]
    return dict(passed=not failures,counts=counts,floors=floors,failures=failures,
                next_action='eligible_for_later_confirmation' if not failures else 'stop_promotion')
