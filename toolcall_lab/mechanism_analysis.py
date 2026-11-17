"""Fixed-budget, single-factor mechanism contrasts; never promotes a model."""
from .analyze_clarification_ablation import metrics_for,gate_candidate,units,effect_interval,REFERENCES

ARMS=('M0','M1','M2')
STEPS=(200,400,600)


def analyze(cases,outputs,references,complete=True,replicates=2000):
    expected={(a,s) for a in ARMS for s in STEPS}
    if set(outputs)-expected or set(references)-set(REFERENCES):raise ValueError('Unexpected mechanism grid')
    if complete and (set(outputs)!=expected or set(references)!=set(REFERENCES)):raise ValueError('Incomplete mechanism grid')
    refs={n:metrics_for(cases,r) for n,r in references.items()};metrics={k:metrics_for(cases,r) for k,r in outputs.items()}
    for m in [*refs.values(),*metrics.values()]:
        if [m[k]['n'] for k in ['C','Q','F','P','U','J','boundary','routine']]!=[60,48,60,48,12,48,30,30]:
            raise ValueError('Wrong mechanism denominators')
    candidates=[]
    for (a,s),m in sorted(metrics.items()):
        gate=gate_candidate(m,list(refs.values()),metrics['M0',s]) if len(refs)==4 and ('M0',s) in metrics else dict(eligible=False,reasons=['missing_reference'],complete_floor_count=None)
        candidates.append(dict(arm=a,microstep=s,updates=s//8,metrics=m,gate=gate))
    effects={};verdicts={};gates={(c['arm'],c['microstep']):c['gate'] for c in candidates}
    for arm in ['M1','M2']:
        if not all((a,600) in outputs for a in ['M0',arm]):continue
        effect={k:effect_interval([units(cases,outputs['M0',600],k),units(cases,outputs[arm,600],k)],[-1,1],replicates) for k in ['C','Q','F','J']}
        effects[arm+'-M0']=effect
        if complete:
            gate=gates[arm,600]['eligible'];q=effect['Q']['delta_count'];f=effect['F']['delta_count']
            verdicts[arm]=dict(capability_gate_passed=gate,Q_gain_at_least_four=q>=4,F_reduced=f<0,
                joint_goal=gate and q>=4 and f<0,next_action='independent_confirmation' if gate and q>=4 and f<0 else 'diagnose_remaining_tradeoff')
    return dict(status='complete' if complete else 'incomplete',kind='development mechanism diagnostic',primary_updates=75,
        candidates=candidates,references=refs,effects=effects,verdicts=verdicts,
        common_updates=[s//8 for s in STEPS if all((a,s) in outputs for a in ARMS)],
        missing_cells=[f'{a}-{s}' for a,s in sorted(expected-set(outputs))],missing_references=sorted(set(REFERENCES)-set(refs)),
        limitations=['Single training seed; synthetic tasks on a previously consumed development set.',
                     'Family bootstrap intervals do not measure variation across training seeds.',
                     'M0 outputs are reused only after exact 25-update numerical certification.',
                     'No new final test, second seed, historical regression or combined mechanism arm.',
                     'An incomplete grid is descriptive only; no joint-goal verdict is issued.'])
