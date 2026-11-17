"""Parameterized pilot loop preserving the audited historical accumulation math."""
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path
from .ablation_data import read_rows
from .formal_data import sha
from .live_progress import update_state
from .train_formal import (SEED,LORA,RECIPE,configure_mlx_memory,exposure_schedule,
                           save_training_state,restore_training_state)
from .training_spike import encode_row

CHECKPOINTS=(200,400,600,800,1200)


def checkpoint_steps(limit):
    if limit not in (200,1200):raise ValueError('Only certification or one-epoch pilot is supported')
    return tuple(s for s in CHECKPOINTS if s<=limit)


def recipe(learning_rate,max_microsteps,checkpoints,fingerprint,sft_sha256):
    if learning_rate not in (1e-4,5e-5):raise ValueError('Unplanned learning rate')
    if tuple(checkpoints)!=checkpoint_steps(max_microsteps):raise ValueError('Checkpoint schedule mismatch')
    return dict(RECIPE,learning_rate=learning_rate,max_microsteps=max_microsteps,checkpoints=list(checkpoints),
                max_seq_length=1217,fingerprint=fingerprint,sft_sha256=sft_sha256)


def validate_config(old,new):
    if old!=new:raise ValueError('Training configuration changed; cannot resume')


def make_optimizer(learning_rate):
    import mlx.optimizers as optim
    return optim.Adam(learning_rate=learning_rate)


def train_variant(snapshot:Path,sft_path:Path,out:Path,*,learning_rate:float,max_microsteps:int,
                  checkpoints:tuple[int,...],fingerprint:str,progress_root:Path,resources)->dict:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten,tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from .training_loss import assistant_loss
    configure_mlx_memory(mx)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arm=out.name.removeprefix('train-')
    config=recipe(learning_rate,max_microsteps,checkpoints,fingerprint,sha(sft_path))
    cp=out/'config.json'
    if cp.exists():validate_config(json.loads(cp.read_text()),config)
    cp.write_text(json.dumps(config,indent=2))
    update_state(progress_root,stage='certify' if max_microsteps==200 else 'training_'+arm,status='running',
                 step=0,total_steps=max_microsteps,detail=f'{arm} · lr={learning_rate}',message='加载原模型，按冻结配方独立训练。')
    resources.check();model,tokenizer=load(str(snapshot));resources.check();model.freeze()
    mx.random.seed(SEED);linear_to_lora_layers(model,8,LORA)
    optimizer=make_optimizer(learning_rate)
    rows=read_rows(sft_path)
    if len(rows)!=1200:raise ValueError('Expected 1200 training rows')
    encoded=[encode_row(r,tokenizer,1217) for r in rows]
    schedule=exposure_schedule(len(rows),1,SEED)
    exposure_ids=[rows[i]['id'] for i in schedule]
    schedule_sha=hashlib.sha256(json.dumps(exposure_ids).encode()).hexdigest()
    (out/'exposure.json').write_text(json.dumps(exposure_ids))
    events_path=out/'events.jsonl';events=read_rows(events_path) if events_path.exists() else []
    states=sorted([p for p in out.glob('step-*') if (p/'state.json').exists()],key=lambda p:int(p.name[5:]))
    initial=restore_training_state(states[-1],model,optimizer,fingerprint) if states else 0
    if initial>max_microsteps:raise ValueError('Checkpoint exceeds budget')
    events=[e for e in events if e['microstep']<=initial]
    events_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    state=[model.state,optimizer.state,mx.random.state]
    loss_grad=nn.value_and_grad(model,assistant_loss)

    @partial(mx.compile,inputs=state,outputs=state)
    def step(batch,lengths,prev,update):
        (loss,ntokens),grad=loss_grad(model,batch,lengths)
        if prev is not None:grad=tree_map(lambda a,b:a+b,grad,prev)
        if update:
            optimizer.update(model,tree_map(lambda x:x/8,grad));grad=None
        return loss,ntokens,grad

    model.train();grad=None;losses=[];durations=[];supervised=0
    mx.reset_peak_memory();started=time.perf_counter()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    update_state(progress_root,step=initial,loss_history=[])
    for index in range(initial,max_microsteps):
        resources.check();tokens,offset=encoded[schedule[index]]
        padded=min(1217,1+32*math.ceil(len(tokens)/32))
        if padded<len(tokens):raise ValueError('Silent truncation prohibited')
        arr=np.zeros((1,padded),dtype=np.int32);arr[0,:len(tokens)]=tokens
        tick=time.perf_counter()
        loss,ntokens,grad=step(mx.array(arr),mx.array([[offset,len(tokens)]]),grad,(index+1)%8==0)
        mx.eval(state,loss,ntokens,grad)
        duration=time.perf_counter()-tick;value=float(loss.item())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        losses.append(value);durations.append(duration);supervised+=int(ntokens.item());micro=index+1
        if micro%8==0:
            if int(optimizer.step.item())!=micro//8:raise ValueError('Optimizer update mismatch')
            sample=resources.samples[-1]
            event=dict(microstep=micro,optimizer_updates=micro//8,loss=sum(losses)/len(losses),seconds=sum(durations),
                wall_seconds=time.perf_counter()-started,supervised_tokens=supervised,
                mlx_active_gb=mx.get_active_memory()/1e9,mlx_peak_gb=mx.get_peak_memory()/1e9,mlx_cache_gb=mx.get_cache_memory()/1e9,
                rss_gib=sample['rss_bytes']/1024**3,swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
            with events_path.open('a') as handle:handle.write(json.dumps(event)+'\n')
            events.append(event);losses=[];durations=[];supervised=0
            update_state(progress_root,step=micro,metrics=dict(arm=arm,learning_rate=learning_rate,**event),
                         loss_history=[dict(step=e['microstep'],loss=e['loss']) for e in events])
            print(json.dumps(dict(arm=arm,**event)),flush=True);mx.clear_cache()
        # Historical code also saved 1000; preserve its side effects but evaluate only declared points.
        if micro%200==0 or micro in checkpoints or micro==max_microsteps:
            save_training_state(out/f'step-{micro:04d}',model,optimizer,micro,fingerprint)
    result=dict(status='complete',arm=arm,microsteps=max_microsteps,optimizer_updates=max_microsteps//8,
        trainable_parameters=trainable,schedule_sha256=schedule_sha,source_fingerprint=fingerprint,learning_rate=learning_rate,
        active_training_seconds=sum(e['seconds'] for e in events),events=len(events),
        total_input_tokens=sum(encoded[i][1] for i in schedule[:max_microsteps]),
        mlx_peak_gb=max((e['mlx_peak_gb'] for e in events),default=0),resumed_from_microstep=initial)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    del model,optimizer,grad,state,loss_grad,step
    mx.clear_cache()
    return result


def certify_control(candidate:Path,historical:Path)->dict:
    import mlx.core as mx
    candidate,historical=Path(candidate),Path(historical)
    c=json.loads((candidate/'step-0200/state.json').read_text());h=json.loads((historical/'step-0200/state.json').read_text())
    if (c['microstep'],c['optimizer_updates'])!=(200,25) or (h['microstep'],h['optimizer_updates'])!=(200,25):raise ValueError('Control cursor mismatch')
    details={}
    for name in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:
        paths=[p/'step-0200'/name for p in [candidate,historical]]
        if any(sha(p)!=s['hashes'][name] for p,s in zip(paths,[c,h])):raise ValueError('Corrupt control checkpoint')
        a,b=[mx.load(str(p)) for p in paths]
        equal=set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and bool(mx.array_equal(a[k],b[k])) for k in a)
        details[name]=dict(equal=equal,arrays=len(a),candidate_sha256=sha(paths[0]),historical_sha256=sha(paths[1]))
    ce=[(r['microstep'],r['loss'],r['supervised_tokens']) for r in read_rows(candidate/'events.jsonl') if r['microstep']<=200]
    he=[(r['microstep'],r['loss'],r['supervised_tokens']) for r in read_rows(historical/'events.jsonl') if r['microstep']<=200]
    oldconfig=json.loads((historical/'config.json').read_text());newconfig=json.loads((candidate/'config.json').read_text())
    fixed=['seed','batch_size','grad_accumulation_steps','learning_rate','optimizer','num_layers','lora_parameters','max_seq_length','model','revision','mlx_free_cache_limit_bytes']
    config_equal=all(oldconfig[k]==newconfig[k] for k in fixed)
    report=dict(status='passed' if all(v['equal'] for v in details.values()) and ce==he and config_equal else 'failed',
                arrays=details,losses_and_supervised_tokens_equal=ce==he,recipe_equal=config_equal,updates=25)
    if report['status']!='passed':
        (candidate/'equivalence-failed.json').write_text(json.dumps(report,indent=2))
        raise ValueError('Historical control differs; see equivalence-failed.json')
    return report
