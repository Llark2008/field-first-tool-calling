"""Matched-exposure QLoRA with continuous optimizer and exact-state checkpoints."""
import hashlib
import json
import math
import random
import time
from functools import partial
from pathlib import Path

from .formal_data import sha
from .live_progress import update_state
from .training_spike import encode_row, MODEL_ID, MODEL_REVISION

SEED=20261113
STEPS=(600,1200,2400)
CACHE_BYTES=256*1024**2
LORA=dict(rank=8,scale=20.0,dropout=0.0,keys=['self_attn.q_proj','self_attn.v_proj'])
RECIPE=dict(seed=SEED,batch_size=1,grad_accumulation_steps=8,learning_rate=1e-4,
    optimizer='Adam',num_layers=8,lora_parameters=LORA,max_microsteps=2400,checkpoints=list(STEPS),
    model=MODEL_ID,revision=MODEL_REVISION,mlx_free_cache_limit_bytes=CACHE_BYTES)


def exposure_schedule(n,epochs,seed):
    rng=random.Random(seed); result=[]
    for _ in range(epochs):
        order=list(range(n)); rng.shuffle(order); result.extend(order)
    return result


def configure_mlx_memory(mx):
    """Bound only unused allocator cache; never raise memory/wired limits."""
    mx.set_cache_limit(CACHE_BYTES)
    mx.clear_cache()


def verify_frozen(root,manifest):
    for name,entry in manifest['files'].items():
        if sha(Path(root)/name)!=entry['sha256']: raise ValueError('Frozen file changed: '+name)


def select_checkpoint(baseline,a,b):
    threshold=baseline['complete']['success']/baseline['complete']['n']-0.05
    candidates=[]
    if set(a)!=set(STEPS) or set(b)!=set(STEPS): raise ValueError('All three matched checkpoints are required')
    for step in STEPS:
        arms=[a[step],b[step]]
        rates=[m['complete']['success']/m['complete']['n'] for m in arms]
        macro=sum(sum(m[c]['success']/m[c]['n'] for c in ['complete','natural','unsupported'])/3 for m in arms)/2
        candidates.append(dict(step=step,eligible=all(x+1e-12>=threshold for x in rates),
            mean_macro=macro,complete_rates=rates))
    eligible=[c for c in candidates if c['eligible']]
    winner=max(eligible,key=lambda c:(c['mean_macro'],-c['step'])) if eligible else None
    return dict(selected_step=winner['step'] if winner else None,threshold=threshold,candidates=candidates,
        rule='Both arms complete accuracy >= base v2 minus 5pp; maximize mean 3-category macro; earlier tie.')


def save_training_state(path,model,optimizer,microstep,fingerprint):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    mx.eval(model.trainable_parameters(),optimizer.state,mx.random.state)
    if microstep%8: raise ValueError('Can save only at completed accumulation boundaries')
    adapters=dict(tree_flatten(model.trainable_parameters()))
    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in adapters.values()): raise FloatingPointError('Nonfinite adapter')
    mx.save_safetensors(str(path/'adapters.safetensors'),adapters)
    mx.save_safetensors(str(path/'optimizer.safetensors'),{k:mx.array(v) for k,v in tree_flatten(optimizer.state)})
    mx.save_safetensors(str(path/'rng.safetensors'),{str(i):v for i,v in enumerate(mx.random.state)})
    (path/'adapter_config.json').write_text(json.dumps(dict(fine_tune_type='lora',num_layers=8,lora_parameters=LORA),indent=2))
    metadata=dict(microstep=microstep,optimizer_updates=int(optimizer.step.item()),fingerprint=fingerprint,
                  hashes={name:sha(path/name) for name in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']})
    tmp=path/'state.tmp'; tmp.write_text(json.dumps(metadata,indent=2)); tmp.replace(path/'state.json')


def restore_training_state(path,model,optimizer,fingerprint):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    path=Path(path); state=json.loads((path/'state.json').read_text())
    if state['fingerprint']!=fingerprint: raise ValueError('Resume fingerprint mismatch')
    if state['microstep']%8 or state['optimizer_updates']!=state['microstep']//8: raise ValueError('Resume update-count mismatch')
    for name,value in state['hashes'].items():
        if sha(path/name)!=value: raise ValueError('Incomplete/corrupt resume state')
    model.load_weights(str(path/'adapters.safetensors'),strict=False)
    optimizer.state=tree_unflatten(list(mx.load(str(path/'optimizer.safetensors')).items()))
    rng=mx.load(str(path/'rng.safetensors'))
    # MLX 0.32.4 exposes a read-only state sentinel. key(seed) stores the
    # uint64 seed as high/low uint32 words, so reseeding reconstructs that key.
    # https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/random.cpp#L28-L32
    if set(rng)!={'0'} or rng['0'].shape!=(2,) or rng['0'].dtype!=mx.uint32:
        raise ValueError('Unsupported saved PRNG format')
    hi,lo=rng['0'].tolist()
    mx.random.seed((int(hi)<<32)|int(lo))
    mx.eval(model.parameters(),optimizer.state,mx.random.state)
    return state['microstep']


def audit_tokens(dataset,tokenizer):
    results={}; encoded={}
    for arm in ['A','B']:
        rows=[json.loads(x) for x in (Path(dataset)/(arm+'-sft.jsonl')).read_text().splitlines()]
        enc=[encode_row(row,tokenizer,2048) for row in rows]; encoded[arm]=enc
        lengths=[len(t) for t,o in enc]; answers=[len(t)-o for t,o in enc]
        results[arm]=dict(n=len(enc),min_tokens=min(lengths),max_tokens=max(lengths),
            mean_tokens=sum(lengths)/len(lengths),total_input_tokens=sum(o for t,o in enc),
            total_answer_tokens=sum(answers),prefixes_verified=len(enc),min_answer_tokens=min(answers))
    assert [t[o:] for t,o in encoded['A']]==[t[o:] for t,o in encoded['B']], 'A/B target tokens differ'
    # Prefixes for all evaluation prompts are audited without generating outputs.
    from .controls import build_control_messages
    for split in ['dev','test']:
        cases=[json.loads(x) for x in (Path(dataset)/(split+'.jsonl')).read_text().splitlines()]
        for prompt in ['v2','no_examples','checklist']:
            lengths=[len(tokenizer.apply_chat_template(build_control_messages(c,prompt),tokenize=True,
                return_dict=False,add_generation_prompt=True,enable_thinking=False)) for c in cases]
            if max(lengths)>2048: raise ValueError('Evaluation input exceeds audited budget')
            results[split+'/'+prompt]=dict(n=len(lengths),max_input_tokens=max(lengths),mean_input_tokens=sum(lengths)/len(lengths))
    cap=1+32*math.ceil(max(results[a]['max_tokens'] for a in ['A','B'])/32)
    if cap>2048: cap=2048
    return dict(sequence_cap=cap,details=results)


def train_arm(snapshot,dataset,arm,out,fingerprint,token_audit,progress_root,resources,preflight=False):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten,tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from .training_loss import assistant_loss
    configure_mlx_memory(mx)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config=dict(RECIPE,arm=arm,max_seq_length=token_audit['sequence_cap'],fingerprint=fingerprint)
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('Training configuration changed')
    config_path.write_text(json.dumps(config,indent=2))
    model,tokenizer=load(str(snapshot)); model.freeze()
    mx.random.seed(SEED)
    linear_to_lora_layers(model,8,LORA)
    optimizer=optim.Adam(learning_rate=1e-4)
    rows=[json.loads(x) for x in (Path(dataset)/(arm+'-sft.jsonl')).read_text().splitlines()]
    encoded=[encode_row(row,tokenizer,token_audit['sequence_cap']) for row in rows]
    schedule=exposure_schedule(len(rows),2,SEED)
    schedule_sha=hashlib.sha256(json.dumps([rows[i]['id'] for i in schedule]).encode()).hexdigest()
    limit=8 if preflight else 2400
    events_path=out/'events.jsonl'
    events=[json.loads(x) for x in events_path.read_text().splitlines()] if events_path.exists() else []
    states=sorted([p for p in out.glob('step-*') if (p/'state.json').exists()],key=lambda p:int(p.name[5:]))
    initial=restore_training_state(states[-1],model,optimizer,fingerprint) if states else 0
    events=[e for e in events if e['microstep']<=initial]
    events_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    state=[model.state,optimizer.state,mx.random.state]
    loss_grad=nn.value_and_grad(model,assistant_loss)

    @partial(mx.compile,inputs=state,outputs=state)
    def step(batch,lengths,prev,update):
        (loss,ntokens),grad=loss_grad(model,batch,lengths)
        if prev is not None: grad=tree_map(lambda a,b:a+b,grad,prev)
        if update:
            optimizer.update(model,tree_map(lambda x:x/8,grad)); grad=None
        return loss,ntokens,grad

    model.train(); grad=None; losses=[]; durations=[]; supervised=0
    mx.reset_peak_memory(); started=time.perf_counter()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    stage='audit' if preflight else 'training_'+arm
    update_state(progress_root,stage=stage,status='running',step=initial,total_steps=limit,
        loss_history=[],detail=f'{arm} · 每 8 个微步更新一次',
        message='短程管线与速度检查；该适配器不会用于正式结果。' if preflight else f'{arm} 组从原模型独立训练；持续保留优化器、随机状态与数据游标。')
    for index in range(initial,limit):
        resources.check(); tokens,offset=encoded[schedule[index]]
        padded=min(token_audit['sequence_cap'],1+32*math.ceil(len(tokens)/32))
        if padded<len(tokens): raise ValueError('Silent truncation prohibited')
        arr=np.zeros((1,padded),dtype=np.int32); arr[0,:len(tokens)]=tokens
        tick=time.perf_counter()
        loss,ntokens,grad=step(mx.array(arr),mx.array([[offset,len(tokens)]]),grad,(index+1)%8==0)
        mx.eval(state,loss,ntokens,grad)
        duration=time.perf_counter()-tick; value=float(loss.item())
        if not math.isfinite(value): raise FloatingPointError('Nonfinite training loss')
        losses.append(value); durations.append(duration); supervised+=int(ntokens.item())
        micro=index+1
        if micro%8==0:
            if int(optimizer.step.item())!=micro//8: raise ValueError('Optimizer update mismatch')
            sample=resources.samples[-1]
            event=dict(microstep=micro,optimizer_updates=micro//8,loss=sum(losses)/len(losses),
                seconds=sum(durations),wall_seconds=time.perf_counter()-started,supervised_tokens=supervised,
                mlx_active_gb=mx.get_active_memory()/1e9,mlx_peak_gb=mx.get_peak_memory()/1e9,
                mlx_cache_gb=mx.get_cache_memory()/1e9,
                rss_gib=sample['rss_bytes']/1024**3,
                swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
            with events_path.open('a') as handle: handle.write(json.dumps(event)+'\n')
            events.append(event); losses=[]; durations=[]; supervised=0
            update_state(progress_root,step=micro,metrics=event,loss_history=[dict(step=e['microstep'],loss=e['loss']) for e in events])
            print(json.dumps(dict(arm=arm,**event)),flush=True)
            mx.clear_cache()
        if micro%200==0 or micro in STEPS or micro==limit:
            save_training_state(out/f'step-{micro:04d}',model,optimizer,micro,fingerprint)
    result=dict(status='complete',arm=arm,microsteps=limit,optimizer_updates=limit//8,
        trainable_parameters=trainable,schedule_sha256=schedule_sha,source_fingerprint=fingerprint,
        active_training_seconds=sum(e['seconds'] for e in events),events=len(events),
        mlx_peak_gb=max((e['mlx_peak_gb'] for e in events),default=0),resumed_from_microstep=initial)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    del model,optimizer,grad,state,loss_grad,step
    mx.clear_cache()
    return result
