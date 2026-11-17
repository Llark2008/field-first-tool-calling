"""Fixed action-position objective, separate from the frozen historical source."""
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path

from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from toolcall_lab.train_formal import (SEED, LORA, RECIPE, configure_mlx_memory,
    exposure_schedule, save_training_state, restore_training_state)
from toolcall_lab.training_spike import encode_row


def recipe(fingerprint, sft_sha256):
    return dict(RECIPE, arm='W20', action_weight=.20, normalization='action_rest_groups',
        bias_correction=False, betas=[.9,.999], eps=1e-8, max_seq_length=1217,
        max_microsteps=600, checkpoints=[200,400,600], fingerprint=fingerprint, sft_sha256=sft_sha256)


def checkpoint_identity(config):
    return dict(arm=config['arm'], config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def verify_checkpoint(path, config, step):
    path=Path(path); state=json.loads((path/'state.json').read_text())
    if step not in config['checkpoints'] or (state['fingerprint'],state['microstep'],state['optimizer_updates'])!=(config['fingerprint'],step,step//8):
        raise ValueError('Checkpoint experiment or cursor mismatch')
    if state.get('action_weighting')!=checkpoint_identity(config): raise ValueError('Weighted recipe mismatch')
    if set(state['hashes'])!={'adapters.safetensors','optimizer.safetensors','rng.safetensors'}: raise ValueError('Incomplete checkpoint')
    for name,digest in state['hashes'].items():
        if sha(path/name)!=digest: raise ValueError('Checkpoint file changed: '+name)
    adapter=path/'adapter_config.json'
    expected=dict(fine_tune_type='lora',num_layers=config['num_layers'],lora_parameters=config['lora_parameters'])
    if json.loads(adapter.read_text())!=expected or sha(adapter)!=state.get('adapter_config_sha256'):
        raise ValueError('Inference adapter configuration changed')


def decision_index(tokens, offset, prefix, decision_ids, gold_action):
    index=offset+len(prefix)
    if offset<1 or tokens[offset:index]!=list(prefix) or index>=len(tokens)-1:
        raise ValueError('Action prefix or target boundary mismatch')
    if gold_action not in decision_ids or tokens[index]!=decision_ids[gold_action]:
        raise ValueError('Action decision target mismatch')
    if len(set(decision_ids.values()))!=3: raise ValueError('Actions do not have distinct first tokens')
    return index


def action_vocabulary(tokenizer):
    prefix=tokenizer.encode('{"action":"',add_special_tokens=False)
    ids={}
    for action in ['call','clarify','unsupported']:
        encoded=tokenizer.encode('{"action":"'+action+'"}',add_special_tokens=False)
        if encoded[:len(prefix)]!=prefix: raise ValueError('Tokenizer merges across decision boundary')
        ids[action]=encoded[len(prefix)]
    if len(set(ids.values()))!=3: raise ValueError('Non-distinct action tokens')
    return prefix,ids


def weighted_loss(model, batch, lengths, decisions, action_weight):
    import mlx.core as mx
    import mlx.nn as nn
    positions=mx.arange(1,batch.shape[1])[None,:]
    supervised=(positions>=lengths[:,0:1]) & (positions<lengths[:,1:2])
    decision=supervised & (positions==decisions[:,None])
    rest=supervised & ~decision
    ce=nn.losses.cross_entropy(model(batch[:,:-1]),batch[:,1:]).astype(mx.float32)
    count=supervised.sum(axis=1)
    action_loss=(ce*decision).sum(axis=1)
    rest_loss=(ce*rest).sum(axis=1)/rest.sum(axis=1)
    raw_loss=(ce*supervised).sum(axis=1)/count
    objective=(action_weight*action_loss+(1-action_weight)*rest_loss).mean()
    return objective,mx.stack([raw_loss.mean(),action_loss.mean(),rest_loss.mean(),count.sum().astype(mx.float32)])


def build_step(model, optimizer, action_weight):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map,tree_flatten
    if not 0<action_weight<1: raise ValueError('Weight must be between zero and one')
    state=[model.state,optimizer.state,mx.random.state]
    fn=lambda m,b,l,d:weighted_loss(m,b,l,d,action_weight)
    loss_grad=nn.value_and_grad(model,fn)

    @partial(mx.compile,inputs=state,outputs=state)
    def step(batch,lengths,decisions,prev,update):
        (loss,aux),grad=loss_grad(model,batch,lengths,decisions)
        if prev is not None: grad=tree_map(lambda a,b:a+b,grad,prev)
        norm=mx.array(0.,dtype=mx.float32)
        if update:
            averaged=tree_map(lambda x:x/8,grad)
            norm=mx.sqrt(sum((v.astype(mx.float32)**2).sum() for _,v in tree_flatten(averaged)))
            optimizer.update(model,averaged); grad=None
        return loss,aux,grad,norm
    return step,state


def train(snapshot,sft,out,*,fingerprint,progress_root,resources):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    configure_mlx_memory(mx)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config=recipe(fingerprint,sha(sft)); cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('Training configuration changed')
    cp.write_text(json.dumps(config,indent=2))
    states=sorted([p for p in out.glob('step-*') if (p/'state.json').exists()])
    for checkpoint in states: verify_checkpoint(checkpoint,config,int(checkpoint.name[5:]))
    if (out/'summary.json').exists():
        result=json.loads((out/'summary.json').read_text())
        if len(states)!=3 or result['microsteps']!=600: raise ValueError('Incomplete saved training')
        return result
    update_state(progress_root,stage='training_W20',status='running',step=0,total_steps=600,
        detail='W20 · 动作20% / 其余80%',message='加载原模型，独立初始化动作加权训练。')
    resources.check(); model,tokenizer=load(str(snapshot)); resources.check(); model.freeze()
    mx.random.seed(SEED); linear_to_lora_layers(model,8,LORA)
    optimizer=optim.Adam(learning_rate=1e-4,betas=[.9,.999],eps=1e-8,bias_correction=False)
    rows=read_rows(sft)
    if len(rows)!=1200: raise ValueError('Expected unchanged 1200-row D0')
    encoded=[encode_row(r,tokenizer,1217) for r in rows]
    prefix,ids=action_vocabulary(tokenizer)
    decisions=[decision_index(t,o,prefix,ids,json.loads(r['messages'][-1]['content'])['action']) for r,(t,o) in zip(rows,encoded)]
    schedule=exposure_schedule(1200,1,SEED); exposure=[rows[i]['id'] for i in schedule]
    (out/'exposure.json').write_text(json.dumps(exposure))
    events_path=out/'events.jsonl'; events=read_rows(events_path) if events_path.exists() else []
    initial=restore_training_state(states[-1],model,optimizer,fingerprint) if states else 0
    events=[e for e in events if e['microstep']<=initial]
    if len(events)!=initial//8: raise ValueError('Checkpoint/event cursor mismatch')
    events_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    step,state=build_step(model,optimizer,.20); model.train(); grad=None
    window=[]; started=time.perf_counter(); mx.reset_peak_memory()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    update_state(progress_root,step=initial,loss_history=[])
    for index in range(initial,600):
        resources.check(); item=schedule[index]; tokens,offset=encoded[item]
        padded=min(1217,1+32*math.ceil(len(tokens)/32))
        if padded<len(tokens): raise ValueError('Silent truncation prohibited')
        arr=np.zeros((1,padded),dtype=np.int32); arr[0,:len(tokens)]=tokens; tick=time.perf_counter()
        loss,aux,grad,norm=step(mx.array(arr),mx.array([[offset,len(tokens)]]),mx.array([decisions[item]]),grad,(index+1)%8==0)
        mx.eval(state,loss,aux,grad,norm)
        values=[float(loss.item()),*aux.tolist(),time.perf_counter()-tick]
        if not all(math.isfinite(x) for x in values) or not math.isfinite(float(norm.item())): raise FloatingPointError('Nonfinite loss/gradient')
        if int(values[4])!=len(tokens)-offset: raise ValueError('Supervised denominator mismatch')
        window.append(values); micro=index+1
        if micro%8==0:
            if int(optimizer.step.item())!=micro//8 or len(window)!=8: raise ValueError('Accumulation mismatch')
            sample=resources.samples[-1]
            event=dict(microstep=micro,optimizer_updates=micro//8,objective_loss=sum(v[0] for v in window)/8,
                loss=sum(v[1] for v in window)/8,decision_loss=sum(v[2] for v in window)/8,
                rest_loss=sum(v[3] for v in window)/8,supervised_tokens=int(sum(v[4] for v in window)),
                seconds=sum(v[5] for v in window),gradient_norm=float(norm.item()),wall_seconds=time.perf_counter()-started,
                mlx_active_gb=mx.get_active_memory()/1e9,mlx_peak_gb=mx.get_peak_memory()/1e9,mlx_cache_gb=mx.get_cache_memory()/1e9,
                rss_gib=sample['rss_bytes']/1024**3,swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
            with events_path.open('a') as handle: handle.write(json.dumps(event)+'\n')
            events.append(event); window=[]
            update_state(progress_root,step=micro,metrics=dict(arm='W20',**event),loss_history=[dict(step=e['optimizer_updates'],loss=e['objective_loss'],raw_loss=e['loss']) for e in events])
            print(json.dumps(event),flush=True); mx.clear_cache()
        if micro in config['checkpoints']:
            saved=out/f'step-{micro:04d}'; save_training_state(saved,model,optimizer,micro,fingerprint)
            metadata=json.loads((saved/'state.json').read_text()); metadata['action_weighting']=checkpoint_identity(config)
            metadata['adapter_config_sha256']=sha(saved/'adapter_config.json')
            tmp=saved/'state.tmp'; tmp.write_text(json.dumps(metadata,indent=2)); tmp.replace(saved/'state.json')
    resources.check()
    result=dict(status='complete',arm='W20',microsteps=600,optimizer_updates=75,trainable_parameters=trainable,
        source_fingerprint=fingerprint,consumed_ids_sha256=hashlib.sha256(json.dumps(exposure[:600]).encode()).hexdigest(),
        active_training_seconds=sum(e['seconds'] for e in events),events=len(events),
        total_input_tokens=sum(encoded[i][1] for i in schedule[:600]),total_answer_tokens=sum(len(encoded[i][0])-encoded[i][1] for i in schedule[:600]),
        mlx_peak_gb=max(e['mlx_peak_gb'] for e in events),resumed_from_microstep=initial)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    del model,optimizer,grad,state,step; mx.clear_cache()
    return result
