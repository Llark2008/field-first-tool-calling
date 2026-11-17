"""Matched 600-example training; frozen numerical primitives, new exposure identity."""
import json
import math
import time
import tempfile
from pathlib import Path
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from toolcall_lab.run_clarification_ablation import write_json,read_json as read
from toolcall_lab.train_formal import RECIPE,SEED,LORA,configure_mlx_memory,save_training_state,restore_training_state
from toolcall_lab.train_mechanism import build_step
from toolcall_lab.training_spike import encode_row
from .data import digest

def recipe(arm,fingerprint,sft_sha256,cap):
    if arm not in ['J','F'] or cap>2048:raise ValueError('Invalid training configuration')
    return dict(RECIPE,arm=arm,fingerprint=fingerprint,sft_sha256=sft_sha256,max_seq_length=cap,
        max_microsteps=600,checkpoints=[200,400,600],normalization='sample',bias_correction=False,betas=[.9,.999],eps=1e-8)

def identity(config):return dict(arm=config['arm'],config_sha256=digest(config))

def verify_checkpoint(path,config,step):
    path=Path(path);state=read(path/'state.json')
    if step not in config['checkpoints'] or (state['fingerprint'],state['microstep'],state['optimizer_updates'])!=(config['fingerprint'],step,step//8):raise ValueError('Checkpoint cursor/experiment changed')
    if state.get('fields_sft')!=identity(config):raise ValueError('Checkpoint recipe changed')
    if set(state['hashes'])!={'adapters.safetensors','optimizer.safetensors','rng.safetensors'}:raise ValueError('Incomplete state')
    for name,value in state['hashes'].items():
        if sha(path/name)!=value:raise ValueError('Checkpoint bytes changed')
    expected=dict(fine_tune_type='lora',num_layers=config['num_layers'],lora_parameters=config['lora_parameters'])
    if read(path/'adapter_config.json')!=expected or sha(path/'adapter_config.json')!=state.get('adapter_config_sha256'):raise ValueError('Adapter recipe changed')

def checkpoints(out,config):
    states=sorted(p for p in out.glob('step-*') if (p/'state.json').exists())
    for p in states:verify_checkpoint(p,config,int(p.name[5:]))
    return states

def save_checkpoint(destination,model,optimizer,micro,config):
    # A staged directory is never discoverable as a resumable checkpoint.
    staging=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=destination.parent))
    save_training_state(staging,model,optimizer,micro,config['fingerprint'])
    metadata=read(staging/'state.json');metadata.update(fields_sft=identity(config),adapter_config_sha256=sha(staging/'adapter_config.json'))
    write_json(staging/'state.json',metadata);verify_checkpoint(staging,config,micro)
    if destination.exists():raise ValueError('Refuse to replace an existing published checkpoint')
    staging.rename(destination)

def check_training(root,f,arm):
    out=root/f'train-{arm}';config=recipe(arm,f['fingerprint'],sha(root/f'sft-{arm}.jsonl'),f['cap'])
    if read(out/'config.json')!=config:raise ValueError('Training recipe changed')
    for step in config['checkpoints']:verify_checkpoint(out/f'step-{step:04d}',config,step)
    rows=read_rows(root/f'sft-{arm}.jsonl');exposure=[r['id'] for r in rows]
    if read(out/'exposure.json')!=exposure:raise ValueError('Exposure order changed')
    events=read_rows(out/'events.jsonl');summary=read(out/'summary.json');a=f['audit']['tokens'][arm]
    expected=dict(status='complete',arm=arm,microsteps=600,optimizer_updates=75,events=75,source_fingerprint=f['fingerprint'],consumed_ids_sha256=digest(exposure),total_input_tokens=a['input_tokens'],total_answer_tokens=a['answer_tokens'])
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Training summary changed')
    if [e['microstep'] for e in events]!=list(range(8,601,8)) or sum(e['supervised_tokens'] for e in events)!=a['answer_tokens']:raise ValueError('Training event grid/count changed')
    if any(not math.isfinite(e['loss']) or e['optimizer_updates']!=e['microstep']//8 for e in events):raise ValueError('Invalid training event')
    return summary

def train(root,f,arm,resources):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    configure_mlx_memory(mx);out=root/f'train-{arm}';out.mkdir(exist_ok=True)
    config=recipe(arm,f['fingerprint'],sha(root/f'sft-{arm}.jsonl'),f['cap']);cp=out/'config.json'
    if cp.exists() and read(cp)!=config:raise ValueError('Cannot resume changed recipe')
    write_json(cp,config)
    states=checkpoints(out,config)
    if (out/'summary.json').exists():return check_training(root,f,arm)
    update_state(root,stage='training_'+arm,status='running',detail=arm+' · 75更新 / 600条',step=0,total_steps=600,loss_history=[],message='独立加载同一原模型，按固定相同样本顺序训练。')
    resources.check();model,tokenizer=load(f['snapshot']);resources.check();model.freeze()
    mx.random.seed(SEED);linear_to_lora_layers(model,8,LORA)
    optimizer=optim.Adam(learning_rate=1e-4,betas=[.9,.999],eps=1e-8,bias_correction=False)
    rows=read_rows(root/f'sft-{arm}.jsonl');encoded=[encode_row(r,tokenizer,f['cap']) for r in rows]
    if len(rows)!=600:raise ValueError('Wrong exposure count')
    write_json(out/'exposure.json',[r['id'] for r in rows])
    initial=restore_training_state(states[-1],model,optimizer,f['fingerprint']) if states else 0
    path=out/'events.jsonl';events=read_rows(path) if path.exists() else [];events=[e for e in events if e['microstep']<=initial]
    if len(events)!=initial//8:raise ValueError('Resume event cursor mismatch')
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    step,state=build_step(model,optimizer,'sample');model.train();grad=None;window=[];mx.reset_peak_memory()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    for i in range(initial,600):
        resources.check();tokens,offset=encoded[i]
        if i%8==0:window_tokens=sum(len(t)-o for t,o in encoded[i:i+8])
        padded=min(f['cap'],1+32*math.ceil(len(tokens)/32))
        if padded<len(tokens):raise ValueError('Truncation prohibited')
        arr=np.zeros((1,padded),dtype=np.int32);arr[0,:len(tokens)]=tokens;tick=time.perf_counter()
        loss,n,grad=step(mx.array(arr),mx.array([[offset,len(tokens)]]),grad,window_tokens,(i+1)%8==0)
        mx.eval(state,loss,n,grad);value=float(loss.item());count=int(n.item());seconds=time.perf_counter()-tick
        if not math.isfinite(value) or count!=len(tokens)-offset:raise ValueError('Invalid loss/token count')
        window.append((value,count,seconds));micro=i+1
        if micro%8==0:
            if int(optimizer.step.item())!=micro//8 or len(window)!=8 or sum(x[1] for x in window)!=window_tokens:raise ValueError('Accumulation mismatch')
            sample=resources.samples[-1]
            event=dict(microstep=micro,optimizer_updates=micro//8,loss=sum(x[0] for x in window)/8,supervised_tokens=window_tokens,
                seconds=sum(x[2] for x in window),mlx_peak_gb=mx.get_peak_memory()/1e9,rss_gib=sample['rss_bytes']/1024**3,
                swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
            with path.open('a') as h:h.write(json.dumps(event)+'\n')
            events.append(event);window=[]
            update_state(root,step=micro,metrics=dict(arm=arm,**event),loss_history=[dict(step=e['optimizer_updates'],loss=e['loss']) for e in events])
            print(json.dumps(dict(arm=arm,**event)),flush=True);mx.clear_cache()
        if micro in config['checkpoints']:
            save_checkpoint(out/f'step-{micro:04d}',model,optimizer,micro,config)
    resources.check();a=f['audit']['tokens'][arm]
    summary=dict(status='complete',arm=arm,microsteps=600,optimizer_updates=75,events=75,source_fingerprint=f['fingerprint'],
        consumed_ids_sha256=digest([r['id'] for r in rows]),total_input_tokens=a['input_tokens'],total_answer_tokens=a['answer_tokens'],
        active_training_seconds=sum(e['seconds'] for e in events),mlx_peak_gb=max(e['mlx_peak_gb'] for e in events),trainable_parameters=trainable,resumed_from_microstep=initial)
    write_json(out/'summary.json',summary)
    del model,tokenizer,optimizer,grad,state,step;mx.clear_cache()
    return check_training(root,f,arm)
