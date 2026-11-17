"""Continue exact optimizer state, changing neither numerical loss nor data."""
import json
import math
import os
import tempfile
import time
from pathlib import Path
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.train_formal import SEED,LORA,configure_mlx_memory,restore_training_state
from toolcall_lab.train_mechanism import build_step
from toolcall_lab.training_spike import encode_row
from toolcall_lab.live_progress import update_state
from experiments.fields_sft.train import checkpoints,save_checkpoint,verify_checkpoint
from experiments.field_action_diagnostics.run import read,write,immutable
from .core import recipe,check_lr

def atomic_events(path,events):
    fd,name=tempfile.mkstemp(prefix='.events-',dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(fd,'w') as handle:
            handle.write(''.join(json.dumps(e)+'\n' for e in events));handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()

def load_events(path,committed_micro):
    lines=path.read_text().splitlines(keepends=True);events=[]
    for i,line in enumerate(lines):
        try:events.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n') or (events[-1]['microstep'] if events else 0)<committed_micro:
                raise ValueError('Invalid committed training journal')
    if (events[-1]['microstep'] if events else 0)<committed_micro:raise ValueError('Missing committed training events')
    return events

def resume_events(events,micro):
    kept=[e for e in events if e['microstep']<=micro]
    if [e['microstep'] for e in kept]!=list(range(8,micro+1,8)):raise ValueError('Resume event cursor mismatch')
    if any(e['optimizer_updates']!=e['microstep']//8 or not math.isfinite(e['loss']) or e['loss']<0 or e['supervised_tokens']<=0 for e in kept):raise ValueError('Invalid training event')
    return kept

def resume_origin(root,parent,arm,states):
    if states:return states[-1],int(states[-1].name[5:]),False
    if arm=='A':return parent/'train-F/step-0600',600,True
    return None,0,False

def checkpoint_path(root,f,arm,epoch):
    return Path(f['parent'])/'train-F/step-0600' if arm=='A' and epoch==1 else root/f'train-{arm}/step-{600*epoch:04d}'

def config_for(root,f,arm):return recipe(arm,f['fingerprint'],sha(Path(f['parent'])/'sft-F.jsonl'),f['cap'])

def verify_epoch(root,f,arm,epoch):
    config=config_for(root,f,arm);path=checkpoint_path(root,f,arm,epoch)
    if arm=='A' and epoch==1:
        from experiments.fields_sft.train import recipe as old_recipe
        verify_checkpoint(path,old_recipe('F',f['parent_fingerprint'],config['sft_sha256'],f['cap']),600)
    else:verify_checkpoint(path,config,600*epoch)
    return path

def events_for(root,f,arm):
    p=root/f'train-{arm}/events.jsonl'
    if p.exists():
        states=checkpoints(p.parent,config_for(root,f,arm))
        _,committed,_=resume_origin(root,Path(f['parent']),arm,states)
        return load_events(p,committed)
    return read_rows(Path(f['parent'])/'train-F/events.jsonl') if arm=='A' else []

def validate_training(root,f,arm,epoch):
    verify_epoch(root,f,arm,epoch);config=config_for(root,f,arm)
    if arm!='A' or epoch>1:
        if read(root/f'train-{arm}/config.json')!=config:raise ValueError('Training config changed')
    events=resume_events(events_for(root,f,arm),epoch*600)
    if sum(e['supervised_tokens'] for e in events)!=epoch*f['audit']['answer_tokens_per_epoch']:raise ValueError('Training target exposure changed')
    if arm=='A' and events[:75]!=read_rows(Path(f['parent'])/'train-F/events.jsonl'):raise ValueError('Inherited epoch1 events changed')
    return events

def train_epoch(root,f,arm,epoch,resources):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    target=epoch*600;parent=Path(f['parent']);destination=checkpoint_path(root,f,arm,epoch)
    if destination.exists():validate_training(root,f,arm,epoch);return
    out=root/f'train-{arm}';out.mkdir(exist_ok=True);config=config_for(root,f,arm);immutable(out/'config.json',config)
    states=checkpoints(out,config);origin,initial,inherited=resume_origin(root,parent,arm,states)
    if initial>=target:raise ValueError('Resume cursor exceeds target epoch')
    update_state(root,stage='training',status='running',detail=f'{arm} 学习率{config["learning_rate"]} · 第{epoch}轮',step=initial,total_steps=target,
        message='固定600条原数据与两段答案；优化器连续累计，每200微步保存恢复点。')
    resources.check();configure_mlx_memory(mx);model,tokenizer=load(f['snapshot']);resources.check();model.freeze()
    mx.random.seed(SEED);linear_to_lora_layers(model,8,LORA)
    optimizer=optim.Adam(learning_rate=config['learning_rate'],betas=config['betas'],eps=config['eps'],bias_correction=config['bias_correction'])
    rows=read_rows(parent/'sft-F.jsonl');encoded=[encode_row(r,tokenizer,f['cap']) for r in rows]
    if len(rows)!=600 or [r['id'] for r in rows]*3!=read(root/'exposure.json'):raise ValueError('Training data/order changed')
    if origin:
        actual=restore_training_state(origin,model,optimizer,f['parent_fingerprint'] if inherited else f['fingerprint'])
        if actual!=initial:raise ValueError('Restored cursor mismatch')
    check_lr(float(optimizer.learning_rate.item()),config['learning_rate'])
    path=out/'events.jsonl';events=resume_events(load_events(path,initial) if path.exists() else (read_rows(parent/'train-F/events.jsonl') if arm=='A' else []),initial)
    atomic_events(path,events)
    immutable(out/f'resume-{initial:04d}-to-{target:04d}.json',dict(origin=str(origin) if origin else None,microstep=initial,inherited=inherited,learning_rate=float(optimizer.learning_rate.item()),optimizer_step=int(optimizer.step.item())))
    step,state=build_step(model,optimizer,'sample');model.train();grad=None;window=[];mx.reset_peak_memory()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    if trainable!=458752:raise ValueError('Trainable parameter count changed')
    try:
        for i in range(initial,target):
            resources.check();tokens,offset=encoded[i%600]
            if i%8==0:window_tokens=sum(len(encoded[j%600][0])-encoded[j%600][1] for j in range(i,i+8))
            padded=min(f['cap'],1+32*math.ceil(len(tokens)/32))
            if padded<len(tokens):raise ValueError('Truncation prohibited')
            arr=np.zeros((1,padded),dtype=np.int32);arr[0,:len(tokens)]=tokens;tick=time.perf_counter()
            loss,n,grad=step(mx.array(arr),mx.array([[offset,len(tokens)]]),grad,window_tokens,(i+1)%8==0)
            mx.eval(state,loss,n,grad);value=float(loss.item());count=int(n.item());seconds=time.perf_counter()-tick
            if not math.isfinite(value) or count!=len(tokens)-offset:raise ValueError('Invalid training loss/token count')
            window.append((value,count,seconds));micro=i+1
            if micro%8==0:
                if int(optimizer.step.item())!=micro//8 or len(window)!=8 or sum(x[1] for x in window)!=window_tokens:raise ValueError('Accumulation mismatch')
                sample=resources.samples[-1]
                event=dict(microstep=micro,optimizer_updates=micro//8,loss=sum(x[0] for x in window)/8,supervised_tokens=window_tokens,
                    seconds=sum(x[2] for x in window),mlx_peak_gb=mx.get_peak_memory()/1e9,rss_gib=sample['rss_bytes']/1024**3,
                    swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
                with path.open('a') as h:h.write(json.dumps(event)+'\n');h.flush()
                events.append(event);window=[]
                update_state(root,step=micro,metrics=dict(arm=arm,epoch=epoch,learning_rate=config['learning_rate'],**event),**{'loss_'+arm:events})
                print(json.dumps(dict(arm=arm,epoch=epoch,**event)),flush=True);mx.clear_cache()
            if micro in config['checkpoints']:save_checkpoint(out/f'step-{micro:04d}',model,optimizer,micro,config)
        resources.check();events=validate_training(root,f,arm,epoch)
        write(out/f'epoch-{epoch}-summary.json',dict(arm=arm,epoch=epoch,updates=target//8,trainable_parameters=trainable,
            total_training_seconds=sum(e['seconds'] for e in events),new_training_seconds=sum(e['seconds'] for e in events[(75 if arm=='A' else 0):]),
            peak_mlx_gb=max(e['mlx_peak_gb'] for e in events),resumed_from=initial))
    finally:
        del model,tokenizer,optimizer,grad,state,step;mx.clear_cache()
