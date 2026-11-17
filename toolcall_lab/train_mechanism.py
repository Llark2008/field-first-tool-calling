"""Isolated token-normalization and Adam-bias experiments, preserving M0 math."""
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path

from .ablation_data import read_rows
from .formal_data import sha
from .live_progress import update_state
from .train_ablation import validate_config
from .train_formal import SEED,LORA,RECIPE,configure_mlx_memory,exposure_schedule,save_training_state,restore_training_state
from .training_spike import encode_row

MODES={'M0':('sample',False),'M1':('token',False),'M2':('sample',True)}
CHECKPOINTS=(200,400,600)


def recipe(arm,limit,fingerprint,sft_sha256):
    if arm not in MODES or limit!=(200 if arm=='M0' else 600):
        raise ValueError('Unplanned mechanism or training budget')
    mode,bias=MODES[arm]
    return dict(RECIPE,arm=arm,normalization=mode,bias_correction=bias,betas=[.9,.999],eps=1e-8,
                learning_rate=1e-4,max_seq_length=1217,max_microsteps=limit,
                checkpoints=[s for s in CHECKPOINTS if s<=limit],fingerprint=fingerprint,sft_sha256=sft_sha256)


def checkpoint_identity(config):
    return dict(arm=config['arm'],normalization=config['normalization'],bias_correction=config['bias_correction'],
                config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())


def verify_checkpoint(path,config,step):
    path=Path(path);state=json.loads((path/'state.json').read_text())
    if (state['fingerprint'],state['microstep'],state['optimizer_updates'])!=(config['fingerprint'],step,step//8):
        raise ValueError('Checkpoint experiment or cursor mismatch')
    if state.get('mechanism')!=checkpoint_identity(config):raise ValueError('Checkpoint mechanism mismatch')
    if set(state['hashes'])!={'adapters.safetensors','optimizer.safetensors','rng.safetensors'}:
        raise ValueError('Incomplete checkpoint')
    for name,digest in state['hashes'].items():
        if sha(path/name)!=digest:raise ValueError('Checkpoint file changed: '+name)
    adapter_config=path/'adapter_config.json'
    expected=dict(fine_tune_type='lora',num_layers=config['num_layers'],lora_parameters=config['lora_parameters'])
    if json.loads(adapter_config.read_text())!=expected or state.get('adapter_config_sha256')!=sha(adapter_config):
        raise ValueError('Inference adapter configuration changed')


def make_optimizer(arm):
    import mlx.optimizers as optim
    if arm not in MODES:raise ValueError('Unknown mechanism')
    return optim.Adam(learning_rate=1e-4,betas=[.9,.999],eps=1e-8,bias_correction=MODES[arm][1])


def token_sum_loss(model,batch,lengths):
    import mlx.core as mx
    import mlx.nn as nn
    from .training_loss import completion_mask
    logits=model(batch[:,:-1]);targets=batch[:,1:]
    mask=completion_mask(mx.arange(1,targets.shape[1]+1),lengths[:,0:1],lengths[:,1:])
    return (nn.losses.cross_entropy(logits,targets)*mask).astype(mx.float32).sum(),mask.sum()


def build_step(model,optimizer,normalization):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from .training_loss import assistant_loss
    if normalization not in ('sample','token'):raise ValueError('Unknown normalization')
    state=[model.state,optimizer.state,mx.random.state]
    loss_grad=nn.value_and_grad(model,assistant_loss if normalization=='sample' else token_sum_loss)

    @partial(mx.compile,inputs=state,outputs=state)
    def compiled(batch,lengths,prev,window_tokens,update):
        (loss,ntokens),grad=loss_grad(model,batch,lengths)
        if prev is not None:grad=tree_map(lambda a,b:a+b,grad,prev)
        if update:
            if normalization=='sample':optimizer.update(model,tree_map(lambda x:x/8,grad))
            else:optimizer.update(model,tree_map(lambda x:x/window_tokens,grad))
            grad=None
        return loss,ntokens,grad

    def step(batch,lengths,prev,window_tokens,update):
        if not isinstance(window_tokens,int) or window_tokens<=0:raise ValueError('Invalid supervised-token denominator')
        return compiled(batch,lengths,prev,mx.array(window_tokens),update)
    return step,state


def train_variant(snapshot,sft,out,*,arm,limit,fingerprint,progress_root,resources):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    configure_mlx_memory(mx)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=recipe(arm,limit,fingerprint,sha(sft));cp=out/'config.json'
    if cp.exists():validate_config(json.loads(cp.read_text()),config)
    cp.write_text(json.dumps(config,indent=2))
    states=sorted([p for p in out.glob('step-*') if (p/'state.json').exists()],key=lambda p:int(p.name[5:]))
    for checkpoint in states:verify_checkpoint(checkpoint,config,int(checkpoint.name[5:]))
    if (out/'summary.json').exists():
        result=json.loads((out/'summary.json').read_text())
        if not states or int(states[-1].name[5:])!=limit or result['microsteps']!=limit:raise ValueError('Incomplete saved training')
        return result
    update_state(progress_root,stage='certify' if arm=='M0' else 'training_'+arm,status='running',step=0,total_steps=limit,
                 detail=f'{arm} · {config["normalization"]} · bias={config["bias_correction"]}',message='按冻结配方加载原模型并独立训练。')
    resources.check();model,tokenizer=load(str(snapshot));resources.check();model.freeze()
    mx.random.seed(SEED);linear_to_lora_layers(model,8,LORA);optimizer=make_optimizer(arm)
    rows=read_rows(sft)
    if len(rows)!=1200:raise ValueError('Expected unchanged 1200-row D0')
    encoded=[encode_row(r,tokenizer,1217) for r in rows];schedule=exposure_schedule(1200,1,SEED)
    exposure=[rows[i]['id'] for i in schedule];(out/'exposure.json').write_text(json.dumps(exposure))
    events_path=out/'events.jsonl';events=read_rows(events_path) if events_path.exists() else []
    initial=restore_training_state(states[-1],model,optimizer,fingerprint) if states else 0
    if initial>limit:raise ValueError('Checkpoint exceeds budget')
    events=[e for e in events if e['microstep']<=initial]
    events_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    step,state=build_step(model,optimizer,config['normalization']);model.train();grad=None
    means=[];sums=[];durations=[];supervised=0;started=time.perf_counter();mx.reset_peak_memory()
    trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    update_state(progress_root,step=initial,loss_history=[])
    for index in range(initial,limit):
        resources.check();tokens,offset=encoded[schedule[index]]
        if index%8==0:
            window_tokens=sum(len(encoded[i][0])-encoded[i][1] for i in schedule[index:index+8])
            if len(schedule[index:index+8])!=8 or window_tokens<=0:raise ValueError('Incomplete/empty accumulation window')
        padded=min(1217,1+32*math.ceil(len(tokens)/32))
        if padded<len(tokens):raise ValueError('Silent truncation prohibited')
        arr=np.zeros((1,padded),dtype=np.int32);arr[0,:len(tokens)]=tokens;tick=time.perf_counter()
        loss,n,grad=step(mx.array(arr),mx.array([[offset,len(tokens)]]),grad,window_tokens,(index+1)%8==0)
        mx.eval(state,loss,n,grad);duration=time.perf_counter()-tick;value=float(loss.item());count=int(n.item())
        if count!=len(tokens)-offset:raise ValueError('Target token denominator mismatch')
        if not math.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        means.append(value/count if config['normalization']=='token' else value)
        sums.append(value if config['normalization']=='token' else value*count);durations.append(duration);supervised+=count
        micro=index+1
        if micro%8==0:
            if supervised!=window_tokens or int(optimizer.step.item())!=micro//8:raise ValueError('Accumulation/update mismatch')
            sample=resources.samples[-1]
            event=dict(microstep=micro,optimizer_updates=micro//8,loss=sum(means)/8,
                       objective_loss=sum(sums)/supervised if config['normalization']=='token' else sum(means)/8,
                       seconds=sum(durations),wall_seconds=time.perf_counter()-started,supervised_tokens=supervised,
                       mlx_active_gb=mx.get_active_memory()/1e9,mlx_peak_gb=mx.get_peak_memory()/1e9,mlx_cache_gb=mx.get_cache_memory()/1e9,
                       rss_gib=sample['rss_bytes']/1024**3,swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3)
            with events_path.open('a') as handle:handle.write(json.dumps(event)+'\n')
            events.append(event);means=[];sums=[];durations=[];supervised=0
            update_state(progress_root,step=micro,metrics=dict(arm=arm,normalization=config['normalization'],bias_correction=config['bias_correction'],**event),
                         loss_history=[dict(step=e['microstep'],loss=e['objective_loss']) for e in events])
            print(json.dumps(dict(arm=arm,**event)),flush=True);mx.clear_cache()
        if micro in config['checkpoints']:
            saved=out/f'step-{micro:04d}';save_training_state(saved,model,optimizer,micro,fingerprint)
            metadata=json.loads((saved/'state.json').read_text());metadata['mechanism']=checkpoint_identity(config)
            metadata['adapter_config_sha256']=sha(saved/'adapter_config.json')
            temp=saved/'state.tmp';temp.write_text(json.dumps(metadata,indent=2));temp.replace(saved/'state.json')
    resources.check()
    result=dict(status='complete',arm=arm,microsteps=limit,optimizer_updates=limit//8,trainable_parameters=trainable,
                normalization=config['normalization'],bias_correction=config['bias_correction'],learning_rate=1e-4,
                schedule_sha256=hashlib.sha256(json.dumps(exposure).encode()).hexdigest(),
                consumed_ids_sha256=hashlib.sha256(json.dumps(exposure[:limit]).encode()).hexdigest(),source_fingerprint=fingerprint,
                active_training_seconds=sum(e['seconds'] for e in events),events=len(events),
                total_input_tokens=sum(encoded[i][1] for i in schedule[:limit]),total_answer_tokens=sum(len(encoded[i][0])-encoded[i][1] for i in schedule[:limit]),
                mlx_peak_gb=max(e['mlx_peak_gb'] for e in events),resumed_from_microstep=initial)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    del model,optimizer,grad,state,step;mx.clear_cache()
    return result
