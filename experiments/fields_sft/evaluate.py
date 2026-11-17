"""Unassisted free generation and separately labeled teacher-forced diagnostics."""
import json
import math
import time
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from toolcall_lab.run_clarification_ablation import write_json,read_json as read
from toolcall_lab.train_formal import configure_mlx_memory
from .data import digest,messages,encoded_target
from .scoring import score
from .train import recipe,verify_checkpoint

def validate_rows(cases,rows,arm,partial=False):
    if len(rows)>len(cases) or (not partial and len(rows)!=len(cases)):raise ValueError('Wrong output count')
    for c,r in zip(cases,rows):
        if r['case_id']!=c['id'] or any(r.get(k)!=c[k] for k in ['condition','gold','template_group','target_tool','missing_fields']):raise ValueError('Output case identity changed')
        rescored=score(c,r['output'],r['finish_reason'],arm)
        if any(r.get(k)!=v for k,v in rescored.items()):raise ValueError('Rescoring changed')
    return rows

def metadata(root,f,cases,arm,step,split):
    adapter=root/f'train-{arm}/step-{step:04d}'
    config=recipe(arm,f['fingerprint'],sha(root/f'sft-{arm}.jsonl'),f['cap']);verify_checkpoint(adapter,config,step)
    return dict(fingerprint=f['fingerprint'],arm=arm,microstep=step,split=split,cases_sha256=digest(cases),
        prompts_sha256=digest([messages(c,arm) for c in cases]),adapter_sha256=sha(adapter/'adapters.safetensors'),generation=f['generation'])

def checked(path,cases,meta,partial=False):
    if read(path.with_suffix('.meta.json'))!=meta:raise ValueError('Output metadata changed')
    return validate_rows(cases,read_rows(path),meta['arm'],partial)

def prepare(path,meta):
    mp=path.with_suffix('.meta.json')
    if path.exists() and not mp.exists():raise ValueError('Outputs lack metadata')
    if mp.exists() and read(mp)!=meta:raise ValueError('Resume identity changed')
    write_json(mp,meta)

def generate(model,tokenizer,root,f,cases,arm,step,split,resources):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    path=root/f'{split}-{arm}-{step}.jsonl';meta=metadata(root,f,cases,arm,step,split);prepare(path,meta)
    rows=checked(path,cases,meta,True) if path.exists() else []
    update_state(root,stage='development' if split=='dev' else 'replay',status='running',detail=f'{arm}@{step//8} · {split}',step=len(rows),total_steps=len(cases),message='原始请求自由生成；不提供标准字段、不修复、不重试。')
    model.eval()
    for c in cases[len(rows):]:
        resources.check();prompt=tokenizer.apply_chat_template(messages(c,arm),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tick=time.perf_counter();pieces=[];last=None
        for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=384,sampler=make_sampler(temp=0)):
            pieces.append(response.text);last=response
        if last is None:raise ValueError('Empty generation')
        raw=''.join(pieces)
        row=dict(case_id=c['id'],group_id=c['group_id'],condition=c['condition'],template_group=c['template_group'],target_tool=c['target_tool'],missing_fields=c['missing_fields'],
            output=raw,finish_reason=last.finish_reason,seconds=time.perf_counter()-tick,prompt_tokens=int(last.prompt_tokens),generation_tokens=int(last.generation_tokens),**score(c,raw,last.finish_reason,arm))
        with path.open('a') as h:h.write(json.dumps(row,ensure_ascii=False)+'\n')
        rows.append(row)
        if len(rows)%5==0:update_state(root,step=len(rows),metrics=dict(arm=arm,mean_seconds=sum(r['seconds'] for r in rows)/len(rows),mean_output_tokens=sum(r['generation_tokens'] for r in rows)/len(rows)))
        if len(rows)%40==0:mx.clear_cache()
    resources.check();return checked(path,cases,meta)

def validate_teacher(cases,rows,arm,tokenizer,cap,partial=False):
    if len(rows)>len(cases) or (not partial and len(rows)!=len(cases)):raise ValueError('Wrong teacher count')
    for c,r in zip(cases,rows):
        _,tokens,offset,groups=encoded_target(c,arm,tokenizer,cap)
        if (r['case_id'],r['tokens'],r['offset'],r['groups'])!=(c['id'],tokens,offset,groups):raise ValueError('Teacher target identity changed')
        if len(r['losses'])!=len(tokens)-offset or len(r['correct'])!=len(r['losses']):raise ValueError('Teacher token count mismatch')
        if any(not math.isfinite(x) or x<0 for x in r['losses']) or any(type(v)!=bool for v in r['correct']):raise ValueError('Invalid teacher values')
    return rows

def teacher(model,tokenizer,root,f,cases,arm,resources):
    import mlx.core as mx
    import mlx.nn as nn
    path=root/f'teacher-{arm}.jsonl';meta=dict(metadata(root,f,cases,arm,600,'seen'),mode='teacher_forced',targets_sha256=sha(root/f'sft-{arm}.jsonl'))
    prepare(path,meta);rows=validate_teacher(cases,read_rows(path),arm,tokenizer,f['cap'],True) if path.exists() else []
    update_state(root,stage='replay',status='running',detail=arm+' · teacher forcing（标准前文）',step=len(rows),total_steps=len(cases),message='分开统计已知值、null 和最终动作；不将标准字段条件下的预测当作自由生成能力。')
    for c in cases[len(rows):]:
        resources.check();_,tokens,offset,groups=encoded_target(c,arm,tokenizer,f['cap']);tick=time.perf_counter()
        batch=mx.array([tokens]);logits=model(batch[:,:-1]);ce=nn.losses.cross_entropy(logits,batch[:,1:]).astype(mx.float32)
        correct=mx.argmax(logits,axis=-1)==batch[:,1:];mx.eval(ce,correct)
        r=dict(case_id=c['id'],tokens=tokens,offset=offset,groups=groups,losses=ce[0,offset-1:].tolist(),correct=correct[0,offset-1:].tolist(),seconds=time.perf_counter()-tick)
        with path.open('a') as h:h.write(json.dumps(r)+'\n')
        rows.append(r);del batch,logits,ce,correct
        if len(rows)%5==0:update_state(root,step=len(rows))
        mx.clear_cache()
    resources.check();return validate_teacher(cases,rows,arm,tokenizer,f['cap'])

def evaluate(root,f,cases,arm,step,split,resources):
    import mlx.core as mx
    from mlx_lm import load
    meta=metadata(root,f,cases,arm,step,split);path=root/f'{split}-{arm}-{step}.jsonl'
    if path.exists() and len(checked(path,cases,meta,True))==len(cases) and split=='dev':return
    configure_mlx_memory(mx);resources.check()
    model,tokenizer=load(f['snapshot'],adapter_path=str(root/f'train-{arm}/step-{step:04d}'))
    try:
        resources.check();generate(model,tokenizer,root,f,cases,arm,step,split,resources)
        if split=='seen':teacher(model,tokenizer,root,f,cases,arm,resources)
        resources.check()
    finally:del model,tokenizer;mx.clear_cache()

def teacher_summary(rows):
    out={}
    for name in ['all','known','null','action']:
        losses=[];correct=[]
        for r in rows:
            indices=list(range(len(r['losses']))) if name=='all' else [i-r['offset'] for i in r['groups'][name]]
            losses.extend(r['losses'][i] for i in indices);correct.extend(r['correct'][i] for i in indices)
        out[name]=dict(tokens=len(losses),loss=sum(losses)/len(losses) if losses else None,correct=sum(correct),accuracy=sum(correct)/len(correct) if correct else None)
    return out
