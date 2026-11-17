"""Matched free generation and auxiliary fixed-set teacher loss at every epoch."""
import json
import time
from pathlib import Path
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from toolcall_lab.train_formal import configure_mlx_memory
from experiments.fields_sft.data import messages,digest,encoded_target,compact
from experiments.fields_sft.evaluate import validate_teacher,teacher_summary
from experiments.field_action_diagnostics.run import read,immutable
from .train import verify_epoch

def consume_generation(stream,resources):
    parts=[];last=None
    for response in stream:
        resources.check()
        parts.append(response.text);last=response
    return parts,last

def validate_raw(cases,rows,prompts,partial=False):
    if len(rows)>len(cases) or (not partial and len(rows)!=len(cases)):raise ValueError('Wrong evaluation count')
    for c,r in zip(cases,rows):
        if r['case_id']!=c['id'] or r['prompt_sha256']!=prompts[c['id']]:raise ValueError('Raw case/prompt identity changed')
        if not isinstance(r['output'],str) or r['finish_reason'] not in ['stop','length']:raise ValueError('Invalid raw output')
    return rows

def meta(root,f,arm,epoch,split,kind):
    path=verify_epoch(root,f,arm,epoch)
    return dict(fingerprint=f['fingerprint'],arm=arm,epoch=epoch,split=split,kind=kind,adapter_sha256=sha(path/'adapters.safetensors'),
        cases_sha256=f['eval_cases_sha256'][split],prompts_sha256=digest(f['prompt_hashes'][split]),generation=f['generation'])

def paths(root,arm,epoch,split,kind):return root/f'{kind}-{arm}-{epoch}-{split}.jsonl'

def free(model,tok,root,f,arm,epoch,split,cases,resources):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    path=paths(root,arm,epoch,split,'free');immutable(path.with_suffix('.meta.json'),meta(root,f,arm,epoch,split,'free'))
    rows=validate_raw(cases,read_rows(path),f['prompt_hashes'][split],True) if path.exists() else []
    update_state(root,stage='evaluation',status='running',detail=f'{arm} 第{epoch}轮 · {split} 自由生成',step=len(rows),total_steps=120,message='原请求生成字段与动作；不提供标准字段。转换器结果另行评分。')
    for c in cases[len(rows):]:
        resources.check();msgs=messages(c,'F')
        if digest(msgs)!=f['prompt_hashes'][split][c['id']]:raise ValueError('Actual prompt changed')
        prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tick=time.perf_counter()
        parts,last=consume_generation(stream_generate(model,tok,prompt=prompt,max_tokens=384,sampler=make_sampler(temp=0)),resources)
        if last is None:raise ValueError('Empty generation')
        row=dict(case_id=c['id'],prompt_sha256=digest(msgs),output=''.join(parts),finish_reason=last.finish_reason,seconds=time.perf_counter()-tick,
                 prompt_tokens=int(last.prompt_tokens),generation_tokens=int(last.generation_tokens))
        with path.open('a') as h:h.write(compact(row)+'\n');h.flush()
        rows.append(row)
        if len(rows)%5==0:update_state(root,step=len(rows),metrics=dict(arm=arm,epoch=epoch,split=split,mean_seconds=sum(r['seconds'] for r in rows)/len(rows)))
        if len(rows)%40==0:mx.clear_cache()
    resources.check();return validate_raw(cases,rows,f['prompt_hashes'][split])

def teacher(model,tok,root,f,arm,epoch,split,cases,resources):
    import mlx.core as mx
    import mlx.nn as nn
    path=paths(root,arm,epoch,split,'teacher');immutable(path.with_suffix('.meta.json'),meta(root,f,arm,epoch,split,'teacher'))
    cap=f['teacher_cap'];rows=validate_teacher(cases,read_rows(path),'F',tok,cap,True) if path.exists() else []
    update_state(root,stage='teacher',status='running',detail=f'{arm} 第{epoch}轮 · {split} 固定题集loss',step=len(rows),total_steps=120,message='标准答案前文下的逐token loss，仅作辅助；自由生成成绩保持独立。')
    for c in cases[len(rows):]:
        resources.check();_,ids,offset,groups=encoded_target(c,'F',tok,cap);tick=time.perf_counter()
        batch=mx.array([ids]);logits=model(batch[:,:-1]);ce=nn.losses.cross_entropy(logits,batch[:,1:]).astype(mx.float32)
        correct=mx.argmax(logits,axis=-1)==batch[:,1:];mx.eval(ce,correct)
        r=dict(case_id=c['id'],tokens=ids,offset=offset,groups=groups,losses=ce[0,offset-1:].tolist(),correct=correct[0,offset-1:].tolist(),seconds=time.perf_counter()-tick)
        with path.open('a') as h:h.write(compact(r)+'\n');h.flush()
        rows.append(r);del batch,logits,ce,correct;mx.clear_cache()
        if len(rows)%5==0:update_state(root,step=len(rows))
    resources.check();return validate_teacher(cases,rows,'F',tok,cap)

def evaluate_epoch(root,f,arm,epoch,resources,report_callback):
    import mlx.core as mx
    from mlx_lm import load
    complete=True
    for split in ['dev','seen']:
        for kind in ['free','teacher']:
            p=paths(root,arm,epoch,split,kind)
            complete &= p.exists() and len(read_rows(p))==120
    if complete:return
    resources.check();configure_mlx_memory(mx);adapter=verify_epoch(root,f,arm,epoch)
    model,tok=load(f['snapshot'],adapter_path=str(adapter));model.eval();resources.check()
    try:
        for split in ['dev','seen']:
            cases=read_rows(Path(f['parent'])/f'{split}-cases.jsonl')
            free(model,tok,root,f,arm,epoch,split,cases,resources);report_callback()
            teacher(model,tok,root,f,arm,epoch,split,cases,resources);report_callback()
    finally:del model,tok;mx.clear_cache()
