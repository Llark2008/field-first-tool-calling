"""Fixed free-generation evaluation, with private labels used only by scoring."""
import json
import time
from collections import defaultdict
from pathlib import Path

from .controls import build_control_messages, source_guard
from .generation_scoring import score_generation
from .live_progress import update_state


def failure_type(row):
    s=row['score']; action=s.get('action',{}).get('action')
    if not s.get('schema_valid') or s.get('truncated'): return 'invalid'
    if s['success']: return 'correct'
    if row['gold']['action']=='clarify':
        if action=='call': return 'premature_call'
        if action=='clarify': return 'wrong_clarification_fields'
        return 'false_unsupported'
    if row['gold']['action']=='call':
        if action=='clarify': return 'false_clarify'
        if action=='unsupported': return 'false_unsupported'
        return 'wrong_call'
    return 'unsupported_missed'


def summarize(rows):
    result={}
    for condition in sorted({r['condition'] for r in rows}):
        group=[r for r in rows if r['condition']==condition]
        entry=dict(n=len(group),success=0,invalid=0,premature_call=0,false_clarify=0,false_unsupported=0,
            wrong_clarification_fields=0,blocked_calls=0,blocked_correct_calls=0,allowed_premature_calls=0,
            system_success=0,correct_clarifications=0)
        for row in group:
            s=row['score']; kind=s.get('action',{}).get('action'); valid=s.get('schema_valid') and not s.get('truncated')
            premature=row['gold']['action']=='clarify' and kind=='call'
            blocked=valid and kind=='call' and row.get('guard',{}).get('blocked',False)
            entry['success']+=bool(s['success']); entry['invalid']+=not valid
            entry['premature_call']+=premature
            entry['false_clarify']+=row['gold']['action']=='call' and kind=='clarify'
            entry['false_unsupported']+=row['gold']['action']!='unsupported' and kind=='unsupported'
            entry['wrong_clarification_fields']+=valid and row['gold']['action']=='clarify' and kind=='clarify' and not s['success']
            entry['blocked_calls']+=bool(blocked); entry['blocked_correct_calls']+=bool(blocked and s['success'])
            entry['allowed_premature_calls']+=bool(valid and premature and not blocked)
            entry['system_success']+=bool(s['success'] and not blocked)
            entry['correct_clarifications']+=bool(s['success'] and kind=='clarify')
        entry['accuracy']=entry['success']/entry['n']
        result[condition]=entry
    return result


def evaluate_model(model, tokenizer, cases, output, metadata, progress_root, stage, resources):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    meta_path=output.with_suffix('.meta.json')
    if meta_path.exists():
        if json.loads(meta_path.read_text())!=metadata: raise ValueError('Evaluation metadata mismatch')
    else: meta_path.write_text(json.dumps(metadata,indent=2))
    rows=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if [r['case_id'] for r in rows]!=[c['id'] for c in cases[:len(rows)]]: raise ValueError('Evaluation ID alignment mismatch')
    model.eval()
    update_state(progress_root,stage=stage,status='running',step=len(rows),total_steps=len(cases),
        detail=metadata['name'],message='逐条自由生成并严格评测；不重试、不修复模型答案。')
    for i,case in enumerate(cases[len(rows):],start=len(rows)):
        resources.check()
        messages=build_control_messages(case,metadata['prompt'])
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        start=time.perf_counter(); pieces=[]; last=None
        for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=192,sampler=make_sampler(temp=0)):
            pieces.append(response.text); last=response
        if last is None: raise RuntimeError('Model returned no generation')
        text=''.join(pieces); score=score_generation(case,text,last.finish_reason)
        valid_call=score['schema_valid'] and not score['truncated'] and score.get('action',{}).get('action')=='call'
        guard=source_guard(case['request'],case['tools'],score['action']) if valid_call else dict(blocked=False,ungrounded_fields=[])
        row=dict(case_id=case['id'],group_id=case['group_id'],template_group=case.get('template_group',case['group_id']),
            condition=case['condition'],target_tool=case.get('target_tool'),missing_fields=case.get('missing_fields',[]),
            gold=case['gold'],output=text,finish_reason=last.finish_reason,score=score,guard=guard,
            seconds=time.perf_counter()-start)
        row['failure_type']=failure_type(row)
        with output.open('a') as handle:
            handle.write(json.dumps(row,separators=(',',':'))+'\n'); handle.flush()
        rows.append(row)
        if (i+1)%5==0 or i+1==len(cases):
            sample=resources.samples[-1]
            update_state(progress_root,step=i+1,metrics=dict(configuration=metadata['name'],
                generated=i+1,total=len(cases),mean_seconds_per_case=sum(r['seconds'] for r in rows)/len(rows),
                mlx_active_gb=mx.get_active_memory()/1e9,mlx_peak_gb=mx.get_peak_memory()/1e9,
                rss_gib=sample['rss_bytes']/1024**3,swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3))
        if (i+1)%40==0: mx.clear_cache()
    metrics=summarize(rows)
    output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics
