"""One fixed two-call field-first probe, following the single-call format failure."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time

from experiments.fields_first_prompt.run import ROOT as PARENT,verify as verify_parent,SOURCE
from experiments.fields_first_prompt.core import score,field_target
from experiments.action_weighting.report import verify_marker
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.analyze_clarification_ablation import metrics_for
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import Heartbeat,update_state
from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
from toolcall_lab.run_clarification_ablation import write_json,read_json as read,verify_file_map
from toolcall_lab.train_formal import configure_mlx_memory
from scripts.run_train_replay import MONITOR_HTML
from .core import messages

ROOT=Path.home()/'.local/share/local-tool-calling/runs/fields-first-staged-20261115'
PROTOCOL=SOURCE/'docs/superpowers/specs/2026-11-15-fields-first-staged-spike.md'


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def sources():return {str(p.relative_to(SOURCE)):sha(p) for p in [*sorted((SOURCE/'experiments/fields_first_staged').rglob('*.py')),PROTOCOL]}


def freeze():
    p=read(PARENT/'frozen.json');verify_parent(p)
    verify_marker(read(PARENT/'completed.json'),p['fingerprint'],sha(PARENT/'report-data.json'))
    verify_file_map(PARENT,read(PARENT/'report-data.json')['artifacts'])
    if read(PARENT/'report-data.json')['metrics']['V']['format_failures']<96:raise ValueError('Format-failure rationale not met')
    cases=read_rows(Path(p['dataset'])/'dev.jsonl')
    identity=dict(parent_fingerprint=p['fingerprint'],parent_report_sha256=sha(PARENT/'report-data.json'),sources=sources(),
        snapshot=p['snapshot'],dataset=p['dataset'],cases_sha256=digest(cases),
        prompt_templates_sha256=digest({s:[messages(c,s,'MODEL_FIELD_OUTPUT' if s==2 else None) for c in cases] for s in [1,2]}),
        generation=dict(temperature=0,thinking=False,max_tokens_per_stage=384,stages=2),
        rationale='Exploratory two-call protocol after single-call V usually stopped at its field object; no relabeling of V.')
    identity['fingerprint']=digest(identity)
    if (ROOT/'frozen.json').exists() and read(ROOT/'frozen.json')!=identity:raise ValueError('Frozen staged protocol changed')
    write_json(ROOT/'frozen.json',identity)
    for relative in identity['sources']:
        path=ROOT/'source-snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((SOURCE/relative).read_bytes())
    return identity,cases


def verify(f,cases):
    check=dict(f);expected=check.pop('fingerprint')
    if digest(check)!=expected or sources()!=f['sources'] or digest(cases)!=f['cases_sha256']:raise ValueError('Staged identity mismatch')
    if sha(PARENT/'report-data.json')!=f['parent_report_sha256']:raise ValueError('Parent result changed')
    verify_parent(read(PARENT/'frozen.json'))


def load_stage(stage,cases,f):
    path=ROOT/f'stage-{stage}.jsonl';meta=dict(fingerprint=f['fingerprint'],stage=stage,generation=f['generation'])
    mp=path.with_suffix('.meta.json')
    if path.exists() and not mp.exists():raise ValueError('Stage output lacks metadata')
    if mp.exists() and read(mp)!=meta:raise ValueError('Stage metadata changed')
    write_json(mp,meta);rows=read_rows(path) if path.exists() else []
    if len(rows)>len(cases) or [r['case_id'] for r in rows]!=[c['id'] for c in cases[:len(rows)]]:raise ValueError('Stage order changed')
    return rows


def generate(model,tokenizer,case,stage,raw,resources):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    resources.check();msgs=messages(case,stage,raw)
    prompt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if len(tokenizer.encode(prompt))>2560:raise ValueError('Staged prompt exceeds fixed budget')
    tick=time.perf_counter();parts=[];last=None
    for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=384,sampler=make_sampler(temp=0)):
        parts.append(response.text);last=response
    if last is None:raise ValueError('Empty generation')
    row=dict(case_id=case['id'],stage=stage,prompt_sha256=digest(msgs),output=''.join(parts),finish_reason=last.finish_reason,
        prompt_tokens=int(last.prompt_tokens),generation_tokens=int(last.generation_tokens),seconds=time.perf_counter()-tick)
    with (ROOT/f'stage-{stage}.jsonl').open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    resources.check();return row


def score_rows(cases,first,second):
    if len(first)!=len(second) or len(first)>len(cases):raise ValueError('Stage counts differ')
    results=[]
    for c,a,b in zip(cases,first,second):
        if a['case_id']!=c['id'] or b['case_id']!=c['id'] or a['prompt_sha256']!=digest(messages(c,1)) or b['prompt_sha256']!=digest(messages(c,2,a['output'])):raise ValueError('Actual prompts differ from frozen staged protocol')
        final=score(c,b['output'],b['finish_reason'],'I')
        combined=score(c,a['output']+'\n'+b['output'],'length' if 'length' in [a['finish_reason'],b['finish_reason']] else b['finish_reason'],'V')
        results.append(dict(case_id=c['id'],group_id=c['group_id'],template_group=c['template_group'],condition=c['condition'],
            target_tool=c['target_tool'],missing_fields=c['missing_fields'],fields_output=a['output'],output=b['output'],
            finish_reason=b['finish_reason'],seconds=a['seconds']+b['seconds'],prompt_tokens=a['prompt_tokens']+b['prompt_tokens'],
            generation_tokens=a['generation_tokens']+b['generation_tokens'],**final,
            stage_fields=combined['field_diagnostics'],both_stages_correct=combined['score']['success'] and combined['field_diagnostics']['format_valid']))
    return results


def finish(f,cases,first,second):
    verify(f,cases)
    if len(first)!=120 or len(second)!=120:raise ValueError('Incomplete budget')
    resources=read(ROOT/'resource-summary.json')
    if resources['error'] or resources['n_samples']<2:raise ValueError('Clean resource evidence required')
    rows=score_rows(cases,first,second);metrics=metrics_for(cases,rows);old=read(PARENT/'report-data.json')
    fields={k:sum(r['stage_fields'][k] for r in rows) for k in ['format_valid','tool_correct','missing_set_correct','fields_exact','self_consistent']}
    fields['both_stages_correct']=sum(r['both_stages_correct'] for r in rows)
    gates=dict(C=metrics['C']['count']>=56,F=metrics['F']['count']<=3,boundary=metrics['boundary']['count']>=27,U=metrics['U']['count']>=11)
    gain=metrics['Q']['count']-old['metrics']['I']['Q']['count']
    verdict=dict(gates=gates,Q_gain_vs_I=gain,promising=all(gates.values()) and gain>=4)
    examples=[dict(id=c['id'],request=c['request'],gold=c['gold'],fields=r['fields_output'],action=r['output'],correct=r['score']['success'],field_diagnostics=r['stage_fields']) for c,r in zip(cases,rows)]
    data=dict(status='complete',fingerprint=f['fingerprint'],metrics=metrics,fields=fields,verdict=verdict,reference=old['metrics'],resources=resources,
        cost=dict(calls=240,seconds=sum(r['seconds'] for r in rows),mean_output_tokens=sum(r['generation_tokens'] for r in rows)/120,
            mean_prompt_tokens=sum(r['prompt_tokens'] for r in rows)/120),examples=examples,
        artifacts={p.name:sha(p) for p in ROOT.glob('stage-*.jsonl')},
        limitation='Exploratory protocol change after format failure: two calls and different stage instructions; not a preplanned isolated causal comparison or a fine-tuning result.')
    write_json(ROOT/'report-data.json',data);write_json(SOURCE/'reports/fields-first-staged-results.json',data)
    table=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{m[k]["count"]}/{m[k]["n"]}</td>' for k in ['C','Q','F','P','U','J'])+'</tr>' for name,m in [('I 内部字段检查',old['metrics']['I']),('V 单次生成两对象',old['metrics']['V']),('T2 两次调用',metrics)])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段先行两阶段结果</title><style>body{{font:17px system-ui;max-width:1120px;margin:40px auto;padding:20px;background:#101820;color:#e5edf3}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #456;text-align:left}}section{{padding:24px;background:#192630;border-radius:12px;margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}a{{color:#9de7d0}}</style><h1>字段先行：两次模型调用</h1><p>原模型先生成字段，再读取自己生成的字段并输出最终动作。全部120题进入第二阶段，无标准字段、重试或程序代填。</p><section><h2>{'达到筛查门槛' if verdict['promising'] else '未达到筛查门槛'}</h2><table><tr><th>组别</th><th>完整C ↑</th><th>缺参Q ↑</th><th>误追问F ↓</th><th>猜测P ↓</th><th>不支持U ↑</th><th>配对J ↑</th></tr>{table}</table><p>V得分为0主要因为未输出最终动作，不能据此判断字段推理能力。T2为发现格式问题后的独立探索条件，有两次调用和更多上下文成本；不是微调结果。</p></section><section><h2>阶段正确率、成本与全部案例</h2><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></section><p><a href="/">进度</a> · <a href="http://127.0.0.1:49407/results">动作加权结果</a></p></html>'''
    (ROOT/'results.html').write_text(page);(SOURCE/'reports/fields-first-staged-results.html').write_text(page)
    write_json(ROOT/'completed.json',dict(status='complete',fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json')))
    update_state(ROOT,stage='complete',status='complete',step=120,total_steps=120,finished_at=time.time(),results_ready=True,
        message='固定两阶段试验完成，240次调用均已核验。',checkpoints=[dict(name='T2',complete=f'{metrics["C"]["count"]}/60',clarify=f'{metrics["Q"]["count"]}/48',unsupported=f'{metrics["U"]["count"]}/12')])
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if args.serve:
        from toolcall_lab import live_progress
        live_progress.HTML=MONITOR_HTML.replace('已见训练样本回放','字段先行：固定两次调用').replace('训练样本回放','字段两阶段').replace('固定120题 · 原模型与M0/M1/M2最新75更新检查点 · 自由生成与teacher forcing · 只评测，不训练','120题 × 先字段、后动作 · 相同原模型 · 无微调 · 模型自己的字段进入第二阶段')
        sys.argv=[sys.argv[0],'--root',str(ROOT),'--port','49411'];live_progress.main();return
    import fcntl
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    beat=Heartbeat(ROOT);beat.start();resources=None
    try:
        update_state(ROOT,stage='audit',status='running',step=0,total_steps=120,message='核验固定两阶段流程和父实验；全部题目均使用模型自己的字段。')
        f,cases=freeze()
        if (ROOT/'completed.json').exists():
            verify_marker(read(ROOT/'completed.json'),f['fingerprint'],sha(ROOT/'report-data.json'));verify_file_map(ROOT,read(ROOT/'report-data.json')['artifacts'])
            update_state(ROOT,stage='complete',status='complete');print('Already complete');return
        first=load_stage(1,cases,f);second=load_stage(2,cases,f)
        if not len(second)<=len(first)<=len(second)+1:raise ValueError('Invalid stage resume cursor')
        score_rows(cases,first[:len(second)],second)
        if len(first)>len(second) and first[-1]['prompt_sha256']!=digest(messages(cases[len(second)],1)):raise ValueError('Unpaired first-stage prompt changed')
        resources=ResourceMonitor(ROOT/f'resources-{int(time.time())}.jsonl');resources.start();baseline=ROOT/'resource-baseline.json'
        if baseline.exists():
            original=read(baseline);current=resources.samples[0];resources.samples.insert(0,original)
            error=resource_violation(current,original['swap_used_bytes'])
            if error:resources.error=error
        else:write_json(baseline,resources.samples[0])
        resources.check()
        import mlx.core as mx
        from mlx_lm import load
        configure_mlx_memory(mx);model,tokenizer=load(f['snapshot']);model.eval();resources.check()
        update_state(ROOT,stage='generation',detail='T2：先字段、后动作',step=len(second),total_steps=120,status='running')
        try:
            for i in range(len(second),120):
                if i==len(first):first.append(generate(model,tokenizer,cases[i],1,None,resources))
                second.append(generate(model,tokenizer,cases[i],2,first[i]['output'],resources))
                if (i+1)%5==0:
                    sample=resources.samples[-1];update_state(ROOT,step=i+1,metrics=dict(completed_tasks=i+1,calls=len(first)+len(second),
                        mean_seconds_per_task=sum(a['seconds']+b['seconds'] for a,b in zip(first,second))/(i+1),mlx_peak_gb=mx.get_peak_memory()/1e9,
                        swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3))
                if (i+1)%20==0:mx.clear_cache()
        finally:del model,tokenizer;mx.clear_cache()
        resources.stop();resources.check();write_json(ROOT/'resource-summary.json',resources.summary());resources=None
        data=finish(f,cases,first,second);print(json.dumps(data['verdict']),flush=True)
    except BaseException as exc:update_state(ROOT,status='failed',message=repr(exc));raise
    finally:
        if resources:resources.stop();write_json(ROOT/'resource-summary.json',resources.summary())
        beat.stop();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
