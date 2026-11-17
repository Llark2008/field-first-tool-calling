"""Small no-training experiment: internal versus explicitly emitted fields."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

from toolcall_lab.ablation_data import read_rows
from toolcall_lab.analyze_clarification_ablation import metrics_for
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import Heartbeat,update_state
from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
from toolcall_lab.run_clarification_ablation import write_json,read_json as read,verify_file_map
from toolcall_lab.train_formal import configure_mlx_memory
from scripts.run_train_replay import MONITOR_HTML
from experiments.action_weighting.run import verify as verify_weighting,ROOT as WEIGHTING,SOURCE
from experiments.action_weighting.report import verify_marker
from .core import messages,audit_case,score

ROOT=Path.home()/'.local/share/local-tool-calling/runs/fields-first-prompt-20261115'
PROTOCOL=SOURCE/'docs/superpowers/specs/2026-11-15-fields-first-prompt-spike.md'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_map():return {str(p.relative_to(SOURCE)):sha(p) for p in [*sorted((SOURCE/'experiments/fields_first_prompt').rglob('*.py')),PROTOCOL]}


def freeze():
    from mlx_lm.utils import load_tokenizer
    weighted=read(WEIGHTING/'frozen.json');verify_weighting(WEIGHTING,weighted)
    verify_marker(read(WEIGHTING/'completed.json'),weighted['fingerprint'],sha(WEIGHTING/'report-data.json'))
    if read(WEIGHTING/'report-data.json')['verdict']['joint_pass']:raise ValueError('Conditional next experiment not triggered')
    dataset=Path(weighted['dataset']);cases=read_rows(dataset/'dev.jsonl');training=read_rows(dataset/'D0.jsonl')
    targets=[audit_case(c) for c in cases];[audit_case(c) for c in training]
    tokenizer=load_tokenizer(Path(weighted['snapshot']))
    lengths={a:[len(tokenizer.apply_chat_template(messages(c,a),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)) for c in cases] for a in ['I','V']}
    if any(max(v)>2048 for v in lengths.values()):raise ValueError('Prompt exceeds audited budget')
    identity=dict(sources=source_map(),parent_fingerprint=weighted['fingerprint'],parent_report_sha256=sha(WEIGHTING/'report-data.json'),
        dataset=str(dataset),snapshot=weighted['snapshot'],dataset_files={name:sha(dataset/name) for name in ['D0.jsonl','dev.jsonl']},
        cases_sha256=digest(cases),targets_sha256=digest(targets),prompt_sha256={a:digest([messages(c,a) for c in cases]) for a in ['I','V']},
        generation=dict(temperature=0,max_tokens=384,thinking=False,arms=['I','V']),
        audit=dict(development_rows=len(cases),training_annotations=len(training),conditions=dict(Counter(c['condition'] for c in cases)),
            max_prompt_tokens={a:max(v) for a,v in lengths.items()},private_labels_excluded=True))
    identity['fingerprint']=digest(identity)
    path=ROOT/'frozen.json'
    if path.exists() and read(path)!=identity:raise ValueError('Cannot change frozen prompt experiment')
    write_json(path,identity);write_json(ROOT/'field-targets.json',targets)
    write_json(ROOT/'prompt-examples.json',{a:messages(cases[0],a) for a in ['I','V']})
    for relative in identity['sources']:
        destination=ROOT/'source-snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((SOURCE/relative).read_bytes())
    update_state(ROOT,stage='audit',status='running',audit=identity['audit'],message='1320条字段标注与两组提示长度审计通过。')
    return identity,cases


def verify(f):
    check=dict(f);expected=check.pop('fingerprint')
    if digest(check)!=expected or source_map()!=f['sources']:raise ValueError('Frozen spike identity changed')
    verify_file_map(Path(f['dataset']),f['dataset_files'])
    if sha(WEIGHTING/'report-data.json')!=f['parent_report_sha256']:raise ValueError('Parent result changed')
    verify_weighting(WEIGHTING,read(WEIGHTING/'frozen.json'),full=False)


def metadata(f,arm):return dict(fingerprint=f['fingerprint'],arm=arm,prompt_sha256=f['prompt_sha256'][arm],cases_sha256=f['cases_sha256'],generation=f['generation'])


def checked_rows(path,cases,f,arm,partial=False):
    if read(path.with_suffix('.meta.json'))!=metadata(f,arm):raise ValueError('Evaluation identity changed')
    rows=read_rows(path)
    if len(rows)>len(cases) or (not partial and len(rows)!=len(cases)) or [r['case_id'] for r in rows]!=[c['id'] for c in cases[:len(rows)]]:raise ValueError('Output case alignment mismatch')
    for c,r in zip(cases,rows):
        again=score(c,r['output'],r['finish_reason'],arm)
        if any(r[k]!=value for k,value in again.items()):raise ValueError('Rescoring mismatch')
    return rows


def evaluate(model,tokenizer,cases,f,arm,resources):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    path=ROOT/f'free-{arm}.jsonl';mp=path.with_suffix('.meta.json')
    if path.exists() and not mp.exists():raise ValueError('Outputs lack metadata')
    if mp.exists() and read(mp)!=metadata(f,arm):raise ValueError('Resume metadata mismatch')
    write_json(mp,metadata(f,arm));rows=checked_rows(path,cases,f,arm,partial=True) if path.exists() else []
    update_state(ROOT,stage='generation',status='running',detail='I 内部字段检查' if arm=='I' else 'V 显式字段先行',step=len(rows),total_steps=len(cases),message='原模型，无微调；字段与最终动作均由模型自行生成。')
    model.eval()
    for c in cases[len(rows):]:
        resources.check();prompt=tokenizer.apply_chat_template(messages(c,arm),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tick=time.perf_counter();pieces=[];last=None
        for response in stream_generate(model,tokenizer,prompt=prompt,max_tokens=384,sampler=make_sampler(temp=0)):
            pieces.append(response.text);last=response
        if last is None:raise ValueError('No generation')
        raw=''.join(pieces)
        row=dict(case_id=c['id'],group_id=c['group_id'],template_group=c['template_group'],condition=c['condition'],
            target_tool=c['target_tool'],missing_fields=c['missing_fields'],output=raw,finish_reason=last.finish_reason,
            seconds=time.perf_counter()-tick,prompt_tokens=int(last.prompt_tokens),generation_tokens=int(last.generation_tokens),**score(c,raw,last.finish_reason,arm))
        with path.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        rows.append(row)
        if len(rows)%5==0:
            sample=resources.samples[-1]
            update_state(ROOT,step=len(rows),metrics=dict(arm=arm,mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
                mean_output_tokens=sum(r['generation_tokens'] for r in rows)/len(rows),mlx_peak_gb=mx.get_peak_memory()/1e9,
                swap_growth_gib=(sample['swap_used_bytes']-resources.samples[0]['swap_used_bytes'])/1024**3))
        if len(rows)%40==0:mx.clear_cache()
    resources.check();return checked_rows(path,cases,f,arm)


def report(f,cases,complete=False):
    verify(f);groups={};allrows={}
    for arm in ['I','V']:
        path=ROOT/f'free-{arm}.jsonl'
        if not path.exists():continue
        rows=checked_rows(path,cases,f,arm,partial=True)
        if len(rows)!=len(cases):continue
        allrows[arm]=rows;m=metrics_for(cases,rows)
        m['format_failures']=sum(r['format_error'] is not None for r in rows)
        m['seconds']=sum(r['seconds'] for r in rows);m['mean_output_tokens']=sum(r['generation_tokens'] for r in rows)/len(rows)
        if arm=='V':m['fields']={k:sum(r['field_diagnostics'][k] for r in rows) for k in ['format_valid','tool_correct','missing_set_correct','fields_exact','self_consistent']}
        groups[arm]=m
    verdict=None;resources=read(ROOT/'resource-summary.json') if (ROOT/'resource-summary.json').exists() else None
    if complete:
        if set(groups)!={'I','V'} or not resources or resources['error']:raise ValueError('Incomplete experiment/resources')
        m=groups['V'];gates=dict(C=m['C']['count']>=56,F=m['F']['count']<=3,boundary=m['boundary']['count']>=27,U=m['U']['count']>=11)
        gain=m['Q']['count']-groups['I']['Q']['count'];verdict=dict(gates=gates,Q_gain=gain,promising=all(gates.values()) and gain>=4)
    examples=[]
    if len(allrows)==2:
        for c,i,v in zip(cases,allrows['I'],allrows['V']):
            if i['score']['success']!=v['score']['success']:
                examples.append(dict(id=c['id'],request=c['request'],gold=c['gold'],I=i['output'],V=v['output'],I_correct=i['score']['success'],V_correct=v['score']['success']))
    data=dict(status='complete' if complete else 'partial',fingerprint=f['fingerprint'],metrics=groups,verdict=verdict,resources=resources,examples=examples,
        artifacts={p.name:sha(p) for p in ROOT.glob('free-*.jsonl')},limitation='No training. Same pretrained model and field procedure; visible output also changes format and token cost. Development pilot only.')
    write_json(ROOT/'report-data.json',data);write_json(SOURCE/'reports/fields-first-prompt-results.json',data)
    table=''.join('<tr><td>'+arm+'</td>'+''.join(f'<td>{m[k]["count"]}/{m[k]["n"]}</td>' for k in ['C','Q','F','P','U','J'])+f'<td>{m["format_failures"]}</td><td>{m["mean_output_tokens"]:.1f}</td></tr>' for arm,m in groups.items())
    state='尚未完成，不作联合判定' if verdict is None else ('达到提示试验的积极信号标准' if verdict['promising'] else '未达到提示试验的积极信号标准')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段先行提示对照</title><style>body{{font:17px system-ui;max-width:1120px;margin:40px auto;padding:20px;background:#101820;color:#e5edf3}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #456;text-align:left}}section{{background:#192630;padding:24px;border-radius:12px;margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}a{{color:#9de7d0}}</style><h1>字段先行：不微调的配对提示对照</h1><p>I 内部检查后只输出动作；V 先输出字段，再输出动作。原模型相同、任务相同、字段检查步骤相同。</p><section><h2>{state}</h2><table><tr><th>组</th><th>完整C ↑</th><th>缺参Q ↑</th><th>误追问F ↓</th><th>猜测P ↓</th><th>不支持U ↑</th><th>配对J ↑</th><th>格式失败</th><th>输出token均值</th></tr>{table}</table></section><section><h2>完整指标与同题案例</h2><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></section><p>该试验检验输出过程，不代表字段先行微调效果。开发集已使用多轮；封存测试未打开。</p><p><a href="/">实时进度</a> · <a href="http://127.0.0.1:49407/results">动作加权结果</a></p></html>'''
    (ROOT/'results.html').write_text(page);(SOURCE/'reports/fields-first-prompt-results.html').write_text(page)
    checks=[dict(name=a,complete=f'{m["C"]["count"]}/60',clarify=f'{m["Q"]["count"]}/48',unsupported=f'{m["U"]["count"]}/12') for a,m in groups.items()]
    update_state(ROOT,results_ready=True,checkpoints=checks)
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if args.serve:
        from toolcall_lab import live_progress
        live_progress.HTML=MONITOR_HTML.replace('已见训练样本回放','字段先行提示对照').replace('训练样本回放','字段先行对照').replace('固定120题 · 原模型与M0/M1/M2最新75更新检查点 · 自由生成与teacher forcing · 只评测，不训练','同一原模型 · I 内部字段检查 / V 显式字段先行 · 各120题 · 不微调')
        sys.argv=[sys.argv[0],'--root',str(ROOT),'--port','49409'];live_progress.main();return
    import fcntl
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    beat=Heartbeat(ROOT);beat.start();resources=None
    try:
        update_state(ROOT,stage='audit',status='running',step=0,total_steps=120,message='核验第一阶段已完成且未达目标，然后审计字段标签与输入。')
        f,cases=freeze()
        if (ROOT/'completed.json').exists():
            verify_marker(read(ROOT/'completed.json'),f['fingerprint'],sha(ROOT/'report-data.json'))
            verify_file_map(ROOT,read(ROOT/'report-data.json')['artifacts'])
            update_state(ROOT,stage='complete',status='complete',message='已完成，来源与原始输出核验通过。');print('Already complete');return
        resources=ResourceMonitor(ROOT/f'resources-{int(time.time())}.jsonl');resources.start()
        bp=ROOT/'resource-baseline.json'
        if bp.exists():
            initial=read(bp);current=resources.samples[0];resources.samples.insert(0,initial)
            problem=resource_violation(current,initial['swap_used_bytes'])
            if problem:resources.error=problem
        else:write_json(bp,resources.samples[0])
        resources.check()
        import mlx.core as mx
        from mlx_lm import load
        configure_mlx_memory(mx);model,tokenizer=load(f['snapshot']);resources.check()
        try:
            for arm in ['I','V']:evaluate(model,tokenizer,cases,f,arm,resources);report(f,cases)
        finally:del model,tokenizer;mx.clear_cache()
        resources.stop();resources.check();write_json(ROOT/'resource-summary.json',resources.summary());resources=None
        data=report(f,cases,True)
        write_json(ROOT/'completed.json',dict(status='complete',fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json')))
        update_state(ROOT,stage='complete',status='complete',finished_at=time.time(),message='两组共240条输出已重评分，字段先行提示对照完成。')
        print(json.dumps(data['verdict']),flush=True)
    except BaseException as exc:
        update_state(ROOT,status='failed',message=repr(exc));raise
    finally:
        if resources:resources.stop();write_json(ROOT/'resource-summary.json',resources.summary())
        beat.stop();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
