"""Inference-only diagnosis. Parent artifacts are read-only and never rebuilt."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time
from toolcall_lab.formal_data import sha
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.live_progress import update_state,Heartbeat
from experiments.fields_sft.data import compact,digest,messages,encoded_target
from experiments.fields_sft.scoring import score,equal
from experiments.fields_first_prompt.core import field_target
from .core import convert,first_object,mapping_messages,parse_action,action_matches,probes
from .progress import ROOT

SOURCE=Path(__file__).resolve().parents[2]
PARENT=ROOT.parent/'fields-sft-20261115'
ADAPTER=PARENT/'train-F/step-0600'
DOCS=[SOURCE/'docs/superpowers'/p for p in ['specs/2026-11-16-field-action-diagnostics.md','plans/2026-11-16-field-action-diagnostics.md']]
ARMS=('G','S','P')

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False));tmp.replace(p)
def immutable(p,obj):
    if p.exists() and read(p)!=obj:raise ValueError('Refusing changed frozen artifact: '+str(p))
    if not p.exists():write(p,obj)
def sources():
    # Explicit dependency packages, not every future experiment added to the workspace.
    packages=['toolcall_lab','experiments/field_action_diagnostics','experiments/fields_sft','experiments/fields_first_prompt','experiments/action_weighting']
    files=[p for package in packages for p in (SOURCE/package).rglob('*.py')]+DOCS
    replay=SOURCE/'scripts/run_train_replay.py'
    if replay.exists():files.append(replay)
    return {str(p.relative_to(SOURCE)):sha(p) for p in sorted(set(files))}

def publish_report(path,result,expected=None):
    if expected is not None and result!=expected:raise ValueError('Recomputed report changed; preserved original bytes')
    if expected is None:write(path,result)

def validate_raw(jobs,rows,partial=False):
    if len(rows)>len(jobs) or (not partial and len(rows)!=len(jobs)):raise ValueError('Wrong output count')
    for job,row in zip(jobs,rows):
        if any(job[k]!=row[k] for k in ['id','arm','prompt_sha256']):raise ValueError('Output/prompt identity changed')
        if not isinstance(row['output'],str) or row['finish_reason'] not in ['stop','length']:raise ValueError('Invalid raw output')
    return rows

def make_job(case,arm,tok):
    if arm not in ARMS:raise ValueError('Unknown arm')
    first=case['first'] if arm=='P' else field_target(case)
    if arm=='G':
        prompt=tok.apply_chat_template(messages(case,'F'),tokenize=False,add_generation_prompt=True,enable_thinking=False)+compact(first)+'\n'
    else:prompt=tok.apply_chat_template(mapping_messages(case['tools'],first),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(prompt)
    if len(ids)>2560:raise ValueError('Prompt exceeds fixed diagnostic budget')
    return dict(id=case['id'],arm=arm,first=first,prompt_text=prompt,prompt_ids=ids,prompt_sha256=digest(ids))

def summarize(cases,rows):
    if [c['id'] for c in cases]!=[r['case_id'] for r in rows]:raise ValueError('Scoring alignment changed')
    byid={r['case_id']:r for r in rows};out=dict(n=len(rows),success=sum(bool(r['score']['success']) for r in rows),
        valid=sum(bool(r['score']['schema_valid']) and not r['score'].get('truncated',False) for r in rows),rejected=sum(bool(r.get('rejection')) for r in rows))
    for key,cond,kind in [('C','complete',None),('Q','natural',None),('U','unsupported',None),('F','complete','clarify'),('P','natural','call')]:
        subset=[byid[c['id']] for c in cases if c['condition']==cond]
        out[key]=sum(bool(r['score']['success']) if kind is None else r['score'].get('action',{}).get('action')==kind for r in subset)
        out[key+'_n']=len(subset)
    complete={c['intent_group']:c['id'] for c in cases if c['condition']=='complete'}
    paired=[(byid[c['id']],byid[complete[c['intent_group']]]) for c in cases if c['condition']=='natural' and c['intent_group'] in complete]
    out['J']=sum(a['score']['success'] and b['score']['success'] and not a.get('guard',{}).get('blocked') and not b.get('guard',{}).get('blocked') for a,b in paired)
    out['J_n']=len(paired);out['failures']=dict(Counter(r.get('failure_type','unknown') for r in rows if not r['score']['success']))
    return out

def score_action(c,raw,finish='stop',**extra):
    error=None
    try:parse_action(raw)
    except (ValueError,TypeError) as exc:error=str(exc)
    scored=score(c,raw if error is None else 'INVALID_FORMAT',finish,'J')
    return dict(case_id=c['id'],condition=c['condition'],target_tool=c['target_tool'],missing_fields=c['missing_fields'],
        template_group=c['template_group'],output=raw,finish_reason=finish,strict_parse_error=error,**scored,**extra)

def deterministic(cases,base):
    rows=[]
    for c,b in zip(cases,base):
        if c['id']!=b['case_id']:raise ValueError('Baseline alignment changed')
        first=None;rejection=None
        try:first=first_object(b['output'],b['finish_reason']);action=convert(c['tools'],first);raw=compact(action)
        except (ValueError,TypeError) as exc:rejection=str(exc);raw='CONVERSION_REJECTED'
        rows.append(score_action(c,raw,rejection=rejection,predicted_fields=first))
    return rows

def contrast(a,b):
    if [r['case_id'] for r in a]!=[r['case_id'] for r in b]:raise ValueError('Contrast misaligned')
    wins=[x['case_id'] for x,y in zip(a,b) if not x['score']['success'] and y['score']['success']]
    losses=[x['case_id'] for x,y in zip(a,b) if x['score']['success'] and not y['score']['success']]
    return dict(wins=wins,losses=losses,net=len(wins)-len(losses))

def load_inputs():
    return read_rows(PARENT/'dev-cases.jsonl'),read_rows(PARENT/'seen-cases.jsonl')

def audit():
    from mlx_lm.utils import load_tokenizer
    from experiments.fields_sft.run import verify
    from experiments.fields_sft.evaluate import validate_rows,metadata
    update_state(ROOT,stage='audit',status='running',detail='输入与前缀审计',step=0,total_steps=240,message='核对历史模型、标签、严格转换规则和助手前缀边界。')
    parent=read(PARENT/'frozen.json');verify(parent)
    if read(PARENT/'completed.json')['report_sha256']!=sha(PARENT/'report-data.json'):raise ValueError('Parent signed report changed')
    dev,seen=load_inputs();tok=load_tokenizer(Path(parent['snapshot']))
    for split,cases in [('dev',dev),('seen',seen)]:
        base=read_rows(PARENT/f'{split}-F-600.jsonl');validate_rows(cases,base,'F')
        if read(PARENT/f'{split}-F-600.meta.json')!=metadata(PARENT,parent,cases,'F',600,split):raise ValueError('Parent output metadata mismatch')
        for c in cases:
            action=convert(c['tools'],field_target(c))
            if not action_matches(c['tools'],c['gold'],action) or not score_action(c,compact(action))['score']['success']:raise ValueError('Gold conversion positive control failed: '+c['id'])
    ps=probes(dev);jobs=[]
    for arm in ARMS:
        for c in ps if arm=='P' else dev:
            job=make_job(c,arm,tok)
            if arm=='G':
                _,tokens,offset,groups=encoded_target(c,'F',tok)
                if job['prompt_ids']!=tokens[:len(job['prompt_ids'])] or len(job['prompt_ids'])!=min(groups['action']):raise ValueError('Oracle prefix contains/misses action boundary: '+c['id'])
            jobs.append(job)
    for p in ps:
        if not action_matches(p['tools'],p['gold'],convert(p['tools'],p['first'])):raise ValueError('Independent probe target disagrees')
    immutable(ROOT/'jobs.json',jobs);immutable(ROOT/'probes.json',ps)
    result=dict(status='passed',gold_converter_positive_controls=240,oracle_prefix_boundaries=120,dev_cases=len(dev),seen_cases=len(seen),
        new_generations=dict(Counter(j['arm'] for j in jobs)),prompt_tokens={a:dict(min=min(len(j['prompt_ids']) for j in jobs if j['arm']==a),max=max(len(j['prompt_ids']) for j in jobs if j['arm']==a)) for a in ARMS},
        labels_in_D=False,training=False,sealed_test_used=False)
    immutable(ROOT/'audit.json',result)
    deterministic_results={s:deterministic(c,read_rows(PARENT/f'{s}-F-600.jsonl')) for s,c in [('dev',dev),('seen',seen)]}
    immutable(ROOT/'deterministic.json',deterministic_results)
    comparisons=[dict(arm='B 原始模型',**summarize(dev,read_rows(PARENT/'dev-F-600.jsonl'))),dict(arm='D 确定性转换',**summarize(dev,deterministic_results['dev']))]
    update_state(ROOT,status='pending',step=240,audit=result,comparisons=comparisons,message='审计完成；确定性转换已重评分，等待冻结后进行动作生成诊断。')
    print(compact(dict(audit=result,comparisons=comparisons)),flush=True)

def freeze():
    audit();parent=read(PARENT/'frozen.json')
    inputs=[PARENT/n for n in ['frozen.json','completed.json','report-data.json','dev-cases.jsonl','seen-cases.jsonl','dev-F-600.jsonl','seen-F-600.jsonl','dev-F-600.meta.json','seen-F-600.meta.json']]
    inputs+=list(ADAPTER.iterdir())
    inputs+=[Path(parent['snapshot'])/n for n in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json'] if (Path(parent['snapshot'])/n).exists()]
    f=dict(parent_fingerprint=parent['fingerprint'],sources=sources(),inputs={str(p):sha(p) for p in inputs if p.is_file()},
           snapshot=parent['snapshot'],adapter=str(ADAPTER),local_files={n:sha(ROOT/n) for n in ['audit.json','jobs.json','probes.json','deterministic.json']},
           generation=dict(temperature=0,max_tokens=384,thinking=False),training=False)
    f['fingerprint']=digest(f);immutable(ROOT/'frozen.json',f)
    for rel in f['sources']:
        p=ROOT/'source-snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((SOURCE/rel).read_bytes())
    print('Frozen '+f['fingerprint'],flush=True)

def verify(f):
    obj=dict(f);fingerprint=obj.pop('fingerprint')
    if digest(obj)!=fingerprint or sources()!=f['sources']:raise ValueError('Frozen source changed')
    for p,h in f['inputs'].items():
        if sha(Path(p))!=h:raise ValueError('Parent/model input changed: '+p)
    for n,h in f['local_files'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen local input changed: '+n)

def all_rows(f,partial=False):
    jobs=read(ROOT/'jobs.json');result={}
    for arm in ARMS:
        path=ROOT/f'raw-{arm}.jsonl';aj=[j for j in jobs if j['arm']==arm]
        rows=read_rows(path) if path.exists() else []
        if path.exists() and read(path.with_suffix('.meta.json'))!=dict(fingerprint=f['fingerprint'],arm=arm):raise ValueError('Raw manifest changed')
        result[arm]=validate_raw(aj,rows,partial)
    return result

def generate(f):
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    from toolcall_lab.train_formal import configure_mlx_memory
    from toolcall_lab.resource_monitor import ResourceMonitor,resource_violation
    jobs=read(ROOT/'jobs.json');previous=all_rows(f,True);session=str(time.time_ns());resources=ResourceMonitor(ROOT/f'resources-{session}.jsonl')
    resources.start();bp=ROOT/'resource-baseline.json'
    if bp.exists():
        initial=read(bp);problem=resource_violation(resources.samples[0],initial['swap_used_bytes'])
        resources.samples.insert(0,initial)
        if problem:resources.error=problem
    else:write(bp,resources.samples[0])
    model=tok=None
    try:
        update_state(ROOT,stage='loading',status='running',detail='加载固定 F@75',step=0,total_steps=len(jobs),message='仅加载模型和 LoRA 权重；不加载优化器，不执行训练。')
        resources.check();configure_mlx_memory(mx);model,tok=load(f['snapshot'],adapter_path=f['adapter']);model.eval();resources.check()
        for arm in ARMS:
            aj=[j for j in jobs if j['arm']==arm];rows=previous[arm];path=ROOT/f'raw-{arm}.jsonl'
            immutable(path.with_suffix('.meta.json'),dict(fingerprint=f['fingerprint'],arm=arm))
            update_state(ROOT,stage=arm,status='running',detail={'G':'G 正确字段 · 原提示续写','S':'S 正确字段 · 独立动作映射','P':'P 字段最小改动测试'}[arm],step=len(rows),total_steps=len(aj),message='标准字段仅用于诊断；无动作答案提示，无修复重试。')
            for job in aj[len(rows):]:
                resources.check();parts=[];last=None;tick=time.perf_counter()
                for i,response in enumerate(stream_generate(model,tok,prompt=job['prompt_ids'],max_tokens=384,sampler=make_sampler(temp=0))):
                    parts.append(response.text);last=response
                    if i%16==0:resources.check()
                if last is None:raise ValueError('Empty generation')
                row=dict(id=job['id'],arm=arm,prompt_sha256=job['prompt_sha256'],output=''.join(parts),finish_reason=last.finish_reason,
                         seconds=time.perf_counter()-tick,prompt_tokens=int(last.prompt_tokens),generation_tokens=int(last.generation_tokens))
                with path.open('a') as h:h.write(compact(row)+'\n');h.flush()
                rows.append(row);resources.check()
                if len(rows)%5==0 or len(rows)==len(aj):
                    summary=resources.summary();update_state(ROOT,step=len(rows),metrics=dict(mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
                        peak_rss_GiB=summary['peak_sampled_rss_bytes']/1024**3,peak_swap_growth_GiB=summary['peak_swap_growth_bytes']/1024**3,generated=len(rows)))
                    print(arm,len(rows),'/',len(aj),flush=True)
                if len(rows)%40==0:mx.clear_cache()
            validate_raw(aj,rows);report(f,partial=True)
    finally:
        if model is not None:del model
        if tok is not None:del tok
        mx.clear_cache();resources.stop();write(ROOT/f'resource-summary-{session}.json',dict(resources.summary(),raw_sha256=sha(resources.path),raw_file=resources.path.name));resources.check()

def resource_check():
    from toolcall_lab.resource_monitor import resource_violation
    files=sorted(ROOT.glob('resources-*.jsonl'))
    if not files:raise ValueError('No inference resource evidence')
    baseline=read(ROOT/'resource-baseline.json')['swap_used_bytes'];summaries=[]
    for path in files:
        summary=read(ROOT/path.name.replace('resources-','resource-summary-').replace('.jsonl','.json'))
        if summary['raw_sha256']!=sha(path) or summary['error']:raise ValueError('Resource session failed or changed')
        for sample in read_rows(path):
            if resource_violation(sample,baseline):raise ValueError('Historical resource limit exceeded')
        summaries.append(summary)
    return summaries

def report(f,partial=False,expected=None):
    dev,seen=load_inputs();raw=all_rows(f,partial);base=read_rows(PARENT/'dev-F-600.jsonl');det=read(ROOT/'deterministic.json')
    # Independently rescore deterministic outputs; never trust stored metrics.
    for split,cases in [('dev',dev),('seen',seen)]:
        if det[split]!=deterministic(cases,read_rows(PARENT/f'{split}-F-600.jsonl')):raise ValueError('Deterministic scoring changed')
    rows={'B':base,'D':det['dev']}
    for arm in ['G','S']:
        if len(raw[arm])==len(dev):rows[arm]=[score_action(c,r['output'],r['finish_reason']) for c,r in zip(dev,raw[arm])]
    metrics={a:summarize(dev,r) for a,r in rows.items()};seenbase=read_rows(PARENT/'seen-F-600.jsonl')
    conditional={}
    for a,rs in rows.items():
        ids=[i for i,b in enumerate(base) if b['field_diagnostics']['fields_exact']]
        conditional[a]=dict(n=len(ids),success=sum(rs[i]['score']['success'] for i in ids),failed_ids=[dev[i]['id'] for i in ids if not rs[i]['score']['success']])
    ps=read(ROOT/'probes.json');pr=[]
    for p,r in zip(ps,raw['P']):
        try:action=parse_action(r['output']);ok=action_matches(p['tools'],p['gold'],action) and r['finish_reason']!='length'
        except (ValueError,TypeError):ok=False
        pr.append(dict(id=p['id'],source_id=p['source_id'],kind=p['kind'],first=p['first'],gold=p['gold'],output=r['output'],finish_reason=r['finish_reason'],success=ok))
    examples=[]
    for i,c in enumerate(dev):
        examples.append(dict(id=c['id'],condition=c['condition'],request=c['request'],gold=c['gold'],correct_fields=field_target(c),predicted_fields=det['dev'][i]['predicted_fields'],
            original_fields_correct=base[i]['field_diagnostics']['fields_exact'],arms={a:dict(output=r[i]['output'],success=r[i]['score']['success'],failure_type=r[i]['failure_type'],rejection=r[i].get('rejection')) for a,r in rows.items()}))
    result=dict(kind='no_training_field_action_diagnosis',status='partial' if partial else 'complete',fingerprint=f['fingerprint'],metrics=metrics,
        contrasts={a+'-B':contrast(base,r) for a,r in rows.items() if a!='B'},conditional_on_original_correct_fields=conditional,
        seen=dict(B=summarize(seen,seenbase),D=summarize(seen,det['seen']),contrast=contrast(seenbase,det['seen'])),
        probe_metrics={kind:dict(n=sum(p['kind']==kind for p in pr),success=sum(p['success'] for p in pr if p['kind']==kind)) for kind in sorted({p['kind'] for p in pr})},
        probes=pr,examples=examples,generation={a:dict(n=len(rs),truncated=sum(r['finish_reason']=='length' for r in rs),seconds=sum(r['seconds'] for r in rs),output_tokens=sum(r['generation_tokens'] for r in rs)) for a,rs in raw.items()},
        limitations=['D is a model+deterministic system, not improved model weights.','G/S use correct fields, not end-to-end accuracy.','S changes prompt and removes original request/context; not a one-variable comparison.','P small constructed mapping probes, separate from dev120.','Single checkpoint and repeatedly used synthetic dev set; sealed test unused.','No LR/epoch causal effect measured.'])
    if not partial:
        result['resources']=resource_check()
        result['raw_sha256']={str(p.name):sha(p) for p in sorted([*ROOT.glob('raw-*.jsonl'),*ROOT.glob('raw-*.meta.json'),*ROOT.glob('resources-*.jsonl'),*ROOT.glob('resource-summary-*.json')])}
    publish_report(ROOT/'report-data.json',result,expected)
    update_state(ROOT,comparisons=[dict(arm=a,**m) for a,m in metrics.items()])
    if not partial:render(result)
    return result

def render(r):
    labels={'B':'原始模型','D':'模型字段 → 确定性转换','G':'正确字段 → 原提示续写（诊断）','S':'正确字段 → 独立映射（诊断）'}
    lines=['# 字段到动作：无训练诊断结果','', '固定 F@75 / Qwen3-1.7B-4bit；无新训练。D不读取标准字段，G/S使用标准字段，仅作诊断。','',
        '| 方案 | 完整 /60 | 缺参 /48 | 误追问 /60 ↓ | 猜参调用 /48 ↓ | 不支持 /12 | 总正确 /120 | 配对全对 /48 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,m in r['metrics'].items():lines.append('| '+labels[a]+' | '+' | '.join(str(m[k]) for k in ['C','Q','F','P','U','success','J'])+' |')
    lines+=['','模型原本字段全对的95题，最终动作正确：'+', '.join(a+' '+str(m['success'])+'/'+str(m['n']) for a,m in r['conditional_on_original_correct_fields'].items())+'。','',
            'D相对B：改善'+str(len(r['contrasts']['D-B']['wins']))+'题，退步'+str(len(r['contrasts']['D-B']['losses']))+'题。',
            '已见训练120题：B '+str(r['seen']['B']['success'])+'，D '+str(r['seen']['D']['success'])+'。','',
            '字段改动探针：'+compact(r['probe_metrics']),'','限制：'+ ' '.join(r['limitations']),'', '完整逐题输出见同名JSON与HTML。指纹：'+r['fingerprint']]
    (SOURCE/'reports/field-action-diagnostics-results.md').write_text('\n'.join(lines)+'\n')
    write(SOURCE/'reports/field-action-diagnostics-results.json',r)
    esc=html.escape
    table='<table><tr><th>方案</th><th>完整 /60</th><th>缺参 /48</th><th>误追问 /60 ↓</th><th>猜参调用 /48 ↓</th><th>不支持 /12</th><th>总正确 /120</th><th>配对全对 /48</th></tr>'
    for a,m in r['metrics'].items():table+='<tr><td>'+esc(labels[a])+'</td>'+''.join('<td>'+str(m[k])+'</td>' for k in ['C','Q','F','P','U','success','J'])+'</tr>'
    table+='</table>'
    cases=''
    # All cases retained; cases changed by D or incorrect under G appear first.
    order=sorted(r['examples'],key=lambda x:(not(x['arms']['B']['success']!=x['arms']['D']['success']),x['arms'].get('G',{}).get('success',False),x['id']))
    for e in order:
        cases+='<details class="example"><summary>'+esc(e['id'])+' · '+ ' / '.join(a+(' ✓' if v['success'] else ' ✗') for a,v in e['arms'].items())+'</summary><p>'+esc(e['request'])+'</p><pre>模型字段：'+esc(compact(e['predicted_fields']))+'\n正确字段：'+esc(compact(e['correct_fields']))+'\n标准动作：'+esc(compact(e['gold']))+'</pre>'
        for a,v in e['arms'].items():cases+='<p><b>'+esc(labels[a])+'</b> '+('正确' if v['success'] else esc(v['failure_type']))+'</p><pre>'+esc(v['output'])+'</pre>'
        cases+='</details>'
    probes_html=''.join('<details><summary>'+esc(p['id'])+(' ✓' if p['success'] else ' ✗')+'</summary><pre>'+esc(compact(p['first']))+'\n标准：'+esc(compact(p['gold']))+'\n输出：'+esc(p['output'])+'</pre></details>' for p in r['probes'])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段到动作 · 无训练诊断</title><style>body{font:16px system-ui;background:#f2f6f7;color:#18303d;max-width:1250px;margin:36px auto;padding:0 24px;line-height:1.7}h1{font-size:32px}.card,details{background:white;border:1px solid #c9d8df;border-radius:12px;padding:18px;margin:14px 0}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #d4e0e5;text-align:left}pre{white-space:pre-wrap;word-break:break-word;background:#eff4f6;padding:14px;border-radius:8px;font-size:13px}summary{cursor:pointer;font-weight:600}input{padding:12px;width:95%;border:1px solid #98b2bf;border-radius:8px}.muted{color:#58727f}.scroll{overflow-x:auto}</style><h1>字段到动作：无训练诊断</h1><p class="muted">F@75 · Qwen3-1.7B-4bit · 固定120道开发题 · 本轮没有训练</p><div class="card"><b>如何读这张表</b><p>B是原始生成；D只使用模型自己预测的字段，通过程序转换动作。G/S提供正确字段，不能当作真实任务准确率。S还改变了提示词并移除了原请求上下文。</p></div>'''+ '<div class="card scroll">'+table+'</div><div class="card"><pre>'+esc('\n'.join(lines[11:]))+'</pre></div><h2>逐题对比</h2><input id="search" placeholder="搜索案例、工具、字段或输出"><div id="cases">'+cases+'</div><h2>字段最小改动探针</h2><p>只提供工具定义和字段，无相互矛盾的原始请求。这里检查动作映射，不声称实际任务执行准确率。</p>'+probes_html+'''<script>document.getElementById('search').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('#cases details').forEach(d=>{d.hidden=!d.textContent.toLowerCase().includes(q)})});</script></html>'''
    (ROOT/'results.html').write_text(doc);(SOURCE/'reports/field-action-diagnostics-results.html').write_text(doc)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['audit','freeze','run','report','verify']);args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat=Heartbeat(ROOT);beat.start()
    try:
        if args.stage=='audit':audit();return
        if args.stage=='freeze':freeze();return
        f=read(ROOT/'frozen.json');verify(f)
        if (ROOT/'completed.json').exists():
            marker=read(ROOT/'completed.json')
            if marker['fingerprint']!=f['fingerprint'] or marker['report_sha256']!=sha(ROOT/'report-data.json'):raise ValueError('Completion marker changed')
            original=read(ROOT/'report-data.json');rebuilt=report(f,expected=original)
            if original!=rebuilt:raise ValueError('Recomputed report changed')
            print('Complete outputs, input/model identity and scores verified',flush=True);return
        if args.stage=='run':generate(f)
        if args.stage in ['run','report','verify']:
            verify(f);r=report(f)
            write(ROOT/'completed.json',dict(fingerprint=f['fingerprint'],report_sha256=sha(ROOT/'report-data.json'),finished_at=time.time()))
            update_state(ROOT,stage='complete',detail='诊断完成',status='complete',finished_at=time.time(),step=sum(x['n'] for x in r['generation'].values()),total_steps=len(read(ROOT/'jobs.json')),results_ready=True,message='确定性转换、正确字段续写与独立映射已完成并重新评分。模型权重未改变。')
            print(compact(r['metrics']),flush=True)
    except BaseException as exc:
        update_state(ROOT,status='failed',message=repr(exc));raise
    finally:beat.stop();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
