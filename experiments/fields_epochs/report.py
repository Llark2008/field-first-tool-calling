"""Rescore all raw cells, retain rejection denominators and enforce full grid."""
from collections import Counter
import html
import math
from pathlib import Path
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from experiments.fields_sft.scoring import score
from experiments.fields_sft.evaluate import teacher_summary
from experiments.fields_sft.resources import verify_sessions
from experiments.field_action_diagnostics.run import read,write,deterministic,summarize,contrast,publish_report
from .core import require_grid,RATES
from .train import events_for,validate_training
from .evaluate import paths,meta,validate_raw

def check_teacher(cases,rows,targets):
    if len(cases)!=len(rows):raise ValueError('Incomplete teacher set')
    for c,r in zip(cases,rows):
        t=targets[c['id']]
        if r['case_id']!=c['id'] or any(r.get(k)!=t[k] for k in ['tokens','offset','groups']):raise ValueError('Teacher target changed')
        n=len(t['tokens'])-t['offset']
        if len(r['losses'])!=n or len(r['correct'])!=n or any(not math.isfinite(x) or x<0 for x in r['losses']) or any(type(v)!=bool for v in r['correct']):raise ValueError('Invalid teacher values')
    return rows

def scored_rows(cases,raw):
    if len(cases)!=len(raw) or [c['id'] for c in cases]!=[r['case_id'] for r in raw]:raise ValueError('Output order mismatch')
    return [dict(case_id=c['id'],condition=c['condition'],target_tool=c['target_tool'],missing_fields=c['missing_fields'],template_group=c['template_group'],
                 output=r['output'],finish_reason=r['finish_reason'],**score(c,r['output'],r['finish_reason'],'F')) for c,r in zip(cases,raw)]

def cell_scores(cases,raw):
    rows=scored_rows(cases,raw);converted=deterministic(cases,rows);fields=[r['field_diagnostics'] for r in rows]
    return dict(n=len(rows),model=summarize(cases,rows),D=summarize(cases,converted),fields=dict(exact=sum(x['fields_exact'] for x in fields),
        tool_correct=sum(x['tool_correct'] for x in fields),self_consistent=sum(x['self_consistent'] for x in fields),
        correct_fields_wrong_action=sum(x['fields_exact'] and not r['score']['success'] for x,r in zip(fields,rows)),
        missing_null_correct=sum(x['required_missing_correct'] for x in fields),missing_null_n=sum(x['required_missing_n'] for x in fields)),
        conversion_change=contrast(rows,converted))

def build(root,f,source,complete=False,expected=None):
    cases={s:read_rows(Path(f['parent'])/f'{s}-cases.jsonl') for s in ['dev','seen']};targets=read(root/'teacher-targets.json')
    cells={};teachers={};rawmap={};artifacts={}
    for arm in RATES:
        for epoch in [1,2,3]:
            for split in cases:
                key=f'{arm}-{epoch}-{split}'
                for kind in ['free','teacher']:
                    p=paths(root,arm,epoch,split,kind)
                    if not p.exists():continue
                    mp=p.with_suffix('.meta.json')
                    if read(mp)!=meta(root,f,arm,epoch,split,kind):raise ValueError('Evaluation metadata changed')
                    rows=read_rows(p)
                    if len(rows)>120:raise ValueError('Too many evaluation rows')
                    if kind=='free':validate_raw(cases[split],rows,f['prompt_hashes'][split],True)
                    if len(rows)!=120:continue
                    if kind=='free':
                        cells[key]=dict(arm=arm,epoch=epoch,split=split,**cell_scores(cases[split],rows));rawmap[key]=rows
                    else:
                        check_teacher(cases[split],rows,targets[split]);teachers[key]=dict(arm=arm,epoch=epoch,split=split,**teacher_summary(rows)['all'])
                    artifacts[p.name]=sha(p);artifacts[mp.name]=sha(mp)
    training={a:events_for(root,f,a) for a in RATES}
    result=dict(status='complete' if complete else 'partial',fingerprint=f['fingerprint'],cells=cells,teacher=teachers,training_events=training,
        audit=f['audit'],rates=RATES,artifacts=artifacts,reused=dict(training_updates_A=75,free_generations_A1=240,teacher_rows_A1_seen=120),
        limitations=['One training seed; repeatedly used synthetic dev set.','Fixed seen120 is a subset, not the entire600 training set.','D is a system using model-predicted fields; model scores remain independent.','Teacher loss uses gold histories and token averaging, unlike online sample-mean loss.','No sealed test or checkpoint promotion; epoch3 is the fixed primary comparison.'])
    if complete:
        require_grid(cells);require_grid({k:dict(n=120) for k in teachers})
        for a in RATES:
            for e in [1,2,3]:validate_training(root,f,a,e)
        result['resources']=verify_sessions(root,True)
        result['resource_summaries']=[read(p) for p in sorted(root.glob('resource-summary-*.json'))]
        result['contrasts']={}
        for split in cases:
            for mode in ['model','D']:
                def get(a,e):
                    rs=scored_rows(cases[split],rawmap[f'{a}-{e}-{split}'])
                    return deterministic(cases[split],rs) if mode=='D' else rs
                result['contrasts'][split+'-'+mode]=dict(B3_minus_A3=contrast(get('A',3),get('B',3)),A3_minus_A1=contrast(get('A',1),get('A',3)),B3_minus_B1=contrast(get('B',1),get('B',3)))
        result['examples']=[]
        for c in cases['dev']:
            i=next(i for i,v in enumerate(cases['dev']) if v['id']==c['id'])
            items={f'{a}-{e}':dict(output=rawmap[f'{a}-{e}-dev'][i]['output'],**score(c,rawmap[f'{a}-{e}-dev'][i]['output'],rawmap[f'{a}-{e}-dev'][i]['finish_reason'],'F')) for a in RATES for e in [1,2,3]}
            result['examples'].append(dict(id=c['id'],request=c['request'],gold=c['gold'],condition=c['condition'],checkpoints=items))
        for p in [*root.glob('train-*/*.json'),*root.glob('train-*/*.jsonl'),*root.glob('train-*/step-*/*'),*root.glob('resource*.json*')]:
            if p.is_file():result['artifacts'][str(p.relative_to(root))]=sha(p)
    publish_report(root/'report-data.json',result,expected)
    update_state(root,cells=list(cells.values()),teacher_cells=list(teachers.values()),**{'loss_'+a:es for a,es in training.items()})
    if complete:render(root,source,result)
    return result

def render(root,source,result):
    lines=['# 固定600条：学习率与三轮训练对照','','A=1e-4，B=5e-5。数据、目标、样本顺序及优化器其他配置保持一致。','',
           '| 组 | 轮次 | 题集 | 输出方式 | 完整 /60 | 缺参 /48 | 误追问 /60 ↓ | 猜参 /48 ↓ | 不支持 /12 | 总正确 /120 |',
           '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for c in result['cells'].values():
        for mode in ['model','D']:lines.append('| '+f'{c["arm"]} | {c["epoch"]} | {c["split"]} | {mode} | '+' | '.join(str(c[mode][k]) for k in ['C','Q','F','P','U','success'])+' |')
    lines+=['','限制：'+ ' '.join(result['limitations']),'','实验指纹：'+result['fingerprint']]
    (source/'reports/fields-epochs-results.md').write_text('\n'.join(lines)+'\n');write(source/'reports/fields-epochs-results.json',result)
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in ['组','轮','题集','方式','完整/60','缺参/48','误追问/60↓','猜参/48↓','不支持/12','总正确/120'])+'</tr>'
    for c in result['cells'].values():
        for mode in ['model','D']:table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [c['arm'],c['epoch'],c['split'],mode,*[c[mode][k] for k in ['C','Q','F','P','U','success']]])+'</tr>'
    table+='</table>'
    doc='<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三轮训练结果</title><style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:20px;line-height:1.7;color:#193241;background:#f5f8fa}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd9e0;text-align:left}pre{white-space:pre-wrap;word-break:break-word;background:#e9eff3;padding:14px}details{padding:15px;border:1px solid #ccd9e0;margin:10px 0;border-radius:10px}summary{cursor:pointer}.scroll{overflow-x:auto}</style><h1>600条数据 × 三轮训练</h1><p>A=1e-4，B=5e-5；model=模型独立输出，D=确定性转换。已见训练和开发集分开显示。</p><div class="scroll">'+table+'</div><p>'+html.escape(' '.join(result['limitations']))+'</p><h2>开发集逐题输出</h2>'
    for e in result['examples']:
        doc+='<details><summary>'+html.escape(e['id'])+'</summary><p>'+html.escape(e['request'])+'</p><pre>标准动作 '+html.escape(str(e['gold']))+'</pre>'
        for k,r in e['checkpoints'].items():doc+='<b>'+k+' '+('✓' if r['score']['success'] else '✗')+'</b><pre>'+html.escape(r['output'])+'</pre>'
        doc+='</details>'
    doc+='</html>';(root/'results.html').write_text(doc);(source/'reports/fields-epochs-results.html').write_text(doc)
