"""Joint behavior gates, all-slot field metrics, and a locally served comparison."""
import html
import json
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.analyze_clarification_ablation import metrics_for
from toolcall_lab.formal_data import sha
from toolcall_lab.live_progress import update_state
from toolcall_lab.run_clarification_ablation import read_json as read,write_json
from .evaluate import metadata,checked,validate_rows,validate_teacher,teacher_summary
from .train import check_training
from .resources import verify_sessions

def verdict(grid,replays,resources_clean):
    expected={f'{a}-{s}' for a in ['J','F'] for s in [0,200,400,600]}
    if set(grid)!=expected or any(v is None for v in grid.values()) or replays!={'J','F'} or not resources_clean:raise ValueError('Incomplete grid/replay or failed resources')
    m=grid['F-600'];floor=max(17,grid['J-600']['Q']['count']+4)
    gates=dict(complete=m['C']['count']>=56,false_clarify=m['F']['count']<=3,boundary=m['boundary']['count']>=27,unsupported=m['U']['count']>=11,missing=m['Q']['count']>=floor)
    return dict(joint_pass=all(gates.values()),gates=gates,missing_floor=floor,Q_gain=m['Q']['count']-grid['J-600']['Q']['count'])

def field_summary(rows):
    ds=[r['field_diagnostics'] for r in rows]
    if any(d is None for d in ds):return None
    keys=['format_valid','tool_correct','fields_exact','missing_set_correct','self_consistent','pipeline_success',
        'required_missing_n','required_missing_correct','required_known_n','required_known_correct','false_null_known','required_slots_covered']
    out={k:sum(d[k] for d in ds) for k in keys};out['cases']=len(rows)
    errors=['required_missing_invented','required_missing_omitted','required_missing_wrong_tool','required_known_wrong_value','required_known_omitted','required_known_wrong_tool','false_null_known']
    legal=[d for d in ds if d['eligible_output']]
    out['legal_output_errors']=dict(cases=len(legal),**{k:sum(d[k] for d in legal) for k in errors})
    return out

def metric(cases,rows):
    out=metrics_for(cases,rows)
    out.update(fields=field_summary(rows),format_failures=sum(r['format_error'] is not None for r in rows),
        schema_valid=sum(r['score']['schema_valid'] and not r['score']['truncated'] for r in rows),
        seconds=sum(r.get('seconds',0) for r in rows),mean_output_tokens=sum(r.get('generation_tokens',0) for r in rows)/len(rows))
    return out

def build(root,f,source,complete=False):
    from mlx_lm.utils import load_tokenizer
    from pathlib import Path
    cases=read_rows(root/'dev-cases.jsonl');seen=read_rows(root/'seen-cases.jsonl');grid={};outputs={};replay={};training={};events={}
    for arm in ['J','F']:
        for s in [0,200,400,600]:
            path=root/f'dev-{arm}-{s}.jsonl'
            if not path.exists():continue
            if s==0:
                if sha(path)!=f['baselines'][arm]['sha256']:raise ValueError('Reused baseline changed')
                rows=validate_rows(cases,read_rows(path),arm)
            else:rows=checked(path,cases,metadata(root,f,cases,arm,s,'dev'),True)
            if len(rows)==len(cases):grid[f'{arm}-{s}']=metric(cases,rows);outputs[arm,s]=rows
        out=root/f'train-{arm}'
        if (out/'summary.json').exists():training[arm]=check_training(root,f,arm)
        events[arm]=read_rows(out/'events.jsonl') if (out/'events.jsonl').exists() else []
        path=root/f'teacher-{arm}.jsonl'
        if path.exists() and len(read_rows(path))==len(seen):
            meta=metadata(root,f,seen,arm,600,'seen')
            if read(path.with_suffix('.meta.json'))!=dict(meta,mode='teacher_forced',targets_sha256=sha(root/f'sft-{arm}.jsonl')):raise ValueError('Teacher metadata changed')
            tf=validate_teacher(seen,read_rows(path),arm,load_tokenizer(Path(f['snapshot'])),f['cap'])
            free=checked(root/f'seen-{arm}-600.jsonl',seen,meta)
            replay[arm]=dict(free=metric(seen,free),teacher=teacher_summary(tf))
    resources=[read(p) for p in sorted(root.glob('resource-summary-*.json'))]
    v=None
    if complete:
        if set(training)!={'J','F'}:raise ValueError('Incomplete training')
        verify_sessions(root,require_complete=True)
        v=verdict(grid,set(replay),bool(resources) and all(r['error'] is None for r in resources))
    examples=[]
    if ('J',600) in outputs and ('F',600) in outputs:
        for c,j,ff in zip(cases,outputs['J',600],outputs['F',600]):
            if j['score']['success']!=ff['score']['success']:
                examples.append(dict(id=c['id'],condition=c['condition'],request=c['request'],gold=c['gold'],J=j['output'],F=ff['output'],J_correct=j['score']['success'],F_correct=ff['score']['success']))
    data=dict(status='complete' if complete else 'partial',fingerprint=f['fingerprint'],audit=f['audit'],development=grid,replay=replay,
        training=training,training_events=events,resources=resources,verdict=v,examples=examples,
        limitations=['单种子、反复使用的开发集，封存测试未使用。','相同样本/更新；F增加字段目标、格式说明和token计算，效果属于整个方法组合。','F最终动作teacher forcing有标准字段前文，不能等同自由生成。','不同目标长度的整体loss不直接排名。'],
        artifacts={str(p.relative_to(root)):sha(p) for pattern in ['dev-*.jsonl','seen-*.jsonl','teacher-*.jsonl','*.meta.json','train-*/events.jsonl','train-*/summary.json','resource-summary-*.json','resources-*.jsonl','resource-baseline.json'] for p in sorted(root.glob(pattern))})
    write_json(root/'report-data.json',data);write_json(source/'reports/fields-sft-results.json',data)
    write_pages(data,root,source)
    update_state(root,results_ready=True,checkpoints=[dict(name=k.replace('-0',' / 原模型') if k.endswith('-0') else f'{k.split("-")[0]} / {int(k.split("-")[1])//8}更新',complete=f'{m["C"]["count"]}/60',clarify=f'{m["Q"]["count"]}/48',unsupported=f'{m["U"]["count"]}/12') for k,m in grid.items()])
    return data

def write_pages(data,root,source):
    grid=data['development'];v=data['verdict'];state='正在运行，尚不作联合结论' if v is None else ('达到预先设定的全部门槛' if v['joint_pass'] else '未达到预先设定的全部门槛')
    rows=''.join('<tr><td>'+k+'</td>'+''.join(f'<td>{m[c]["count"]}/{m[c]["n"]}</td>' for c in ['C','Q','F','P','U','J'])+f'<td>{m["schema_valid"]}/120</td></tr>' for k,m in grid.items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段先行微调对照</title><style>body{{font:17px system-ui;max-width:1180px;margin:30px auto;padding:20px;background:#101820;color:#e5edf3}}section{{background:#192630;border-radius:14px;padding:24px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #456;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}a{{color:#9de7d0}}p{{line-height:1.7}}.scroll{{overflow-x:auto}}</style><h1>字段先行微调：能否改善决策？</h1><p>J：直接生成动作；F：先监督已知字段或 null，再生成动作。相同600条数据、相同顺序，各75次更新。表中200/400/600是微步，对应25/50/75次更新；0是未微调原模型。</p><section><h2>{state}</h2><div class="scroll"><table><tr><th>组／微步</th><th>完整 ↑</th><th>缺参识别 ↑</th><th>误追问 ↓</th><th>猜参调用 ↓</th><th>不支持 ↑</th><th>配对全对 ↑</th><th>动作格式有效</th></tr>{rows}</table></div><p>误追问的分母为60条完整题；缺参识别的分母为48条缺参题，要求完整且准确地列出缺失字段。主要比较固定75更新。</p></section><section><h2>字段与最终动作分别检查</h2><p>字段正确率使用全部应有槽位作分母；格式不合法、字段缺失或工具选错不会从分母删除。字段正确和任务正确分别记录。</p><pre>{html.escape(json.dumps({k:m['fields'] for k,m in grid.items() if m['fields']},ensure_ascii=False,indent=2))}</pre></section><section><h2>训练集回放与资源</h2><p>自由生成反映实际表现；teacher forcing提供标准前文。F的最终动作在标准字段条件下预测，不能替代整条输出评测。两组目标长度不同，整体loss不直接排名。</p><pre>{html.escape(json.dumps(dict(replay=data['replay'],training=data['training'],verdict=v),ensure_ascii=False,indent=2))}</pre></section><section><h2>同题改善与退步</h2><pre>{html.escape(json.dumps(data['examples'][:12],ensure_ascii=False,indent=2))}</pre></section><p>单种子开发实验；封存测试未使用。</p><p><a href="/">实时进度</a></p></html>'''
    (root/'results.html').write_text(page);(source/'reports/fields-sft-results.html').write_text(page)
