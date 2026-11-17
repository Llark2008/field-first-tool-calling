"""Source-backed aggregate report and a small standalone case-comparison page."""
import hashlib
import html
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

from .evaluate_formal import summarize

CONDITIONS={'complete':'完整任务','natural':'自然缺参','explicit':'显式缺参','unsupported':'不支持请求',
    'renamed_complete':'接口改名·完整','renamed_natural':'接口改名·自然缺参',
    'newtool_complete':'新工具·完整','newtool_natural':'新工具·自然缺参'}


def aligned(a,b):
    aa={r['case_id']:r for r in a}; bb={r['case_id']:r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b) or aa.keys()!=bb.keys(): raise ValueError('Comparison IDs do not match uniquely')
    return [(aa[k],bb[k]) for k in sorted(aa)]


def paired_interval(a,b,replicates=2000,seed=20261113,metric='success'):
    groups=defaultdict(list)
    def value(r):
        if metric=='premature_call': return int(r['gold']['action']=='clarify' and r['score'].get('action',{}).get('action')=='call')
        return int(r['score'][metric])
    for x,y in aligned(a,b):
        if x['template_group']!=y['template_group']: raise ValueError('Comparison family mismatch')
        groups[x['template_group']].append(value(y)-value(x))
    if not groups: return dict(n=0,families=0,delta_pp=None,ci95_pp=None)
    keys=sorted(groups); stats=[(sum(groups[k]),len(groups[k])) for k in keys]
    rng=random.Random(seed); draws=[]
    for _ in range(replicates):
        sample=[stats[rng.randrange(len(stats))] for _ in stats]
        draws.append(100*sum(v for v,n in sample)/sum(n for v,n in sample))
    draws.sort()
    return dict(n=len(a),families=len(keys),delta_pp=100*sum(v for v,n in stats)/len(a),
        ci95_pp=[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]],
        bootstrap='paired template-family cluster resampling',replicates=replicates,seed=seed)


def transition_counts(a,b):
    counts={k:0 for k in ['wrong→right','right→wrong','wrong→wrong','right→right']}
    for x,y in aligned(a,b): counts[('right' if x['score']['success'] else 'wrong')+'→'+('right' if y['score']['success'] else 'wrong')]+=1
    return counts


def curated_cases(cases,outputs):
    by_id={c['id']:c for c in cases}; chosen={}
    strata=defaultdict(list)
    for c in cases: strata[(c['condition'],c.get('target_tool'))].append(c)
    for group in strata.values():
        # Hash-based fixed stratification is independent of model outcomes.
        for c in sorted(group,key=lambda c:hashlib.sha256(c['id'].encode()).hexdigest())[:1]:
            chosen[c['id']]='固定分层样本（按 case ID 哈希选择）'
    if 'base_v2' in outputs and 'B_v2' in outputs:
        transitions=defaultdict(list)
        for a,b in aligned(outputs['base_v2'],outputs['B_v2']):
            key=('right' if a['score']['success'] else 'wrong')+'→'+('right' if b['score']['success'] else 'wrong')
            transitions[key].append(a['case_id'])
        for key,ids in transitions.items():
            picked=min(ids,key=lambda i:hashlib.sha256(i.encode()).hexdigest())
            chosen[picked]=chosen.get(picked,'')+'；转变类型补充：'+key
    maps={name:{r['case_id']:r for r in rows} for name,rows in outputs.items()}
    return [dict(id=i,condition=by_id[i]['condition'],missing_fields=by_id[i].get('missing_fields',[]),
        request=by_id[i]['request'],context=by_id[i].get('context',''),tools=by_id[i]['tools'],gold=by_id[i]['gold'],
        selection_reason=reason,models={name:rows[i] for name,rows in maps.items()}) for i,reason in sorted(chosen.items())]


def percent(entry,key='success'):
    return '未运行' if entry is None else f"{entry[key]}/{entry['n']} ({100*entry[key]/entry['n']:.1f}%)"


def load_rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]


def transfer_results(outputs):
    tools=sorted({r['target_tool'] for rows in outputs.values() for r in rows if r['condition'].startswith('newtool_')})
    result={}
    for tool in tools:
        rows={name:[r for r in values if r.get('target_tool')==tool and r['condition'].startswith('newtool_')]
              for name,values in outputs.items()}
        result[tool]=dict(metrics={name:summarize(values) for name,values in rows.items()},comparisons={})
        for comparator in ['base_v2','A_v2']:
            result[tool]['comparisons']['B_vs_'+comparator]={condition:paired_interval(
                [r for r in rows[comparator] if r['condition']==condition],
                [r for r in rows['B_v2'] if r['condition']==condition]) for condition in sorted({r['condition'] for r in rows['B_v2']})}
    return result


def clarification_field_errors(rows):
    """Whole-action conditional accuracy plus omission/extra-field counts.

    Field omissions/extras are counted only for structurally valid, nontruncated
    clarify actions. Non-clarify and invalid responses stay separate failures.
    """
    missing=Counter(); extra=Counter(); eligible=0; invalid=0; non_clarify=0
    by_field={}
    cases=[r for r in rows if r['gold']['action']=='clarify']
    for r in cases:
        s=r['score']; action=s.get('action',{})
        if not s.get('schema_valid') or s.get('truncated'):
            invalid+=1; continue
        if action.get('action')!='clarify': non_clarify+=1; continue
        eligible+=1
        gold=set(r['gold']['missing_fields']); predicted=set(action['missing_fields'])
        missing.update(gold-predicted); extra.update(predicted-gold)
    for f in sorted({f for r in cases for f in r.get('missing_fields',r['gold'].get('missing_fields',[]))}):
        subset=[r for r in cases if f in r.get('missing_fields',r['gold'].get('missing_fields',[]))]
        by_field[f]=dict(n=len(subset),exact_whole_action_correct=sum(r['score']['success'] for r in subset),
            omitted_in_valid_clarify=missing[f])
    return dict(n_missing_inputs=len(cases),valid_clarify_outputs=eligible,invalid_outputs=invalid,
        valid_nonclarify_outputs=non_clarify,conditional_whole_action_accuracy=by_field,
        missing_required_field_counts=dict(missing),extra_field_counts=dict(extra),
        note='Field-error counts only among valid clarify outputs; multiple fields can occur in one output. Conditional accuracy still requires the entire action to be correct.')


def make_report(run_root,dataset,selection,source_root,mode):
    root=Path(run_root); dataset=Path(dataset); source=Path(source_root)
    final=mode=='final'
    names=['base_v2','base_no_examples','base_checklist','A_v2','B_v2'] if final else ['base_v2','A_v2','B_v2']
    chosen=selection['selected_step'] if final else 2400
    paths={name:root/'eval'/('final-'+name+'.jsonl') for name in names} if final else {
        'base_v2':root/'eval/dev-base_v2.jsonl','A_v2':root/f'eval/dev-A-{chosen}.jsonl','B_v2':root/f'eval/dev-B-{chosen}.jsonl'}
    outputs={name:load_rows(path) for name,path in paths.items()}
    cases=load_rows(dataset/('test.jsonl' if final else 'dev.jsonl'))
    expected=480 if final else 300
    if len(cases)!=expected: raise ValueError('Wrong report case count')
    case_ids={c['id'] for c in cases}
    for rows in outputs.values():
        if len(rows)!=expected or {r['case_id'] for r in rows}!=case_ids: raise ValueError('Incomplete report inputs')
    metrics={name:summarize(rows) for name,rows in outputs.items()}
    comparisons={}
    for comparator in ['base_v2','A_v2']:
        comparisons['B_vs_'+comparator]={condition:paired_interval(
            [r for r in outputs[comparator] if r['condition']==condition],
            [r for r in outputs['B_v2'] if r['condition']==condition]) for condition in metrics['B_v2']}
    comparisons['B_vs_base_premature']=paired_interval([r for r in outputs['base_v2'] if r['condition']=='natural'],
        [r for r in outputs['B_v2'] if r['condition']=='natural'],metric='premature_call')
    transitions={name:transition_counts(outputs['base_v2'],outputs[name]) for name in ['A_v2','B_v2']}
    by_field={name:clarification_field_errors(rows) for name,rows in outputs.items()}
    transfer=transfer_results(outputs)
    alias_effect={}
    if final:
        for name,rows in outputs.items():
            byid={r['case_id']:r for r in rows}; alias_effect[name]={}
            for condition in ['renamed_complete','renamed_natural']:
                renamed=[r for r in rows if r['condition']==condition]
                base=[dict(byid[r['case_id'].removesuffix('-alias')],case_id=r['case_id']) for r in renamed]
                alias_effect[name][condition]=paired_interval(base,renamed)
    curves=[]
    baseline=json.loads((root/'eval/dev-base_v2.metrics.json').read_text())
    curves.append(dict(name='base_v2',step=0,metrics=baseline))
    for arm in ['A','B']:
        for step in [600,1200,2400]:
            p=root/f'eval/dev-{arm}-{step}.metrics.json'
            if p.exists(): curves.append(dict(name=arm,step=step,metrics=json.loads(p.read_text())))
    loss={a:load_rows(root/f'train-{a}/events.jsonl') for a in ['A','B']}
    costs={a:json.loads((root/f'train-{a}/summary.json').read_text()) for a in ['A','B']}
    audit=json.loads((dataset/'manifest.json').read_text())['audit']
    resources=json.loads((root/'resource-summary.json').read_text()) if (root/'resource-summary.json').exists() else None
    natural_gain=comparisons['B_vs_base_v2']['natural']['delta_pp']; complete_gain=comparisons['B_vs_base_v2']['complete']['delta_pp']
    ba_natural=comparisons['B_vs_A_v2']['natural']['delta_pp']; ba_complete=comparisons['B_vs_A_v2']['complete']['delta_pp']
    targets=dict(B_over_base=natural_gain>=20 and comparisons['B_vs_base_premature']['delta_pp']<=-20 and complete_gain>=-5,
                 B_over_A=ba_natural>=10 and ba_complete>=-5)
    payload=dict(mode=mode,selection=selection,metrics=metrics,comparisons=comparisons,transitions=transitions,
        per_field=by_field,transfer=transfer,alias_effect=alias_effect,curves=curves,loss=loss,costs=costs,resources=resources,
        audit=audit,practical_targets=targets,cases=curated_cases(cases,outputs),
        n_inputs=expected,n_intents=len({c['intent_group'] for c in cases}),n_families=len({c['template_group'] for c in cases}),
        provenance=dict(run_root=str(root),dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
            frozen_manifest_sha256=hashlib.sha256((root/'frozen.json').read_bytes()).hexdigest(),
            output_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}))
    (root/'report-data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    title='正式测试结果' if final else '开发集结果：保留条件未通过，最终测试未打开'
    lines=['# 工具调用与缺参澄清：'+title,'',
        f"模型：Qwen3-1.7B 4-bit；A 显式缺参 SFT，B 自然缺参 SFT；单次训练种子 20261113。",
        f"评测：{expected} 个输入，{payload['n_intents']} 个潜在任务、{payload['n_families']} 个模板族；相关变体不能视作独立样本。",
        f"共同检查点：{chosen} 微步 / {chosen//8} 次更新。" if final else '没有共同检查点满足两组完整任务成功率均不低于原模型 5 个百分点的条件。下表 A/B 为 2 epoch 开发诊断，未被选为最终模型。',
        '', '| 条件 | '+' | '.join(names)+' |','|---|'+'---|'*len(names)]
    for condition in CONDITIONS:
        if condition in metrics['base_v2']:
            lines.append('| '+CONDITIONS[condition]+' | '+' | '.join(percent(metrics[n].get(condition)) for n in names)+' |')
    lines += ['', '## 两个具体新工具分别评测', '', '| 工具 | 条件 | '+' | '.join(names)+' |','|---|---|'+'---|'*len(names)]
    for name,entry in transfer.items():
        for condition in ['newtool_complete','newtool_natural']:
            lines.append('| '+name+' | '+CONDITIONS[condition]+' | '+' | '.join(percent(entry['metrics'][n].get(condition)) for n in names)+' |')
    lines += ['', '| 工具 | 比较 | 条件 | 变化 pp | 95% 区间 pp | 模板族数 |','|---|---|---|---:|---|---:|']
    for name,entry in transfer.items():
        for comparator,conditions in entry['comparisons'].items():
            for condition,c in conditions.items():
                lines.append(f"| {name} | {comparator} | {CONDITIONS[condition]} | {c['delta_pp']:+.1f} | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | {c['families']} |")
    lines += ['', '## 预先约定的实际目标', '',
        f"B 相对原模型：自然澄清变化 {natural_gain:+.1f}pp，擅自调用变化 {comparisons['B_vs_base_premature']['delta_pp']:+.1f}pp，完整任务变化 {complete_gain:+.1f}pp。目标判定：{targets['B_over_base']}。",
        f"B 相对 A：自然澄清变化 {ba_natural:+.1f}pp，完整任务变化 {ba_complete:+.1f}pp。目标判定：{targets['B_over_A']}。",
        '这些阈值是项目决策标准，不是统计显著性或非劣效性证明。开发集上的目标值只作诊断。',
        '', '## 配对不确定性', '', '| 对比 | 条件 | 变化 pp | 95% 区间 pp | 模板族数 |','|---|---|---:|---|---:|']
    for comp in ['B_vs_base_v2','B_vs_A_v2']:
        for condition,c in comparisons[comp].items():
            lines.append(f"| {comp} | {CONDITIONS[condition]} | {c['delta_pp']:+.1f} | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | {c['families']} |")
    c=comparisons['B_vs_base_premature']
    lines.append(f"| B_vs_base_v2 | 自然缺参擅自调用（越低越好） | {c['delta_pp']:+.1f} | [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}] | {c['families']} |")
    lines += ['', '按模板族配对 bootstrap（2,000 次），同族完整/缺参/改名输入共同抽样。各条件分别展示；区间仅反映本合成任务集，不覆盖训练种子变化。',
        '', '## 错误与来源校验', '', '| 配置 | 完整任务误追问 | 完整任务错误拒绝 | 完整任务格式错误 | 自然缺参擅自调用 | 自然缺参追问字段错误 | 自然缺参格式错误 | 校验后仍放行擅自调用 | 校验误拦完整正确调用 |',
        '|---|---|---|---|---|---|---|---|---|']
    for name in names:
        m=metrics[name]; lines.append('| '+name+' | '+' | '.join([
            percent(m['complete'],'false_clarify'),percent(m['complete'],'false_unsupported'),percent(m['complete'],'invalid'),percent(m['natural'],'premature_call'),
            percent(m['natural'],'wrong_clarification_fields'),percent(m['natural'],'invalid'),percent(m['natural'],'allowed_premature_calls'),percent(m['complete'],'blocked_correct_calls')])+' |')
    lines += ['', '| 配置 | 条件 | 原始成功 | 校验后成功 | 正确追问（拦截不计入） |','|---|---|---|---|---|']
    for name in names:
        for condition,m in metrics[name].items():
            lines.append(f"| {name} | {CONDITIONS[condition]} | {percent(m)} | {percent(m,'system_success')} | {percent(m,'correct_clarifications')} |")
    lines += ['', '### 澄清字段错误', '', '以下字段遗漏/额外字段计数只针对结构合法、未截断的 clarify 输出；无效和非 clarify 输出另列。一个输出可涉及多个字段。条件成功率要求整条动作正确，不是单字段准确率。', '']
    for name,entry in by_field.items():
        lines += [f"- {name}：缺参输入 {entry['n_missing_inputs']}，合法 clarify {entry['valid_clarify_outputs']}，无效输出 {entry['invalid_outputs']}，合法非 clarify {entry['valid_nonclarify_outputs']}。",
            f"  缺少必需澄清字段：{json.dumps(entry['missing_required_field_counts'])}；多余字段：{json.dumps(entry['extra_field_counts'])}。",
            f"  按必需字段分层的整条动作成功计数：{json.dumps(entry['conditional_whole_action_accuracy'])}。"]
    lines += ['', '来源校验仅覆盖显式项目、环境、算法、run ID；不覆盖 metric/stat/order/limit。拦截不计为正确追问。无效 JSON、截断、字段错误也不得因减少调用而算成功。',
        '', '## 成本与限制', '']
    for arm,cost in costs.items(): lines.append(f"- {arm}：{cost['microsteps']} 微步、{cost['optimizer_updates']} 更新；活跃训练 {cost['active_training_seconds']/60:.1f} 分钟，MLX 峰值 {cost['mlx_peak_gb']:.2f} GB。")
    lines += ['- 训练数据每组 1,200 条；完整/缺参/不支持为 600/480/120。A/B 完整与不支持样本、目标输出和曝光次序相同，仅缺参提示措辞不同。',
        '- 指标×排序四组合各 25 条；项目清单位置分布 136/132/132；234 条完整任务覆盖可选项、默认项或隐含但明确的参数。',
        '- 数据为 AI 辅助编写和审计的合成任务。虽按模板族分割，底层槽位语法、领域和工具描述仍共享；不能宣称通用真实场景能力。没有人工专家验证。',
        '- 单模型、单训练种子；新工具只有两个具体接口。旧试跑不是本轮受控基线。',
        '- HTML 使用固定分层样本，并补充可获得的四种对比转变；补充样本标注选择原因，所有转变的总数另行展示。',
        '', '## 复现与原始记录','',f"运行目录：`{root}`",f"数据目录：`{dataset}`",'',
        '命令与固定配置见 README；`frozen.json` 保存数据、模型、源代码与环境版本指纹，`eval/` 保存全部原始输出，`train-A/`、`train-B/` 保存检查点与优化器状态。',
        '本报告不会自动触发再次调参。打开最终测试后，如要继续优化，必须使用新的开发迭代与新的最终测试。']
    markdown='\n'.join(lines)+'\n'
    page=PAGE.replace('__TITLE__',html.escape(title)).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (root/'results.html').write_text(page)
    reports=source/'reports'; reports.mkdir(exist_ok=True)
    (reports/'formal-clarification-results.md').write_text(markdown)
    (reports/'formal-clarification-results.html').write_text(page)
    # Public metrics omit raw local paths and the embedded case subset/loss events.
    public={k:v for k,v in payload.items() if k not in ('cases','loss','provenance')}
    (reports/'formal-clarification-metrics.json').write_text(json.dumps(public,ensure_ascii=False,indent=2))
    return payload


PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>工具调用实验 · 结果对比</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#20333b;background:#f4f6f5}body{max-width:1240px;margin:36px auto;padding:0 24px}h1{font-size:30px}h2{font-size:21px;margin-top:30px}p{line-height:1.75}.muted{color:#62737a;font-size:14px}.card{background:white;border:1px solid #dce5e1;border-radius:12px;padding:20px;margin:16px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border-bottom:1px solid #e1e7e4;text-align:left;white-space:nowrap}th{color:#35585a}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.good{color:#167356}.bad{color:#ad4c39}.badge{padding:4px 8px;background:#e9f1ed;border-radius:6px;font-size:13px}select{padding:10px;max-width:100%;border:1px solid #b9cbc3;border-radius:7px;background:white;margin:5px}.notice{border-left:4px solid #308775;padding:12px 18px;background:#e7f1ec}svg{width:100%;height:auto}.model{padding:12px;border:1px solid #dce5e1;border-radius:8px}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}button{padding:10px 16px;border:0;border-radius:6px;background:#286e60;color:white;cursor:pointer}</style>
<style>@media(max-width:900px){body{padding:0 14px}.card{padding:12px}#summary th,#summary td{padding:10px 4px;white-space:normal;overflow-wrap:anywhere;font-size:12px}}</style>
<h1>工具调用与缺参澄清</h1><p class="muted">Qwen3-1.7B · A 显式缺参微调 / B 自然缺参微调 · 单次训练种子</p>
<div class="notice"><b>__TITLE__</b><p id="lead"></p></div>
<h2>逐条件比较</h2><div class="card" id="summary"></div>
<p class="muted">每格为成功数 / 输入数（比例）。正确调用同时要求参数与实际执行结果正确；追问要求缺失字段完全匹配。较宽的明细表可横向滚动。</p>
<h2>两个具体新工具分别评测</h2><div class="card" id="transfer"></div><div class="card" id="transferCI"></div><p class="muted">逐工具、逐条件列出分母和配对区间；只能解读为这些具体接口的迁移检查。</p>
<h2>开发集上的能力变化</h2><p class="muted">横轴是优化器更新次数。三处检查点对应 0.5 / 1 / 2 epoch；这些曲线用于模型选择，不能作为最终测试成绩。</p><div id="curves" class="grid"></div>
<h2>配对变化与不确定性</h2><div class="card" id="intervals"></div>
<p class="muted">按模板族重采样 2,000 次；区间仅反映此合成任务集合，不包含随机训练种子的变化。5pp 能力保留线并非非劣效性检验。</p>
<h2>错误与校验</h2><div class="card" id="errors"></div><p class="muted">来源校验的拦截是系统弃权，不是模型正确追问。缺参时因无效输出而没有调用，仍然是失败。</p>
<details class="card"><summary>原始与来源校验后成功率</summary><div id="systemSuccess"></div></details>
<h2>澄清字段遗漏与额外字段</h2><div class="card" id="fieldErrors"></div><p class="muted">遗漏/额外字段仅统计合法、未截断的 clarify 输出；一个输出可能有多个字段错误。无效与非 clarify 输出另列。下方明细的“按字段条件成功率”要求整条动作正确，并非单字段准确率。</p>
<div class="grid"><section class="card"><b>四种结果转变（相对原模型）</b><pre id="transitions"></pre></section><section class="card"><b>训练损失（辅助指标）</b><div id="loss"></div></section></div>
<h2>具体样本</h2><p class="muted" id="sampleNote"></p><div><select id="condition" aria-label="输入条件"><option value="">全部条件</option></select><select id="field" aria-label="缺失字段"><option value="">全部缺失字段</option></select><select id="failure" aria-label="错误类型"><option value="">全部结果类型</option></select></div><div id="cases"></div>
<h2>审计、资源与边界</h2><div class="card"><p>AI 辅助构造和审计；合成任务按模板族分割，但共享底层槽位语法。没有人工专家验证；单模型、单训练种子、两个特定新工具，不能推广为通用能力结论。</p><details><summary>数据与资源明细</summary><pre id="audit"></pre></details><details><summary>按缺失字段的结果（字段可重叠）</summary><pre id="fields"></pre></details><details><summary>接口改名前后的配对变化</summary><pre id="alias"></pre></details><details><summary>来源指纹</summary><pre id="provenance"></pre></details></div>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);const C={complete:'完整任务',natural:'自然缺参',explicit:'显式缺参',unsupported:'不支持请求',renamed_complete:'改名·完整',renamed_natural:'改名·缺参',newtool_complete:'新工具·完整',newtool_natural:'新工具·缺参'};
const F={correct:'正确',invalid:'格式错误 / 截断',premature_call:'缺参擅自调用',false_clarify:'完整任务误追问',false_unsupported:'错误拒绝',wrong_clarification_fields:'追问字段错误',wrong_call:'工具或参数错误',unsupported_missed:'未识别不支持请求'};
const N=Object.keys(D.metrics),esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function rate(e,k='success'){return e?`${e[k]}/${e.n} (${(100*e[k]/e.n).toFixed(1)}%)`:'未运行'}
$('lead').textContent=`${D.n_inputs} 个输入，${D.n_intents} 个潜在任务、${D.n_families} 个模板族。`+(D.mode==='final'?`共同选择 ${D.selection.selected_step/8} 次更新的 A/B 检查点；最终数据仅评测一次。`:'未找到满足完整任务能力保留条件的共同检查点；最终测试仍封存。下方 A/B 样本采用 2 epoch 检查点，仅用于开发诊断。');
$('summary').innerHTML=table(['条件',...N],Object.keys(C).filter(c=>D.metrics[N[0]][c]).map(c=>[C[c],...N.map(n=>rate(D.metrics[n][c]))]));
let transferRows=[],transferCI=[];for(let [tool,entry] of Object.entries(D.transfer)){for(let c of ['newtool_complete','newtool_natural'])transferRows.push([tool,C[c],...N.map(n=>rate(entry.metrics[n][c]))]);for(let [comp,conditions] of Object.entries(entry.comparisons))for(let [c,v] of Object.entries(conditions))transferCI.push([tool,comp,C[c],v.delta_pp.toFixed(1)+' pp',v.ci95_pp.map(x=>x.toFixed(1)).join(' ～ ')+' pp',v.families])}$('transfer').innerHTML=table(['工具','条件',...N],transferRows);$('transferCI').innerHTML=table(['工具','比较','条件','变化','95% 区间','模板族'],transferCI);
let intervals=[];for(let comp of ['B_vs_base_v2','B_vs_A_v2'])for(let [c,v] of Object.entries(D.comparisons[comp]))intervals.push([comp,C[c],v.delta_pp.toFixed(1)+' pp',v.ci95_pp.map(x=>x.toFixed(1)).join(' ～ ')+' pp',v.families]);let premature=D.comparisons.B_vs_base_premature;intervals.push(['B_vs_base_v2','擅自调用变化（越低越好）',premature.delta_pp.toFixed(1)+' pp',premature.ci95_pp.map(x=>x.toFixed(1)).join(' ～ ')+' pp',premature.families]);$('intervals').innerHTML=table(['比较','条件','变化','95% 区间','模板族'],intervals);
$('errors').innerHTML=table(['配置','完整误追问','完整错误拒绝','完整格式错','自然缺参擅自调用','自然缺参追问字段错误','自然缺参格式错','校验后仍放行','误拦正确调用'],N.map(n=>{let m=D.metrics[n];return[n,rate(m.complete,'false_clarify'),rate(m.complete,'false_unsupported'),rate(m.complete,'invalid'),rate(m.natural,'premature_call'),rate(m.natural,'wrong_clarification_fields'),rate(m.natural,'invalid'),rate(m.natural,'allowed_premature_calls'),rate(m.complete,'blocked_correct_calls')]}));
let systemRows=[];for(let n of N)for(let [c,m] of Object.entries(D.metrics[n]))systemRows.push([n,C[c],rate(m),rate(m,'system_success'),rate(m,'correct_clarifications')]);$('systemSuccess').innerHTML=table(['配置','条件','原始成功','校验后成功','正确追问（拦截不计）'],systemRows);
$('fieldErrors').innerHTML=table(['配置','缺参输入数','合法 clarify','无效输出','合法非 clarify','遗漏必需字段','额外字段'],N.map(n=>{let f=D.per_field[n];return[n,f.n_missing_inputs,f.valid_clarify_outputs,f.invalid_outputs,f.valid_nonclarify_outputs,JSON.stringify(f.missing_required_field_counts),JSON.stringify(f.extra_field_counts)]}));
function plot(series,title,percent=true){let colors={base_v2:'#8b9298',A:'#4d71bb',B:'#238c70'},w=510,h=230,max=percent?1:Math.max(...series.flatMap(s=>s.points.map(p=>p.y)),.01);let out=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">`;for(let i=0;i<5;i++){let y=180-i*37.5;out+=`<line x1="42" y1="${y}" x2="485" y2="${y}" stroke="#e1e8e4"/><text x="3" y="${y+4}" font-size="11" fill="#65766e">${(max*i/4*(percent?100:1)).toFixed(percent?0:2)}</text>`}for(let x of [0,75,150,300])out+=`<text x="${42+x/300*443}" y="202" font-size="11">${x}</text>`;for(let s of series){let points=s.points.map(p=>`${42+p.x/300*443},${180-p.y/max*150}`).join(' ');out+=`<polyline fill="none" stroke="${colors[s.name]}" stroke-width="2" points="${points}"/>`;for(let p of s.points)out+=`<circle cx="${42+p.x/300*443}" cy="${180-p.y/max*150}" r="2.5" fill="${colors[s.name]}"/>`}out+='</svg>';return '<b>'+esc(title)+'</b>'+out+'<div class="muted">灰：原模型　蓝：A　绿：B</div>'}
for(let [title,c,k] of [['完整任务成功率','complete','success'],['自然缺参精确追问率','natural','success'],['完整任务误追问率','complete','false_clarify'],['完整任务无效输出率','complete','invalid']]){let series=['base_v2','A','B'].map(name=>({name,points:D.curves.filter(v=>v.name===name).map(v=>({x:v.step/8,y:v.metrics[c][k]/v.metrics[c].n}))}));series[0].points.push({...series[0].points[0],x:300});let card=document.createElement('section');card.className='card';card.innerHTML=plot(series,title);$('curves').append(card)}
$('loss').innerHTML=plot(['A','B'].map(name=>({name,points:D.loss[name].filter((_,i)=>i%4===0||i===D.loss[name].length-1).map(v=>({x:v.optimizer_updates,y:v.loss}))})),'训练损失',false);
for(let id of ['transitions','provenance'])$(id).textContent=JSON.stringify(D[id],null,2);$('audit').textContent=JSON.stringify({data:D.audit,costs:D.costs,resources:D.resources,selection:D.selection,practical_targets:D.practical_targets},null,2);$('fields').textContent=JSON.stringify(D.per_field,null,2);$('alias').textContent=D.mode==='final'?JSON.stringify(D.alias_effect,null,2):'最终测试未打开。';
$('sampleNote').textContent=`此页嵌入 ${D.cases.length} 条审核样本，来自固定分层选择，并按需要补充四种转变；选择原因逐条标注。全部 ${D.n_inputs} 条原始输出保存在本机运行目录。`;
function option(id,v,label){let o=document.createElement('option');o.value=v;o.textContent=label;$(id).append(o)}for(let c of [...new Set(D.cases.map(c=>c.condition))])option('condition',c,C[c]);for(let f of [...new Set(D.cases.flatMap(c=>c.missing_fields))].sort())option('field',f,f);for(let f of Object.keys(F))option('failure',f,F[f]);
function render(){let items=D.cases.filter(c=>(!$('condition').value||c.condition===$('condition').value)&&(!$('field').value||c.missing_fields.includes($('field').value))&&(!$('failure').value||Object.values(c.models).some(m=>m.failure_type===$('failure').value)));$('cases').replaceChildren();for(let c of items){let el=document.createElement('section');el.className='card';el.innerHTML=`<span class="badge">${esc(C[c.condition])}</span> <code>${esc(c.id)}</code><p>${esc(c.request)}</p><p class="muted">${esc(c.selection_reason)}</p><details><summary>公开上下文与工具定义</summary><pre>${esc(c.context)}\n${esc(JSON.stringify(c.tools,null,2))}</pre></details><p><b>正确动作</b></p><pre>${esc(JSON.stringify(c.gold))}</pre><div class="grid">`+Object.entries(c.models).map(([n,m])=>`<div class="model"><b>${esc(n)}</b> <span class="${m.score.success?'good':'bad'}">${esc(F[m.failure_type]||m.failure_type)}</span><pre>${esc(m.output)}</pre><div class="muted">来源校验：${m.guard?.blocked?'拦截（不计正确追问）':'未拦截'} · ${m.seconds?.toFixed(2)||'—'} 秒</div><details><summary>完整评分</summary><pre>${esc(JSON.stringify(m.score,null,2))}</pre></details></div>`).join('')+'</div>';$('cases').append(el)}if(!items.length)$('cases').textContent='当前筛选没有审核样本；完整分布请查汇总数据。'}for(let id of ['condition','field','failure'])$(id).addEventListener('change',render);render();
</script></html>'''
