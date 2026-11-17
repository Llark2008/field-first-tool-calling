"""Measured comparison artifacts with raw-output verification and fixed examples."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .ablation_data import read_rows
from .analyze_clarification_ablation import regression_decision,historical_complete_counts
from .controls import source_guard
from .evaluate_formal import summarize,failure_type
from .formal_data import sha
from .generation_scoring import score_generation


def verify_outputs(root,frozen,cases,outputs,refs,partial=False):
    from .run_clarification_ablation import eval_meta,validate_evaluation,adapter_for
    checked=[]
    def verify_rows(cs,rows,path):
        cmap={c['id']:c for c in cs}
        for row in rows:
            c=cmap[row['case_id']];score=score_generation(c,row['output'],row['finish_reason'])
            if score!=row['score']:raise ValueError('Rescoring mismatch: '+row['case_id'])
            valid=score['schema_valid'] and not score['truncated'] and score.get('action',{}).get('action')=='call'
            guard=source_guard(c['request'],c['tools'],score['action']) if valid else dict(blocked=False,ungrounded_fields=[])
            if guard!=row['guard'] or failure_type(row)!=row['failure_type']:raise ValueError('Guard/failure mismatch')
        if summarize(rows)!=json.loads(path.with_suffix('.metrics.json').read_text()):raise ValueError('Saved aggregate mismatch')
        checked.append(dict(file=path.name,rows=len(rows),sha256=sha(path),generation_seconds=sum(r['seconds'] for r in rows)))
    for name,rows in refs.items():verify_rows(cases,rows,root/'eval'/f'dev-{name}.jsonl')
    for (arm,step),rows in outputs.items():verify_rows(cases,rows,root/'eval'/f'dev-{arm}-{step}.jsonl')
    regression=json.loads((root/'regression.json').read_text()) if (root/'regression.json').exists() and not partial else {'status':'not_run'}
    if regression['status']=='evaluated':
        from .expanded import expanded_cases
        cs=expanded_cases();step=regression['microstep']
        for arm in [regression['candidate'],'R0']:
            name=f'{arm}-{step}';path=root/'eval'/f'historical-{name}.jsonl';adapter=adapter_for(root,frozen,arm,step)
            rows=validate_evaluation(path,cs,eval_meta(frozen,name,'historical','v2',adapter,cs));verify_rows(cs,rows,path)
            if summarize(rows)!=regression['metrics'][arm]:raise ValueError('Historical aggregate mismatch')
        counts=historical_complete_counts(regression['metrics'][regression['candidate']])
        if regression_decision(counts)!=regression['decision']:raise ValueError('Historical decision mismatch')
    return dict(status='passed' if not partial else 'passed_completed_configurations',evaluations=checked,total_generations=sum(r['rows'] for r in checked),
        total_generation_seconds=sum(r['generation_seconds'] for r in checked),checks=['Aligned IDs and frozen metadata checked before analysis',
        'Every raw generation rescored; guard, failure type and saved aggregate reproduced','Historical decision reproduced when applicable'])


def examples(cases,outputs):
    groups=defaultdict(list);chosen={};maps={name:{r['case_id']:r for r in rows} for name,rows in outputs.items()}
    for c in cases:groups[(c['condition'],c['target_tool'])].append(c)
    for values in groups.values():
        c=min(values,key=lambda c:hashlib.sha256(c['id'].encode()).hexdigest());chosen[c['id']]='固定分层样本，按 ID 哈希选择'
    for name in ['R1','R2','R3']:
        if 'R0' not in maps or name not in maps:continue
        trans=defaultdict(list)
        for c in cases:
            a=maps['R0'][c['id']]['score']['success'];b=maps[name][c['id']]['score']['success']
            if a!=b:trans['修复' if b else '退化'].append(c)
        for kind,values in trans.items():
            c=min(values,key=lambda c:hashlib.sha256(c['id'].encode()).hexdigest())
            chosen[c['id']]=chosen.get(c['id'],'')+f'；{name} 相对 R0 的{kind}案例（结果驱动补充）'
    cm={c['id']:c for c in cases}
    return [dict(id=i,request=cm[i]['request'],condition=cm[i]['condition'],target_tool=cm[i]['target_tool'],gold=cm[i]['gold'],
        reason=reason,models={n:dict(output=m[i]['output'],success=m[i]['score']['success'],failure=m[i]['failure_type']) for n,m in maps.items()}) for i,reason in sorted(chosen.items())]


def render_html(payload):
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return HTML.replace('__PAYLOAD__',data)


def write_pilot_report(analysis:dict,run_root:Path,source_root:Path,loaded=None)->dict:
    from .run_clarification_ablation import load_all
    root=Path(run_root);source=Path(source_root);frozen=json.loads((root/'frozen.json').read_text())
    cases,outputs,refs=loaded if loaded is not None else load_all(root,frozen)
    payload=dict(analysis,cases=examples(cases,{a:outputs[a,600] for a in ['R0','R1','R2','R3'] if (a,600) in outputs}),
        token_audit=frozen['token_audit'],dataset=frozen['dataset'],run_root=str(root))
    reports=source/'reports';reports.mkdir(exist_ok=True)
    text=json.dumps(payload,ensure_ascii=False,indent=2)
    (root/'report-data.json').write_text(text);(reports/'clarification-ablation-metrics.json').write_text(text)
    html=render_html(payload);(root/'results.html').write_text(html);(reports/'clarification-ablation-results.html').write_text(html)
    def fmt(m):return f"{m['count']}/{m['n']} ({m['rate']*100:.1f}%)"
    lines=['# 学习率 × 配对数据：开发实验结果','',f"状态：{payload['status']}。",'',f"实验指纹：`{payload['fingerprint']}`",'',
        '单随机种子、合成任务、开发集筛选；本轮没有新的最终测试。主因素比较固定在 75 次更新。','',
        '| 配置 | 完整 C | 缺参 Q | 误追问 F | 擅自调用 P | 不支持 U | 配对 J | 能力门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for c in payload['candidates']:
        if c['updates']!=75:continue
        lines.append('| '+c['arm']+' | '+' | '.join(fmt(c['metrics'][k]) for k in ['C','Q','F','P','U','J'])+' | '+('通过' if c['gate']['eligible'] else ', '.join(c['gate']['reasons']))+' |')
    lines+=['','## 同更新数的因素差异','', '| 对比 | C 差异 pp | Q 差异 pp | F 差异 pp | J 差异 pp |','|---|---:|---:|---:|---:|']
    for name,e in payload['effects'].items():lines.append('| '+name+' | '+' | '.join(f"{e[k]['delta_pp']:+.1f} [{e[k]['ci95_pp'][0]:+.1f}, {e[k]['ci95_pp'][1]:+.1f}]" for k in ['C','Q','F','J'])+' |')
    selected=payload['selection']['selected']
    lines+=['','区间为按表达家族配对重采样 2,000 次；按工具与家族大小分层，保持原始分母。不能覆盖训练种子的变化。','',
        '## 候选选择与回归','',f"候选：{selected['arm']+' / '+str(selected['updates'])+' 次更新' if selected else '无合格候选'}。",'',
        '```json',json.dumps(payload['regression'],ensure_ascii=False,indent=2),'```','',
        '## 计算与验证','',f"已完成新增训练：{sum(x['active_training_seconds'] for x in payload['training'].values())/60:.1f} 分钟（不含复现核验与准备）。",
        f"评测 {payload['verification']['total_generations']} 次生成，累计 {payload['verification']['total_generation_seconds']/60:.1f} 分钟。",
        '原始输出已全部重新评分并核对聚合；完整曲线、固定分层案例及结果驱动的修复/退化补充案例见 HTML。','',
        'R0 检查点为通过数值等价核验的历史对照；不以新旧运行耗时推断方法速度。']
    (reports/'clarification-ablation-results.md').write_text('\n'.join(lines)+'\n')
    verification=dict(payload['verification'],fingerprint=payload['fingerprint'],report_data_sha256=sha(root/'report-data.json'),html_sha256=sha(root/'results.html'))
    (reports/'clarification-ablation-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    return payload


HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>学习率 × 配对数据 · 实验结果</title>
<style>:root{color-scheme:dark;font-family:system-ui;background:#101b24;color:#e8edf3}body{max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.6}h1{font-size:28px}.muted{color:#9fb1c0}.card{background:#1b2b37;border:1px solid #344959;border-radius:14px;padding:20px;margin:18px 0;min-width:0}.row{display:flex;flex-wrap:wrap;gap:14px}.row>.card{flex:1;min-width:240px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;text-align:left;border-bottom:1px solid #344959;white-space:nowrap}.table{overflow:auto}select,button{background:#243f4c;color:#f1f6f9;border:1px solid #56717c;border-radius:7px;padding:8px;margin:4px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}svg{width:100%;height:auto}a{color:#98e7ca}.ok{color:#8de7b7}.bad{color:#ffba91}.case{border-top:1px solid #344959;padding:15px 0}.pill{display:inline-block;padding:3px 10px;border-radius:16px;background:#304e45;color:#b1efcf}details{margin:10px 0}summary{cursor:pointer}@media(max-width:600px){body{padding:0 12px}.card{padding:14px}.row>.card{min-width:0;flex-basis:100%}h1{font-size:24px}}</style>
<h1>学习率 × 配对数据：实验结果</h1><p class="muted">开发筛选 · 单随机种子 · 原模型 Qwen3-1.7B 4-bit · 主比较固定在 75 次更新</p><p><a href="/">实时进度</a></p><p id="headline"></p>
<section class="card"><b>实验矩阵与固定参考</b><p>R0：原数据 / 1e-4；R1：原数据 / 5e-5；R2：改进配对数据 / 1e-4；R3：改进配对数据 / 5e-5。每组相同 1,200 条训练样本，D1 仅改写 180 对请求。</p><div class="table" id="refs"></div></section>
<section class="card"><b>四组同更新数比较</b><div><label>优化器更新数 <select id="step"><option>25</option><option>50</option><option selected>75</option><option>100</option><option>150</option></select></label></div><div class="table" id="grid"></div><p class="muted">C 完整成功 /60；Q 精确缺参追问 /48；F 完整任务误追问 /60（越低越好）；P 缺参擅自调用 /48（越低越好）；U 不支持请求正确 /12；J 配对双正确 /48。</p></section>
<section class="card"><b>75 次更新：因素差异与家族区间</b><div class="table" id="effects"></div><p class="muted">R1−R0：原数据下降低学习率；R2−R0：原学习率下改进数据。R3−R2 与 R3−R1 为另一条件下的同一因素。交互=(R3−R2)−(R1−R0)。区间按表达家族配对重采样，不代表跨种子稳定性。</p></section>
<section class="card"><b>训练预算与表现曲线</b><label>指标 <select id="metric"><option value="C">完整任务 C ↑</option><option value="Q">缺参识别 Q ↑</option><option value="F">误追问 F ↓</option><option value="J">配对双正确 J ↑</option></select></label><svg id="curve" viewBox="0 0 700 310" aria-label="四组学习曲线"></svg></section>
<section class="card"><b>可用性门槛与历史回归</b><p>C 至少90%，且最多比最强固定参考或同更新 R0 少3题；F最多3题；边界完整与不支持请求另设保留门槛。通过后按 Q、J、F、C、较早更新数排序。门槛是工程筛选，不是统计非劣效性证明。</p><pre id="selection"></pre><details><summary>全部检查点的门槛判定</summary><div class="table" id="gates"></div></details><details><summary>历史 288 题回归结果</summary><pre id="regression"></pre></details></section>
<section class="card"><b>逐工具与错误类型</b><label>配置 <select id="arm"><option>R0</option><option>R1</option><option>R2</option><option>R3</option></select></label><p class="muted">对应上方所选更新数。来源校验拦截不计为正确追问；原始与系统表现分别展示。</p><div class="table" id="tools"></div><details><summary>字段错误、边界任务与原始／系统统计</summary><pre id="errors"></pre></details></section>
<section class="card"><b>75 次更新：具体输出</b><label>任务 <select id="condition"><option value="all">全部</option><option value="complete">完整</option><option value="natural">缺参</option><option value="unsupported">不支持</option></select></label><div id="cases"></div></section>
<section class="card"><b>计算与审计</b><pre id="cost"></pre><details><summary>限制与数据指纹</summary><pre id="limits"></pre></details></section>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=x=>document.getElementById(x),keys=['C','Q','F','P','U','J'],colors={R0:'#adb8c6',R1:'#82ddc1',R2:'#e9ba74',R3:'#9bafff'};
function table(id,heads,rows){const t=document.createElement('table'),h=document.createElement('tr');for(const v of heads){const c=document.createElement('th');c.textContent=v;h.append(c)}t.append(h);for(const row of rows){const tr=document.createElement('tr');for(const v of row){const td=document.createElement('td');td.textContent=v;tr.append(td)}t.append(tr)}$(id).replaceChildren(t)}
const fmt=m=>m?`${m.count}/${m.n} (${(100*m.rate).toFixed(1)}%)`:'未运行';
function curve(){const svg=$('curve');svg.replaceChildren();const metric=$('metric').value,ns='http://www.w3.org/2000/svg';function add(tag,attrs,text){let e=document.createElementNS(ns,tag);for(let[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e)}for(let y=0;y<=100;y+=25){let p=260-y*2.1;add('line',{x1:55,y1:p,x2:660,y2:p,stroke:'#344959'});add('text',{x:15,y:p+4,fill:'#a1b2c1','font-size':12},y+'%')}for(let s of [25,50,75,100,150])add('text',{x:55+(s-25)/125*605,y:285,fill:'#a1b2c1','font-size':12},s);for(let[a,color]of Object.entries(colors)){let cs=D.candidates.filter(c=>c.arm===a).sort((a,b)=>a.updates-b.updates),points=cs.map(c=>[55+(c.updates-25)/125*605,260-c.metrics[metric].rate*210]);add('polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2.5});for(let [x,y]of points)add('circle',{cx:x,cy:y,r:4,fill:color});add('text',{x:70+Object.keys(colors).indexOf(a)*145,y:23,fill:color,'font-size':14},a)} }
function render(){const step=Number($('step').value),cs=D.candidates.filter(c=>c.updates===step);table('grid',['组别',...keys,'能力门槛'],['R0','R1','R2','R3'].map(a=>{const c=cs.find(c=>c.arm===a);return c?[c.arm,...keys.map(k=>fmt(c.metrics[k])),c.gate.eligible?'通过':c.gate.reasons.join(', ')]:[a,...keys.map(k=>'未运行'),'未运行']}));const c=cs.find(c=>c.arm===$('arm').value);if(c){table('tools',['工具','完整','缺参','完整误追问'],Object.entries(c.metrics.by_tool).map(([n,m])=>[n,`${m.complete?.success||0}/${m.complete?.n||0}`,`${m.natural?.success||0}/${m.natural?.n||0}`,m.complete?.false_clarify||0]));$('errors').textContent=JSON.stringify({routine:c.metrics.routine,boundary:c.metrics.boundary,field_errors:c.metrics.field_errors,raw_and_guarded:c.metrics.by_condition},null,2)}else{$('tools').textContent='此配置尚未完成评测';$('errors').textContent='未运行'}curve();$('cases').replaceChildren();for(const c of D.cases.filter(c=>$('condition').value==='all'||c.condition===$('condition').value)){const box=document.createElement('div');box.className='case';const title=document.createElement('b');title.textContent=c.request;box.append(title);let p=document.createElement('p');p.className='muted';p.textContent=c.id+' · '+c.reason;box.append(p);let gold=document.createElement('pre');gold.textContent='正确动作：'+JSON.stringify(c.gold);box.append(gold);for(const[a,m]of Object.entries(c.models)){let pre=document.createElement('pre');pre.className=m.success?'ok':'bad';pre.textContent=a+' · '+(m.success?'正确':m.failure)+'\n'+m.output;box.append(pre)}$('cases').append(box)}}
table('refs',['固定参考',...keys],Object.entries(D.references).map(([n,m])=>[n,...keys.map(k=>fmt(m[k]))]));table('effects',['差异',...['C','Q','F','J'].map(k=>k+' pp [95%区间]')],Object.entries(D.effects).map(([n,e])=>[n,...['C','Q','F','J'].map(k=>`${e[k].delta_pp>=0?'+':''}${e[k].delta_pp.toFixed(1)} [${e[k].ci95_pp.map(x=>x.toFixed(1)).join(', ')}]`)]));table('gates',['配置','C最低个数','F最多','通过','未过项'],D.candidates.map(c=>[c.arm+' / '+c.updates,c.gate.complete_floor_count,3,c.gate.eligible?'是':'否',c.gate.reasons.join(', ')]));let chosen=D.selection.selected;$('headline').textContent=chosen?`开发集筛选候选：${chosen.arm}，${chosen.updates} 次更新。历史回归状态：${D.regression.status}。`:'没有检查点同时通过全部能力保留门槛。';$('selection').textContent=chosen?JSON.stringify({arm:chosen.arm,updates:chosen.updates,metrics:Object.fromEntries(keys.map(k=>[k,chosen.metrics[k]])),gate:chosen.gate},null,2):'无合格候选，不追加训练搜索。';$('regression').textContent=JSON.stringify(D.regression,null,2);$('cost').textContent=JSON.stringify({training:D.training,evaluated_generations:D.verification.total_generations,generation_minutes:D.verification.total_generation_seconds/60,resources:D.resource_summaries,token_audit:D.token_audit},null,2);$('limits').textContent=JSON.stringify({fingerprint:D.fingerprint,limitations:D.limitations,verification:D.verification.status},null,2);for(let id of ['step','metric','arm','condition'])$(id).addEventListener('change',render);render();if(D.status!=='complete'){$('headline').textContent='实验未完成，仅描述已完成配置；不进行候选选择或晋级。'+(D.reason||'');$('selection').textContent=JSON.stringify({status:D.status,common_updates:D.common_updates,missing_cells:D.missing_cells,partial_evaluations:D.partial_evaluations},null,2)}
</script></html>'''
