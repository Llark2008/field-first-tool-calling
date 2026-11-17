"""Source-backed three-arm report; local standalone HTML/JSON/Markdown."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .formal_data import sha


def examples(cases,outputs):
    groups=defaultdict(list);chosen={};maps={a:{r['case_id']:r for r in rows} for a,rows in outputs.items()}
    key=lambda c:hashlib.sha256(c['id'].encode()).hexdigest()
    for c in cases:groups[c['condition'],c['target_tool']].append(c)
    for values in groups.values():chosen[min(values,key=key)['id']]='固定分层样本（工具 × 条件，按 ID 哈希选择）'
    for a in ['M1','M2']:
        if a not in maps or 'M0' not in maps:continue
        for condition in ['complete','natural']:
            for improved in [True,False]:
                values=[c for c in cases if c['condition']==condition and maps[a][c['id']]['score']['success']==improved and maps['M0'][c['id']]['score']['success']!=improved]
                if values:
                    i=min(values,key=key)['id'];chosen[i]=chosen.get(i,'')+f'；{a} 相对 M0 的'+('修复' if improved else '退化')+'（结果驱动补充）'
    cm={c['id']:c for c in cases}
    return [dict(id=i,request=cm[i]['request'],condition=cm[i]['condition'],target_tool=cm[i]['target_tool'],gold=cm[i]['gold'],reason=reason,
        models={a:dict(output=m[i]['output'],success=m[i]['score']['success'],failure=m[i]['failure_type']) for a,m in maps.items()}) for i,reason in sorted(chosen.items())]


def render_html(payload):
    return HTML.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))


def write_report(payload,root,source):
    root=Path(root);reports=Path(source)/'reports';reports.mkdir(exist_ok=True)
    data=json.dumps(payload,ensure_ascii=False,indent=2);html=render_html(payload)
    for p in [root/'report-data.json',reports/'training-mechanism-metrics.json']:p.write_text(data)
    for p in [root/'results.html',reports/'training-mechanism-results.html']:p.write_text(html)
    lines=['# 损失归一化与 Adam 偏差修正','',f"状态：{payload['status']}。主比较固定75更新。",'',
        'M0：原样本均值 / 无偏差修正；M1：累计窗口内答案 token 均值 / 无偏差修正；M2：原样本均值 / 开启偏差修正。原D0、学习率1e-4、600微步、75更新相同。','',
        '| 配置 | 完整 C /60 | 缺参 Q /48 | 误追问 F /60 ↓ | 擅自调用 P /48 ↓ | 不支持 U /12 | 配对 J /48 | 能力门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for c in payload['candidates']:
        if c['updates']==75:lines.append('| '+c['arm']+' | '+' | '.join(str(c['metrics'][k]['count']) for k in ['C','Q','F','P','U','J'])+' | '+('通过' if c['gate']['eligible'] else ', '.join(c['gate']['reasons']))+' |')
    lines+=['','能力门槛：C≥56/60、F≤3/60、边界完整≥27/30、不支持≥11/12。联合目标额外要求Q比同更新M0至少多4题、F更少。','',
            '| 75更新差异 | C 百分点 [95%区间] | Q 百分点 [95%区间] | F 百分点 [95%区间] | J 百分点 [95%区间] |','|---|---:|---:|---:|---:|']
    for n,e in payload['effects'].items():
        lines.append('| '+n+' | '+' | '.join(f"{e[k]['delta_pp']:+.1f} [{e[k]['ci95_pp'][0]:+.1f}, {e[k]['ci95_pp'][1]:+.1f}]" for k in ['C','Q','F','J'])+' |')
    lines+=['','联合目标判定：'+json.dumps(payload['verdicts'],ensure_ascii=False),'',
        '区间使用2,000次按工具和家族大小分层的配对表达家族bootstrap；不能衡量训练种子的变化。','',
        f"训练耗时：{sum(t['active_training_seconds'] for t in payload['training'].values())/60:.1f}分钟（正式两组；不含M0复现与评测）。",
        f"输出验证：{payload['verification']['counts']}；新增生成耗时{payload['verification']['new_generation_seconds']/60:.1f}分钟。",'',
        '单随机种子、合成任务、已经使用过的开发集；本轮没有新最终测试。不以历史和本轮耗时差推断方法速度。',
        '未完成时仅作描述，不给联合目标结论；成功点也需要独立确认。','',f"实验指纹：`{payload['fingerprint']}`"]
    (reports/'training-mechanism-results.md').write_text('\n'.join(lines)+'\n')
    (reports/'training-mechanism-verification.json').write_text(json.dumps(dict(payload['verification'],fingerprint=payload['fingerprint'],
        report_data_sha256=sha(root/'report-data.json'),html_sha256=sha(root/'results.html')),ensure_ascii=False,indent=2))
    return payload


HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练机制对照 · 实验结果</title>
<style>:root{color-scheme:dark;font-family:system-ui;background:#101b24;color:#e8edf3}body{max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.65}h1{font-size:29px}.muted{color:#9fb1c0}.card{background:#1b2b37;border:1px solid #344959;border-radius:14px;padding:22px;margin:18px 0;min-width:0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #344959;white-space:nowrap}select{background:#243f4c;color:#f1f6f9;border:1px solid #56717c;border-radius:7px;padding:8px;margin:5px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}svg{width:100%;height:auto}a{color:#98e7ca}.ok{color:#8de7b7}.bad{color:#ffba91}.case{border-top:1px solid #344959;padding:15px 0}details{margin:12px 0}summary{cursor:pointer}#headline{font-size:19px;border-left:3px solid #98e7ca;padding-left:16px}@media(max-width:600px){body{padding:0 12px}.card{padding:14px}h1{font-size:24px}}</style>
<h1>训练机制对照：损失归一化与 Adam 偏差修正</h1><p class="muted">Qwen3-1.7B 4-bit · 单随机种子 · 开发诊断 · 主比较固定75更新</p><a href="/">实时进度</a><p id="headline"></p>
<section class="card"><b>三个单因素配置</b><p>M0：每样本答案均值，Adam偏差修正关闭。M1：整个8样本累积窗口内，答案token等权。M2：保持M0归一化，只开启Adam偏差修正。</p><p class="muted">均使用原D0曝光序列前600条、学习率1e-4、75更新。仅在数值复现与来源核验通过后才显示M0；M1/M2独立训练。本轮不测试两种改动的组合。</p><div class="table" id="refs"></div></section>
<section class="card"><b>同更新数表现</b><label>优化器更新数 <select id="step"><option>25</option><option>50</option><option selected>75</option></select></label><div class="table" id="grid"></div><p class="muted">C完整正确/60；Q精确缺参识别/48；F完整任务误追问/60↓；P缺参擅自调用/48↓；U不支持请求正确/12；J成对任务均正确/48。来源拦截不计为正确追问。</p></section>
<section class="card"><b>75更新：单因素差异</b><div class="table" id="effects"></div><p class="muted">M1−M0检验归一化，M2−M0检验偏差修正。区间来自2,000次配对表达家族重采样，按工具与家族大小分层；不是跨种子稳定性区间。</p></section>
<section class="card"><b>训练轨迹</b><label>指标 <select id="metric"><option value="C">完整 C ↑</option><option value="Q">缺参 Q ↑</option><option value="F">误追问 F ↓</option><option value="J">配对 J ↑</option></select></label><svg id="curve" viewBox="0 0 700 310" aria-label="三组表现曲线"></svg></section>
<section class="card"><b>能力保留与联合目标</b><p>C≥56/60、F≤3/60、边界完整≥27/30、不支持≥11/12。联合目标还要求Q比75更新M0至少多4题、F更少。仅恢复完整任务而Q回到零，不算达成目标。</p><pre id="verdict"></pre><details><summary>全部检查点门槛</summary><div class="table" id="gates"></div></details></section>
<section class="card"><b>逐工具与字段错误</b><label>配置 <select id="arm"><option>M0</option><option>M1</option><option>M2</option></select></label><p class="muted">随上方更新数选择联动。</p><div class="table" id="tools"></div><details><summary>边界、字段与原始／系统评分</summary><pre id="errors"></pre></details></section>
<section class="card"><b>75更新具体示例</b><label>任务 <select id="condition"><option value="all">全部</option><option value="complete">完整</option><option value="natural">缺参</option><option value="unsupported">不支持</option></select></label><div id="cases"></div></section>
<section class="card"><b>复现、资源与来源</b><pre id="cost"></pre><details><summary>完整审计与限制</summary><pre id="audit"></pre></details></section>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=x=>document.getElementById(x),keys=['C','Q','F','P','U','J'],arms=['M0','M1','M2'],colors={M0:'#adb8c6',M1:'#82ddc1',M2:'#e9ba74'};
function table(id,heads,rows){const t=document.createElement('table'),h=document.createElement('tr');for(const v of heads){const c=document.createElement('th');c.textContent=v;h.append(c)}t.append(h);for(const row of rows){const tr=document.createElement('tr');for(const v of row){const td=document.createElement('td');td.textContent=v;tr.append(td)}t.append(tr)}$(id).replaceChildren(t)}
const fmt=m=>m?`${m.count}/${m.n} (${(100*m.rate).toFixed(1)}%)`:'未运行';
function curve(){const svg=$('curve');svg.replaceChildren();const k=$('metric').value;function add(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[n,v]of Object.entries(attrs))e.setAttribute(n,v);if(text!==undefined)e.textContent=text;svg.append(e)}for(let y=0;y<=100;y+=25){let p=260-y*2.1;add('line',{x1:55,y1:p,x2:660,y2:p,stroke:'#344959'});add('text',{x:15,y:p+4,fill:'#a1b2c1','font-size':12},y+'%')}for(const s of [25,50,75])add('text',{x:55+(s-25)*12.1,y:285,fill:'#a1b2c1','font-size':12},s);for(const a of arms){const points=D.candidates.filter(c=>c.arm===a).sort((a,b)=>a.updates-b.updates).map(c=>[55+(c.updates-25)*12.1,260-c.metrics[k].rate*210]);add('polyline',{points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:colors[a],'stroke-width':2.5});for(const[x,y]of points)add('circle',{cx:x,cy:y,r:4,fill:colors[a]});add('text',{x:80+arms.indexOf(a)*180,y:23,fill:colors[a],'font-size':14},a)}}
function render(){const cs=D.candidates.filter(c=>c.updates===Number($('step').value));table('grid',['组',...keys,'能力门槛'],arms.map(a=>{const c=cs.find(c=>c.arm===a);return c?[a,...keys.map(k=>fmt(c.metrics[k])),c.gate.eligible?'通过':c.gate.reasons.join(', ')]:[a,...keys.map(k=>'未运行'),'未运行']}));const c=cs.find(c=>c.arm===$('arm').value);if(c){table('tools',['工具','完整正确','缺参正确','完整误追问'],Object.entries(c.metrics.by_tool).map(([n,m])=>[n,`${m.complete?.success||0}/${m.complete?.n||0}`,`${m.natural?.success||0}/${m.natural?.n||0}`,m.complete?.false_clarify||0]));$('errors').textContent=JSON.stringify({routine:c.metrics.routine,boundary:c.metrics.boundary,field_errors:c.metrics.field_errors,raw_and_guarded:c.metrics.by_condition},null,2)}else{$('tools').textContent='尚未完成评测';$('errors').textContent='未运行'}curve();$('cases').replaceChildren();for(const c of D.cases.filter(c=>$('condition').value==='all'||c.condition===$('condition').value)){let box=document.createElement('div');box.className='case';let title=document.createElement('b');title.textContent=c.request;box.append(title);let p=document.createElement('p');p.className='muted';p.textContent=c.id+' · '+c.reason;box.append(p);let gold=document.createElement('pre');gold.textContent='正确动作：'+JSON.stringify(c.gold);box.append(gold);for(const[a,m]of Object.entries(c.models)){let pre=document.createElement('pre');pre.className=m.success?'ok':'bad';pre.textContent=a+' · '+(m.success?'正确':m.failure)+'\n'+m.output;box.append(pre)}$('cases').append(box)}}
table('refs',['固定参考',...keys],Object.entries(D.references).map(([n,m])=>[n,...keys.map(k=>fmt(m[k]))]));table('effects',['因素差异',...['C','Q','F','J'].map(k=>k+' pp [95%区间]')],Object.entries(D.effects).map(([n,e])=>[n,...['C','Q','F','J'].map(k=>`${e[k].delta_pp>=0?'+':''}${e[k].delta_pp.toFixed(1)} [${e[k].ci95_pp.map(x=>x.toFixed(1)).join(', ')}]`)]));table('gates',['组 / 更新','C下限','F上限','通过','未过项'],D.candidates.map(c=>[c.arm+' / '+c.updates,c.gate.complete_floor_count,3,c.gate.eligible?'是':'否',c.gate.reasons.join(', ')]));
const successes=Object.entries(D.verdicts).filter(([a,v])=>v.joint_goal).map(([a])=>a);$('headline').textContent=D.status==='complete'?(successes.length?successes.join('、')+' 在75更新达到本轮联合目标；需要独立确认。':'75更新时，没有改动同时达到完整能力保留、减少误追问和提高缺参识别的联合目标。'):'实验未完成：仅描述已完成配置，不给联合目标结论。'+(D.reason||'');$('verdict').textContent=D.status==='complete'?Object.entries(D.verdicts).map(([a,v])=>`${a}：能力门槛${v.capability_gate_passed?'通过':'未过'}；Q至少增加4题${v.Q_gain_at_least_four?'达到':'未达'}；F减少${v.F_reduced?'达到':'未达'}；联合目标${v.joint_goal?'达到（待确认）':'未达'}`).join('\n'):JSON.stringify({missing:D.missing_cells,partial:D.partial_evaluations},null,2);$('cost').textContent=JSON.stringify({M0_certification:D.certification.status,training_minutes:Object.fromEntries(Object.entries(D.training).map(([a,t])=>[a,t.active_training_seconds/60])),evaluated_rows:D.verification.counts,new_generation_minutes:D.verification.new_generation_seconds/60,consumed:D.consumed,resources:D.resource_summaries.map(r=>({error:r.error,peak_RSS_GiB:r.peak_sampled_rss_bytes/1024**3,peak_swap_growth_GiB:r.peak_swap_growth_bytes/1024**3}))},null,2);$('audit').textContent=JSON.stringify({fingerprint:D.fingerprint,limitations:D.limitations,certification:D.certification,verification:D.verification,token_audit:D.token_audit},null,2);for(const id of ['step','metric','arm','condition'])$(id).addEventListener('change',render);render();
</script></html>'''
