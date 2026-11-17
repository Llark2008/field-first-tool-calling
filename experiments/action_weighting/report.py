"""Descriptive same-case comparison; no verdict from incomplete cells."""
import html
import json
import math
from pathlib import Path
from toolcall_lab.run_clarification_ablation import write_json


def verify_marker(marker,fingerprint,report_sha256):
    if marker.get('status')!='complete' or marker.get('fingerprint')!=fingerprint or marker.get('report_sha256')!=report_sha256:
        raise ValueError('Completion marker does not match report identity')


def require_completion(training,resources):
    if not training or any(training.get(k)!=v for k,v in dict(status='complete',microsteps=600,optimizer_updates=75).items()):
        raise ValueError('Complete training evidence required')
    if not resources or resources[-1].get('error') or resources[-1].get('n_samples',0)<2:
        raise ValueError('Clean final resource evidence required')
    baseline=resources[0]['initial_swap_used_bytes']
    if any(r.get('initial_swap_used_bytes')!=baseline for r in resources): raise ValueError('Resume reset resource baseline')
    if resources[-1]['peak_sampled_rss_bytes']>10*1024**3 or resources[-1]['peak_swap_growth_bytes']>2*1024**3:
        raise ValueError('Final resources exceeded guard')


def decision_metrics(rows):
    result={}
    for action in ['all','call','clarify','unsupported']:
        selected=[r for r in rows if action=='all' or r['gold_action']==action]
        if not selected: continue
        decisions=[r['token_details'][r['action_positions'][0]] for r in selected]
        correct=sum(t['correct'] for t in decisions)
        result[action]=dict(correct=correct,n=len(selected),accuracy=correct/len(selected),loss=sum(t['loss'] for t in decisions)/len(selected))
    return result


def curve_series(events):
    result={}
    for arm,field,label in [('C0','loss','C0 原始loss'),('W20','loss','W20 原始loss'),('W20','objective_loss','W20 加权目标'),('W20','decision_loss','W20 首动作loss')]:
        points=[[e['optimizer_updates'],e[field]] for e in events.get(arm,[])]
        if any(not math.isfinite(y) or y<0 for x,y in points): raise ValueError('Invalid curve value')
        if [x for x,y in points]!=list(range(1,len(points)+1)): raise ValueError('Curve update grid mismatch')
        result[label]=points
    return result


def curve_html(series):
    if not any(series.values()): return '<p>尚无训练记录。</p>'
    maximum=max(y for values in series.values() for x,y in values)*1.05 or 1
    content=f'<svg viewBox="0 0 900 300" role="img" aria-label="训练损失曲线"><path d="M55 15 V250 H865" fill="none" stroke="#a7bbc6"/><text x="55" y="280" fill="#a7bbc6">0</text><text x="820" y="280" fill="#a7bbc6">75 更新</text><text x="5" y="25" fill="#a7bbc6">{maximum:.2f}</text><text x="20" y="250" fill="#a7bbc6">0</text>'
    legend=''
    for (name,points),color in zip(series.items(),['#69b8ff','#7ee0b6','#ffb75d','#df9bff']):
        xy=' '.join(f'{55+x/75*810:.2f},{250-y/maximum*230:.2f}' for x,y in points)
        content+=f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="2"/>'
        legend+=f'<span style="color:{color};margin-right:20px">{html.escape(name)}</span>'
    return '<p>'+legend+'</p>'+content+'</svg><p class="muted">每点为一次更新内8题的平均值，未平滑。原始loss采用相同口径；加权目标及首动作loss为不同目标，不能按数值大小互相排名。</p>'


def verdict(grid, replay_complete):
    if set(grid)!={f'{a}-{s}' for a in ['C0','W20'] for s in [200,400,600]} or not replay_complete: return None
    m=grid['W20-600']; base=grid['C0-600']
    gates={k:m[k]['count']>=v for k,v in [('C',56),('boundary',27),('U',11)]}
    gates['F']=m['F']['count']<=3
    gain=m['Q']['count']-base['Q']['count']
    return dict(gates=gates,capability_pass=all(gates.values()),Q_gain=gain,joint_pass=all(gates.values()) and gain>=4,
        rule='Primary = 75 updates; C>=56/60, F<=3/60, boundary>=27/30, U>=11/12, Q-C0>=4/48')


def render(data):
    esc=lambda x:html.escape(str(x))
    rows=''
    for name,m in data['development'].items():
        rows+='<tr><td>'+esc(name)+'</td>'+''.join(f'<td>{m[k]["count"]}/{m[k]["n"]}</td>' for k in ['C','Q','F','P','U','J'])+'</tr>'
    replay=''
    for name,m in data['replay'].items():
        replay+=f'<tr><td>{esc(name)}</td><td>{m["all"]["sample_mean_loss"]:.4f}</td><td>{m["first_decision_loss"]:.4f}</td><td>{m["first_decision"]["all"]["correct"]}/120</td><td>{m["weighted_objective"]:.4f}</td><td>{m["call"]["strict_correct"]}/60</td><td>{m["clarify"]["strict_correct"]}/48</td><td>{m["all"]["strict_correct"]}/120</td></tr>'
    message='尚未完成：不作联合效果判定。' if data['verdict'] is None else ('达到预设联合目标。' if data['verdict']['joint_pass'] else '未达到预设联合目标；此结论只适用于本次固定权重和预算。')
    details=esc(json.dumps(data,ensure_ascii=False,indent=2))
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>动作位置加权 · 实验结果</title>
<style>body{{font:17px system-ui;background:#101820;color:#e5edf3;max-width:1120px;margin:36px auto;padding:20px}}section{{background:#192630;padding:24px;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px 7px;border-bottom:1px solid #455}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}a{{color:#9de7d0}}.muted{{color:#a7bbc6}}</style>
<h1>动作位置加权 W20</h1><p>固定 600 条曝光 · 75 次更新 · 动作20% + 其余答案80%</p><section><h2>{message}</h2><p>保持数据、优化器、学习率与推理条件不变，对照 C0 复用已认证的历史检查点与输出。</p></section>
<section><h2>开发集自由生成</h2><p>C 完整正确 ↑ · Q 缺参精确正确 ↑ · F 完整误追问 ↓ · P 缺参擅自调用 ↓ · U 不支持正确 ↑ · J 配对双正确 ↑</p><table><tr><th>组别 / 微步</th><th>C</th><th>Q</th><th>F</th><th>P</th><th>U</th><th>J</th></tr>{rows}</table><p class="muted">200 / 400 / 600 微步对应25 / 50 / 75更新。主要结论固定在75更新，不能按最好点替代。</p></section>
<section><h2>训练曲线</h2>{curve_html(data.get('curves',{}))}</section>
<section><h2>75更新：已见训练样本回放</h2><table><tr><th>组别</th><th>原始loss</th><th>首动作loss</th><th>首动作TF正确</th><th>20%目标*</th><th>完整</th><th>缺参</th><th>整体</th></tr>{replay}</table><p class="muted">固定120题为已见600题中的工具/配对平衡子集，不是全训练集普查。*两组在同一评测口径下重新计算；C0没有优化此加权目标。Teacher forcing与实际自由生成分别报告。</p></section>
<section><h2>完整数值、案例与来源</h2><pre>{details}</pre></section><p><a href="/">返回实时进度</a></p></html>'''


def write(data,root,source):
    page=render(data); write_json(Path(root)/'report-data.json',data)
    (Path(root)/'results.html').write_text(page)
    reports=Path(source)/'reports'; reports.mkdir(exist_ok=True)
    write_json(reports/'action-weighting-results.json',data)
    (reports/'action-weighting-results.html').write_text(page)
