"""Local-only diagnostic progress; no training panel or deployment claims."""
from pathlib import Path
import sys
from toolcall_lab import live_progress

ROOT=Path.home()/'.local/share/local-tool-calling/runs/field-action-diagnostics-20261116'
HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段到动作 · 诊断进度</title>
<style>body{font:16px system-ui;background:#101b24;color:#e6edf1;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.7}h1{font-size:32px}a{color:#7ce0bc}.muted{color:#a6bac8}.cards{display:flex;gap:18px;flex-wrap:wrap}.card{background:#1a2b36;border:1px solid #314956;border-radius:14px;padding:22px;margin:18px 0;flex:1;min-width:210px}.big{font-size:28px;font-weight:650}progress{width:100%;accent-color:#77dfba}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #33434e}pre{white-space:pre-wrap;word-break:break-word}#alert{color:#ffc082}</style>
<h1>字段到动作：问题出在哪里？</h1><p class="muted">固定 F@75 检查点 · 本轮不训练 · 开发集 120 题 · 封存测试不使用</p>
<div class="cards"><div class="card">当前阶段<div id="stage" class="big">准备</div><div id="status"></div></div><div class="card">本阶段完成<div id="count" class="big">—</div><progress id="bar"></progress></div><div class="card">运行时间<div id="elapsed" class="big">—</div><div id="beat" class="muted"></div></div></div>
<p id="alert"></p><p id="message"></p><div class="card"><b>已完成的对照</b><p class="muted">B 原始生成；D 模型字段 + 确定性转换；G 正确字段 + 原提示续写；S 正确字段 + 独立映射提示。G/S 使用标准字段，仅作诊断。</p><table><thead><tr><th>方案</th><th>完整正确 /60</th><th>缺参识别 /48</th><th>误追问 /60 ↓</th><th>不支持 /12</th><th>总正确 /120</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="cards"><div class="card"><b>审计 / 状态</b><pre id="audit">准备中</pre></div><div class="card"><b>资源 / 推理</b><pre id="metrics">未加载模型</pre></div></div><p id="result"></p><p class="muted">每 5 秒刷新 · 仅本机访问 · 当前阶段有心跳，完成后显示固定结果</p>
<script>const $=x=>document.getElementById(x);async function refresh(){try{let r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);let s=await r.json();$('stage').textContent=s.detail||s.stage;$('status').textContent=s.status;$('count').textContent=`${s.step||0} / ${s.total_steps||0}`;$('bar').max=s.total_steps||1;$('bar').value=s.step||0;$('message').textContent=s.message||'';$('elapsed').textContent=(((s.finished_at||Date.now()/1000)-s.started_at)/60).toFixed(1)+' 分钟';$('beat').textContent=new Date(s.updated_at*1000).toLocaleTimeString();$('alert').textContent=s.stale?'超过 90 秒没有工作进程心跳，请检查状态。':'';$('audit').textContent=JSON.stringify(s.audit||{},null,2);$('metrics').textContent=JSON.stringify(s.metrics||{},null,2);$('rows').replaceChildren();for(let c of s.comparisons||[]){let tr=document.createElement('tr');for(let v of [c.arm,c.C,c.Q,c.F,c.U,c.success]){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('rows').append(tr)}$('result').replaceChildren();if(s.results_ready){let a=document.createElement('a');a.href='/results';a.textContent='打开完整结果和具体案例 →';$('result').append(a)}}catch(e){$('alert').textContent='暂时无法连接进度服务，保留上次显示。'}}refresh();setInterval(refresh,5000);</script></html>'''

def main():
    live_progress.HTML=HTML
    sys.argv=[sys.argv[0],'--root',str(ROOT),'--port','49414'];live_progress.main()

if __name__=='__main__':main()
