"""Loopback-only live progress; serves public status, never arbitrary runtime files."""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_LOCK = threading.Lock()


def read_state(root):
    path = Path(root) / 'progress.json'
    state = json.loads(path.read_text()) if path.exists() else {'stage': 'preparing', 'status': 'pending'}
    state['age_seconds'] = max(0, time.time() - state.get('updated_at', time.time()))
    state['stale'] = state.get('status') == 'running' and state['age_seconds'] > 90
    return state


def update_state(root, **values):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        path = root / 'progress.json'
        state = json.loads(path.read_text()) if path.exists() else {'started_at': time.time(), 'history': []}
        if 'stage' in values and values['stage'] != state.get('stage'):
            state['history'].append({'stage': values['stage'], 'time': time.time()})
        state.update(values)
        state['updated_at'] = time.time()
        tmp = root / f'.progress-{os.getpid()}.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(path)


class Heartbeat:
    def __init__(self, root):
        self.root = root
        self.stop_event = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.wait(15):
            update_state(self.root)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地工具调用实验 · 实时进展</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#101820;color:#e5edf3}body{max-width:1100px;margin:36px auto;padding:0 22px}h1{font-size:29px;margin-bottom:8px}.muted{color:#9aadb9;font-size:14px}.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:#192630;border:1px solid #2b3d4b;border-radius:15px;padding:21px;margin:15px 0;flex:1;min-width:210px}.value{font-size:29px;font-weight:650;margin:8px 0}progress{width:100%;height:18px;accent-color:#64d7b3}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 8px;border-bottom:1px solid #2b3d4b;text-align:left}pre{white-space:pre-wrap;word-break:break-word;font-size:13px}.pill{display:inline-block;border-radius:20px;padding:5px 13px;background:#284b45;color:#a9efd9}#alert{color:#ffba83}.note{border-left:3px solid #64d7b3;padding-left:15px;line-height:1.8}a{color:#8fdcc9}.timeline{display:flex;gap:8px;flex-wrap:wrap}.timeline span{padding:8px 12px;border-radius:8px;background:#22323e;font-size:13px}.current{outline:1px solid #64d7b3}svg{width:100%;height:180px}button{background:#254d43;color:#dffbf2;border:0;border-radius:8px;padding:10px;cursor:pointer}</style>
<h1>本地工具调用实验</h1><p class="muted">A：显式缺参训练 · B：自然缺参训练 · 数据 / 训练 / 评测全流程</p>
<div class="timeline" id="timeline"></div><p id="alert"></p>
<div class="row"><section class="card"><div class="muted">当前阶段</div><div class="value" id="stage">连接中</div><span class="pill" id="status">读取实时记录</span></section><section class="card"><div class="muted">本阶段进度</div><div class="value" id="steps">—</div><progress id="bar" max="1" value="0"></progress><div class="muted" id="detail"></div></section><section class="card"><div class="muted">耗时 / 最近心跳</div><div class="value" id="elapsed">—</div><div class="muted" id="updated">—</div></section></div>
<div class="note" id="message"></div><div class="row"><section class="card"><b>训练与资源</b><pre id="metrics">尚未开始训练</pre></section><section class="card"><b>训练损失（辅助指标）</b><svg id="chart" viewBox="0 0 500 180"></svg><div class="muted">损失下降不等于实际任务成功率提高。</div></section></div>
<section class="card"><b>开发集检查点</b><p class="muted">完整任务能力保留条件未通过时，最终测试保持封存。</p><table><thead><tr><th>模型 / 更新</th><th>完整任务</th><th>自然缺参正确追问</th><th>不支持请求</th><th>状态</th></tr></thead><tbody id="checks"></tbody></table></section>
<section class="card"><b>数据审计与最终评测</b><pre id="audit">等待审计</pre><div id="result"></div></section><p class="muted">每 5 秒刷新 · 仅绑定本机 127.0.0.1 · 页面不包含封存测试题或访问令牌</p>
<script>
const names={preparing:'实现与数据准备',audit:'数据审计',baseline:'原模型开发评测',training_A:'A 组训练',training_B:'B 组训练',dev_A:'A 组开发评测',dev_B:'B 组开发评测',selection:'检查点选择',final_eval:'最终测试评测',reporting:'结果整理',complete:'已完成',stopped:'已停止'};
const states={pending:'待开始',running:'运行中',complete:'完成',failed:'失败 / 已停止',stopped:'按规则停止'};
const $=id=>document.getElementById(id);const fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
function chart(points){let s=$('chart');s.replaceChildren();if(!points?.length)return;let a=points.slice(-100),max=Math.max(...a.map(x=>x.loss),0.01),path=a.map((v,i)=>`${i?'L':'M'}${20+i*460/Math.max(1,a.length-1)},${155-v.loss/max*130}`).join(' ');let p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',path);p.setAttribute('fill','none');p.setAttribute('stroke','#64d7b3');p.setAttribute('stroke-width','2');s.append(p)}
async function refresh(){try{let r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);let s=await r.json();$('stage').textContent=names[s.stage]||s.stage;$('status').textContent=states[s.status]||s.status;$('alert').textContent=s.stale?'⚠ 超过 90 秒未收到工作进程心跳，请检查运行状态。':'';$('steps').textContent=s.total_steps?`${s.step||0} / ${s.total_steps}`:'—';$('bar').max=s.total_steps||1;$('bar').value=s.step||0;$('detail').textContent=s.detail||'';$('message').textContent=s.message||'';$('elapsed').textContent=fmt(((s.finished_at||Date.now()/1000)-s.started_at)/60)+' 分钟';$('updated').textContent=new Date(s.updated_at*1000).toLocaleString()+' · '+fmt(s.age_seconds,0)+' 秒前';$('metrics').textContent=s.metrics?JSON.stringify(s.metrics,null,2):'尚未开始训练';$('audit').textContent=s.audit?JSON.stringify(s.audit,null,2):'等待审计';chart(s.loss_history);$('checks').replaceChildren();for(let c of s.checkpoints||[]){let tr=document.createElement('tr');for(let v of [c.name,c.complete,c.clarify,c.unsupported,c.status||'已记录']){let td=document.createElement('td');td.textContent=v??'—';tr.append(td)}$('checks').append(tr)}$('timeline').replaceChildren();for(let key of ['preparing','audit','baseline','training_A','training_B','selection','final_eval','reporting','complete']){let el=document.createElement('span');el.textContent=names[key];if(key===s.stage)el.className='current';$('timeline').append(el)}$('result').replaceChildren();if(s.results_ready){let a=document.createElement('a');a.href='/results';a.textContent='打开结果对比页 →';$('result').append(a)}}catch(e){$('alert').textContent='⚠ 无法连接进展服务，已保留最后显示的数据。'} }
refresh();setInterval(refresh,5000);
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                content, mime = HTML.encode(), 'text/html; charset=utf-8'
            elif self.path == '/api/state':
                content, mime = json.dumps(read_state(root), ensure_ascii=False).encode(), 'application/json'
            elif self.path == '/results' and (root / 'results.html').exists():
                content, mime = (root/'results.html').read_bytes(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    (root/'monitor.json').write_text(json.dumps({'url': url, 'pid': os.getpid()}))
    print(url, flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
