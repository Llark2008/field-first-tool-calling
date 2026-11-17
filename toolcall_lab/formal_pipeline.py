"""Run the approved experiment, with frozen inputs and a sealed-test gate."""
import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from .formal_data import sha
from .live_progress import Heartbeat,read_state,update_state
from .run import configure_runtime,runtime_root
from .train_formal import RECIPE,MODEL_REVISION,audit_tokens,train_arm,verify_frozen,select_checkpoint,STEPS,configure_mlx_memory
from .resource_monitor import ResourceMonitor

CRITICAL_SOURCES=['formal_pipeline.py','formal_data.py','formal_phrases.py','fixtures.py','train_formal.py',
    'evaluate_formal.py','evaluation.py','generation_scoring.py','controls.py','tasks.py','expanded.py',
    'training_spike.py','training_loss.py','resource_monitor.py','run.py']


def read_json(path): return json.loads(Path(path).read_text())
def read_rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]


def verify_sources(source,frozen):
    for name,value in frozen['sources'].items():
        if sha(source/'toolcall_lab'/name)!=value: raise ValueError('Frozen execution source changed: '+name)


def freeze(root,dataset,snapshot,source):
    from mlx_lm.utils import load_tokenizer
    manifest=read_json(dataset/'manifest.json'); verify_frozen(dataset,manifest)
    review=read_json(root/'audit-review.json')
    if review.get('status')!='passed' or review.get('dataset_manifest_sha256')!=sha(dataset/'manifest.json'):
        raise ValueError('Dataset review is missing or stale')
    implementation=read_json(root/'implementation-review.json')
    if implementation.get('status')!='passed': raise ValueError('Implementation review must pass')
    tokens=audit_tokens(dataset,load_tokenizer(snapshot))
    (root/'token-audit.json').write_text(json.dumps(tokens,indent=2))
    frozen=dict(dataset=str(dataset),manifest=manifest,recipe=RECIPE,token_audit=tokens,
        sources={name:sha(source/'toolcall_lab'/name) for name in CRITICAL_SOURCES},
        model_files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},
        versions={n:version(n) for n in ['mlx','mlx-lm','transformers','jsonschema']},
        audit_review_sha256=sha(root/'audit-review.json'),implementation_review_sha256=sha(root/'implementation-review.json'),
        protocol='Paired A/B; common dev checkpoint; sealed final conditional on preservation; fixed prompts/temperature/output budget.')
    frozen['fingerprint']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    path=root/'frozen.json'
    if path.exists() and read_json(path)!=frozen: raise ValueError('Refusing to overwrite a different frozen experiment')
    path.write_text(json.dumps(frozen,indent=2))
    return frozen


def record_checkpoint(root,name,metrics):
    old=read_state(root).get('checkpoints',[]); old=[c for c in old if c['name']!=name]
    def display(condition):
        m=metrics[condition]; return f"{m['success']}/{m['n']} ({100*m['success']/m['n']:.1f}%)"
    old.append(dict(name=name,complete=display('complete'),clarify=display('natural'),unsupported=display('unsupported')))
    update_state(root,checkpoints=old)


def run(root,dataset,snapshot,source,frozen,resources):
    import mlx.core as mx
    from mlx_lm import load
    from .evaluate_formal import evaluate_model
    from .report_formal import make_report
    configure_mlx_memory(mx)
    fingerprint=frozen['fingerprint']
    verify_frozen(dataset,frozen['manifest']); verify_sources(source,frozen)
    # Model identity checked before any train/eval, including resumed runs.
    for name,value in frozen['model_files'].items():
        if sha(snapshot/name)!=value: raise ValueError('Cached base model changed: '+name)
    dev=read_rows(dataset/'dev.jsonl')

    def evaluate(cases,split,name,prompt='v2',adapter=None,stage=None):
        verify_sources(source,frozen); resources.check()
        output=root/'eval'/(split+'-'+name+'.jsonl')
        metadata=dict(name=name,prompt=prompt,dataset_fingerprint=fingerprint,split=split,
            adapter_sha256=sha(adapter/'adapters.safetensors') if adapter else None,
            max_output_tokens=192,temperature=0,thinking=False)
        model,tokenizer=load(str(snapshot),adapter_path=str(adapter) if adapter else None)
        try: return evaluate_model(model,tokenizer,cases,output,metadata,root,stage or 'baseline',resources)
        finally:
            del model,tokenizer
            mx.clear_cache()

    baseline=evaluate(dev,'dev','base_v2',stage='baseline')
    record_checkpoint(root,'原模型 v2',baseline)
    arms={}
    for arm in ['A','B']:
        verify_sources(source,frozen)
        train_arm(snapshot,dataset,arm,root/('train-'+arm),fingerprint,frozen['token_audit'],root,resources)
        arms[arm]={}
        for step in STEPS:
            metrics=evaluate(dev,'dev',f'{arm}-{step}',adapter=root/f'train-{arm}/step-{step:04d}',stage='dev_'+arm)
            arms[arm][step]=metrics
            record_checkpoint(root,f'{arm} · {step//8} 次更新',metrics)
    a_summary=read_json(root/'train-A/summary.json'); b_summary=read_json(root/'train-B/summary.json')
    if a_summary['schedule_sha256']!=b_summary['schedule_sha256']: raise ValueError('A/B sample-exposure schedules differ')
    update_state(root,stage='selection',status='running',step=0,total_steps=0,message='按冻结规则选择 A/B 共同检查点。')
    selection=select_checkpoint(baseline,arms['A'],arms['B'])
    selection['fingerprint']=fingerprint
    (root/'selection.json').write_text(json.dumps(selection,indent=2))
    selected=selection['selected_step']
    if selected is None:
        update_state(root,stage='reporting',message='没有共同检查点满足完整任务保留条件；最终测试保持封存，整理开发失败分析。')
        mode='dev_failure'
    else:
        # Historic diagnostics are explicitly development history, not selection input.
        from .expanded import expanded_cases
        historical=expanded_cases()
        for arm in ['A','B']:
            evaluate(historical,'historical',arm+'-selected',adapter=root/f'train-{arm}/step-{selected:04d}',stage='dev_'+arm)
        verify_frozen(dataset,frozen['manifest'])
        gate=dict(status='opened',step=selected,fingerprint=fingerprint,selection_sha256=sha(root/'selection.json'),
                  test_sha256=frozen['manifest']['files']['test.jsonl']['sha256'],opened_at=time.time())
        (root/'FINAL_ACCESS.json').write_text(json.dumps(gate,indent=2))
        final=read_rows(dataset/'test.jsonl')
        for name,prompt,adapter in [('base_v2','v2',None),('base_no_examples','no_examples',None),
            ('base_checklist','checklist',None),('A_v2','v2',root/f'train-A/step-{selected:04d}'),
            ('B_v2','v2',root/f'train-B/step-{selected:04d}')]:
            evaluate(final,'final',name,prompt,adapter,stage='final_eval')
        mode='final'
        update_state(root,stage='reporting',message='最终评测完成，正在汇总配对差异、错误类型、置信区间与案例。')
    resources.check()
    (root/'resource-summary.json').write_text(json.dumps(resources.summary(),indent=2))
    payload=make_report(root,dataset,selection,source,mode)
    (root/'completed.json').write_text(json.dumps(dict(status='complete',mode=mode,selection=selection,
        fingerprint=fingerprint,finished_at=time.time(),practical_targets=payload['practical_targets']),indent=2))
    update_state(root,stage='complete',status='complete',finished_at=time.time(),results_ready=True,step=0,total_steps=0,
        message='正式实验与结果对比页已完成。' if mode=='final' else '开发集实验已完成；能力保留条件未通过，按计划保留最终测试封存，并生成失败分析页。')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True); p.add_argument('--run-root',required=True)
    p.add_argument('--freeze-only',action='store_true'); args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    runtime=runtime_root(source,os.environ.get('TOOLCALL_RUNTIME_DIR')); configure_runtime(runtime)
    root=Path(args.run_root).resolve(); dataset=Path(args.dataset).resolve()
    if not root.is_relative_to(runtime/'runs') or not dataset.is_relative_to(runtime/'datasets'):
        p.error('Use external runtime run and dataset directories')
    root.mkdir(parents=True,exist_ok=True)
    snapshot=runtime/'cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots'/MODEL_REVISION
    # OS lock prevents two training pipelines or two progress writers for this run.
    import fcntl
    lock=(root/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'worker.pid').write_text(str(os.getpid()))
    beat=Heartbeat(root); beat.start()
    resources=ResourceMonitor(root/f'resources-{int(time.time())}.jsonl')
    resources.start()
    try:
        frozen=read_json(root/'frozen.json') if (root/'frozen.json').exists() else freeze(root,dataset,snapshot,source)
        if frozen['dataset']!=str(dataset): raise ValueError('Different dataset in frozen run')
        if args.freeze_only:
            verify_frozen(dataset,frozen['manifest']); verify_sources(source,frozen)
            print(json.dumps(dict(fingerprint=frozen['fingerprint'],status='frozen')),flush=True)
            update_state(root,status='pending',message='数据、评分器、训练配置已冻结；准备正式运行。')
        else: run(root,dataset,snapshot,source,frozen,resources)
    except BaseException as exc:
        (root/'last-error.json').write_text(json.dumps(dict(error=repr(exc),time=time.time(),resources=resources.summary()),indent=2))
        update_state(root,status='failed',message='流程已停止：'+repr(exc))
        raise
    finally:
        beat.stop(); resources.stop()
        (root/'resource-summary.json').write_text(json.dumps(resources.summary(),indent=2))
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__': main()
