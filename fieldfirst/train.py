"""Train the fixed 600 examples from the pinned base on Apple Silicon."""
import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path
from fieldfirst.verify import ROOT, read_rows, verify_manifest
from experiments.fields_sft.data import digest
from experiments.fields_sft.train import recipe, checkpoints, save_checkpoint
from experiments.fields_epochs.train import atomic_events, load_events, resume_events
from experiments.field_action_diagnostics.run import immutable, write
from toolcall_lab.formal_data import sha
from toolcall_lab.run import runtime_root, configure_runtime
from toolcall_lab.resource_monitor import ResourceMonitor
from toolcall_lab.train_formal import SEED, LORA, configure_mlx_memory, restore_training_state
from toolcall_lab.train_mechanism import build_step
from toolcall_lab.training_spike import encode_row, MODEL_ID, MODEL_REVISION


def make_config(arm, smoke):
    if arm not in ('A', 'B'): raise ValueError('Choose arm A or B')
    config = recipe('F', '', sha(ROOT/'data/sft-F.jsonl'), 1217)
    config.update(arm=arm, learning_rate={'A': 1e-4, 'B': 5e-5}[arm],
                  max_microsteps=8 if smoke else 1800,
                  checkpoints=[8] if smoke else list(range(200, 1801, 200)),
                  publication_runner=True, smoke=smoke)
    config['fingerprint'] = digest({k: v for k, v in config.items() if k != 'fingerprint'})
    return config


def exposure(steps):
    if steps <= 0 or steps % 8: raise ValueError('Require complete accumulation windows')
    return [i % 600 for i in range(steps)]


def output_root(source, runtime, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Name must be one simple directory name')
    runtime = runtime_root(source, str(runtime))
    out = (runtime/'runs'/name).resolve()
    if not out.is_relative_to(runtime): raise ValueError('Run path escaped runtime')
    return out


def checked_snapshot(path):
    manifest = json.loads((ROOT/'data/model-files.json').read_text())
    for name, expected in manifest.items():
        if sha(Path(path)/name) != expected: raise ValueError('Model bytes changed: ' + name)
    return str(path)


def train_to(snapshot, out, config, target, resources):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    states = checkpoints(out, config)
    initial = int(states[-1].name[5:]) if states else 0
    if initial >= target: return
    resources.check(); configure_mlx_memory(mx)
    model, tok = load(snapshot); resources.check(); model.freeze()
    mx.random.seed(SEED); linear_to_lora_layers(model, 8, LORA)
    optimizer = optim.Adam(learning_rate=config['learning_rate'], betas=config['betas'], eps=config['eps'], bias_correction=False)
    encoded = [encode_row(r, tok, 1217) for r in read_rows(ROOT/'data/sft-F.jsonl')]
    if len(encoded) != 600 or sum(len(t)-o for t,o in encoded) != 29718:
        raise ValueError('Token exposure changed')
    if states:
        restored = restore_training_state(states[-1], model, optimizer, config['fingerprint'])
        if restored != initial: raise ValueError('Restored cursor changed')
    if not math.isclose(float(optimizer.learning_rate.item()), config['learning_rate'], rel_tol=1e-6):
        raise ValueError('Restored learning rate changed')
    path = out/'events.jsonl'
    events = resume_events(load_events(path, initial), initial) if path.exists() else []
    if initial and not events: raise ValueError('Missing committed journal')
    atomic_events(path, events)
    step, state = build_step(model, optimizer, 'sample')
    model.train(); grad = None; window = []; mx.reset_peak_memory()
    if sum(v.size for _,v in tree_flatten(model.trainable_parameters())) != 458752:
        raise ValueError('Adapter parameter count changed')
    try:
        for i in range(initial, target):
            resources.check(); tokens, offset = encoded[i % 600]
            if i % 8 == 0:
                window_tokens = sum(len(encoded[j % 600][0])-encoded[j % 600][1] for j in range(i, i+8))
            padded = min(1217, 1+32*math.ceil(len(tokens)/32))
            if padded < len(tokens): raise ValueError('Truncation prohibited')
            arr = np.zeros((1, padded), dtype=np.int32); arr[0, :len(tokens)] = tokens
            tick = time.perf_counter()
            loss, n, grad = step(mx.array(arr), mx.array([[offset, len(tokens)]]), grad, window_tokens, (i+1)%8 == 0)
            mx.eval(state, loss, n, grad)
            value, count = float(loss.item()), int(n.item())
            if not math.isfinite(value) or count != len(tokens)-offset: raise ValueError('Invalid loss/token count')
            window.append((value, count, time.perf_counter()-tick)); micro = i+1
            if micro % 8 == 0:
                if int(optimizer.step.item()) != micro//8 or sum(x[1] for x in window) != window_tokens:
                    raise ValueError('Accumulation mismatch')
                event = dict(microstep=micro, optimizer_updates=micro//8, loss=sum(x[0] for x in window)/8,
                             supervised_tokens=window_tokens, seconds=sum(x[2] for x in window), mlx_peak_gb=mx.get_peak_memory()/1e9)
                with path.open('a') as h: h.write(json.dumps(event)+'\n'); h.flush()
                print(json.dumps(event), flush=True); window=[]; mx.clear_cache()
            if micro in config['checkpoints']:
                save_checkpoint(out/f'step-{micro:04d}', model, optimizer, micro, config)
        resources.check()
    finally:
        del model, tok, optimizer, grad, state, step
        mx.clear_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=['A', 'B'])
    p.add_argument('--name', required=True, help='Unique run name; rerun the same command to resume')
    p.add_argument('--runtime', default=os.environ.get('TOOLCALL_RUNTIME_DIR', str(Path.home()/'.local/share/field-first-tool-calling')))
    p.add_argument('--snapshot', help='Optional local copy of the exact pinned model; hashes are checked')
    p.add_argument('--smoke', action='store_true', help='8 microsteps + reload + 2 generations per split; no quality claim')
    args = p.parse_args()
    verify_manifest(ROOT)
    out = output_root(ROOT, Path(args.runtime), args.name)
    configure_runtime(Path(args.runtime).expanduser().resolve()); out.mkdir(parents=True, exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = make_config(args.arm, args.smoke); immutable(out/'config.json', config)
        if args.snapshot: snapshot = checked_snapshot(args.snapshot)
        else:
            from huggingface_hub import snapshot_download
            snapshot = checked_snapshot(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
        resources = ResourceMonitor(out/f'resources-{time.time_ns()}.jsonl'); resources.start()
        try:
            from fieldfirst.evaluate import evaluate_checkpoint
            for target in ([8] if args.smoke else [600, 1200, 1800]):
                train_to(snapshot, out, config, target, resources)
                evaluate_checkpoint(snapshot, out, config, target, resources, limit=2 if args.smoke else None)
        finally:
            resources.stop(); write(out/'resource-summary.json', resources.summary())
        resources.check()
        write(out/'complete.json', dict(config_fingerprint=config['fingerprint'], smoke=args.smoke))
        print('Complete. Outputs:', out)

if __name__ == '__main__': main()
