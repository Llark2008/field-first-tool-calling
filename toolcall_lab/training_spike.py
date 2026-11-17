"""A bounded local QLoRA feasibility run, never a formal model-quality claim."""
import argparse
import hashlib
import json
import math
import os
import resource
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from importlib.metadata import version

from .run import configure_runtime, runtime_root

MODEL_REVISION = '3b1b1768f8f8cf8351c712464f906e86c2b8269e'
MODEL_ID = 'mlx-community/Qwen3-1.7B-4bit'


def training_budget(microsteps, batch_size, accumulation):
    if min(microsteps, batch_size, accumulation) < 1 or microsteps % accumulation:
        raise ValueError('Require positive counts and complete gradient-accumulation windows')
    return dict(microsteps=microsteps, sample_exposures=microsteps * batch_size,
                optimizer_updates=microsteps // accumulation)


def validate_tokens(tokens, prefix, max_length):
    if not prefix or tokens[:len(prefix)] != prefix:
        raise ValueError('Training/inference prefix mismatch')
    if len(tokens) <= len(prefix):
        raise ValueError('No supervised answer tokens')
    if len(tokens) > max_length:
        raise ValueError('Refusing to truncate training example')
    return tokens, len(prefix)


def encode_row(row, tokenizer, max_length=2048):
    messages = row['messages']
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                         add_generation_prompt=False, enable_thinking=False)
    tokens, offset = validate_tokens(full, prefix, max_length)
    answer = tokenizer.backend_tokenizer.decode(tokens[offset:], skip_special_tokens=True).strip()
    if answer != messages[-1]['content'] or tokenizer.eos_token_id not in tokens[offset:]:
        raise ValueError('Supervised suffix must be exactly the target answer and its end marker')
    return tokens, offset


class EncodedDataset:
    def __init__(self, rows, tokenizer):
        self.items = [encode_row(row, tokenizer) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='External dataset root containing manifest.json')
    parser.add_argument('--name', default='qlora-spike-20261113')
    parser.add_argument('--microsteps', type=int, default=64)
    args = parser.parse_args()
    budget = training_budget(args.microsteps, 1, 8)
    if args.microsteps > 128:
        parser.error('This entry point is a bounded spike (at most 128 microsteps)')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a simple directory name')
    source = Path(__file__).resolve().parents[1]
    root = runtime_root(source, os.environ.get('TOOLCALL_RUNTIME_DIR'))
    configure_runtime(root)
    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.is_relative_to(root / 'datasets'):
        parser.error('Dataset must be in the external runtime datasets directory')
    out = root / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=False)
    adapter_path = root / 'adapters' / args.name
    adapter_path.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    prompt_version = manifest['prompt_version']
    train_rows, valid_rows = [], []
    for split, dest in (('train', train_rows), ('valid', valid_rows)):
        raw = (dataset_root / 'natural' / (split + '.jsonl')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['variants']['natural'][split]['sha256']:
            raise ValueError('Dataset fingerprint mismatch')
        dest.extend(json.loads(line) for line in raw.splitlines())
    config = dict(model=MODEL_ID, revision=MODEL_REVISION, prompt_version=prompt_version,
                  variant='natural', seed=20261113, batch_size=1, grad_accumulation_steps=8,
                  num_layers=8, learning_rate=1e-4, optimizer='Adam',
                  lora_parameters=dict(rank=8, scale=20.0, dropout=0.0,
                                       keys=['self_attn.q_proj', 'self_attn.v_proj']),
                  **budget)
    summary = dict(status='running', purpose='local resource and pipeline feasibility only', config=config,
                   dataset_manifest=manifest, adapter_path=str(adapter_path))
    summary['versions'] = {name: version(name) for name in ('mlx', 'mlx-lm', 'transformers', 'jsonschema')}
    summary['source_sha256'] = {name: hashlib.sha256((source / 'toolcall_lab' / name).read_bytes()).hexdigest()
                              for name in ('training_spike.py', 'training_loss.py', 'resource_monitor.py', 'controls.py', 'training_data.py')}
    summary_path = out / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    from .resource_monitor import ResourceMonitor
    monitor = ResourceMonitor(out / 'resources.jsonl')
    monitor.start()
    started = time.perf_counter()
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.trainer import TrainingArgs, train, evaluate, iterate_batches
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from .controls import build_control_messages
        from .expanded import expanded_cases
        from .generation_scoring import score_generation
        from .training_loss import assistant_loss

        snapshot = root / 'cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots' / MODEL_REVISION
        if not snapshot.is_dir():
            raise FileNotFoundError('Fixed model revision is not cached')
        mx.random.seed(config['seed'])
        np.random.seed(config['seed'])
        model, tokenizer = load(str(snapshot))
        train_data = EncodedDataset(train_rows, tokenizer)
        # Evaluate the same eight examples before/after; the library otherwise
        # draws a fresh validation subset, confounding the reported loss change.
        valid_data = EncodedDataset(valid_rows[:8], tokenizer)
        summary['validation_probe'] = dict(n=8, selection='first eight of fixed shuffled validation file')
        # Audit BOTH variants, not just the one used in the feasibility run.
        token_audit = {}
        for variant in ('explicit', 'natural'):
            for split in ('train', 'valid'):
                rows = [json.loads(line) for line in (dataset_root / variant / (split + '.jsonl')).read_text().splitlines()]
                encoded = [encode_row(row, tokenizer) for row in rows]
                lengths = [len(t) for t, _ in encoded]
                answer_lengths = [len(t) - offset for t, offset in encoded]
                token_audit[f'{variant}/{split}'] = dict(n=len(rows), min_tokens=min(lengths), max_tokens=max(lengths),
                    mean_tokens=mean(lengths), total_tokens=sum(lengths), answer_tokens=sum(answer_lengths),
                    prefixes_verified=len(rows), min_answer_tokens=min(answer_lengths))
        max_tokens = max(x['max_tokens'] for x in token_audit.values())
        sequence_cap = math.ceil((max_tokens + 1) / 32) * 32
        if sequence_cap > 2048:
            raise ValueError('Dataset exceeds spike sequence budget')
        config['max_seq_length'] = sequence_cap
        summary['token_audit'] = token_audit
        (out / 'token-audit.json').write_text(json.dumps(token_audit, indent=2))
        print('TOKEN_AUDIT', json.dumps(token_audit), flush=True)
        monitor.check()

        model.freeze()
        linear_to_lora_layers(model, config['num_layers'], config['lora_parameters'])
        parameters = tree_flatten(model.trainable_parameters())
        summary['trainable_parameters'] = sum(value.size for _, value in parameters)
        summary['trainable_parameter_names'] = [name for name, _ in parameters]
        print('TRAINABLE_PARAMETERS', summary['trainable_parameters'], flush=True)
        # Save configuration expected by mlx_lm.load(..., adapter_path=...).
        (adapter_path / 'adapter_config.json').write_text(json.dumps(dict(
            fine_tune_type='lora', num_layers=config['num_layers'], lora_parameters=config['lora_parameters']), indent=2))
        events = []

        def record(kind, info):
            monitor.check()
            loss = info['train_loss'] if kind == 'train' else info['val_loss']
            if not math.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            event = dict(kind=kind, wall_seconds=time.perf_counter() - started, **info)
            events.append(event)
            with (out / 'training-events.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
            print('TRAIN_EVENT', json.dumps(event), flush=True)

        class Callback:
            def on_train_loss_report(self, info):
                record('train', info)

            def on_val_loss_report(self, info):
                record('validation', info)

        def checked_batches(*positional, **kwargs):
            for batch in iterate_batches(*positional, **kwargs):
                monitor.check()
                yield batch

        trainer_args = TrainingArgs(batch_size=1, iters=args.microsteps, val_batches=-1,
            steps_per_report=8, steps_per_eval=args.microsteps + 1, steps_per_save=args.microsteps,
            max_seq_length=sequence_cap, adapter_file=str(adapter_path / 'adapters.safetensors'),
            grad_accumulation_steps=8, grad_checkpoint=False)
        mx.reset_peak_memory()
        train_started = time.perf_counter()
        optimizer = optim.Adam(learning_rate=config['learning_rate'])
        train(model, optimizer, train_data, valid_data, loss=assistant_loss,
              args=trainer_args, iterate_batches=checked_batches, training_callback=Callback())
        mx.eval(model.parameters())
        if not all(bool(mx.all(mx.isfinite(value)).item()) for _, value in tree_flatten(model.trainable_parameters())):
            raise FloatingPointError('Nonfinite adapter parameters')
        summary['training_wall_seconds_including_validation'] = time.perf_counter() - train_started
        summary['training_mlx_peak_gb'] = mx.get_peak_memory() / 1e9
        summary['events'] = events
        summary['optimizer_updates'] = int(optimizer.step.item())
        if summary['optimizer_updates'] != budget['optimizer_updates']:
            raise ValueError('Unexpected number of optimizer updates')
        model.eval()
        np.random.seed(config['seed'])
        summary['post_training_validation_loss'] = evaluate(model, valid_data, 1, -1,
            max_seq_length=sequence_cap, loss=assistant_loss)
        if not math.isfinite(summary['post_training_validation_loss']):
            raise FloatingPointError('Nonfinite post-training validation loss')
        weights = adapter_path / 'adapters.safetensors'
        summary['adapter_bytes'] = weights.stat().st_size
        summary['adapter_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
        # Fresh load proves saved adapter compatibility, without retaining two models.
        del parameters, optimizer, model
        mx.clear_cache()
        model, tokenizer = load(str(snapshot), adapter_path=str(adapter_path))
        smoke_cases = [c for c in expanded_cases() if c['order_id'] == 0 and
                       c['condition'] in ('complete_original', 'missing_natural')]
        smoke = []
        for case in smoke_cases:
            monitor.check()
            messages = build_control_messages(case, prompt_version)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pieces, last = [], None
            for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=192, sampler=make_sampler(temp=0)):
                pieces.append(response.text)
                last = response
            output = ''.join(pieces)
            score = score_generation(case, output, last.finish_reason)
            row = dict(case_id=case['id'], group_id=case['group_id'], condition=case['condition'],
                       output=output, finish_reason=last.finish_reason, score=score)
            smoke.append(row)
            with (out / 'adapter-smoke.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
        summary['smoke'] = {condition: dict(n=sum(r['condition'] == condition for r in smoke),
            success=sum(r['condition'] == condition and r['score']['success'] for r in smoke),
            calls=sum(r['condition'] == condition and r['score'].get('action', {}).get('action') == 'call' for r in smoke))
            for condition in ('complete_original', 'missing_natural')}
        summary['smoke_note'] = '48 familiar development inputs, one order; reload/pipeline check, not evidence of generalization'
        summary['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        summary['status'] = 'failed'
        summary['error'] = repr(exc)
        raise
    finally:
        monitor.stop()
        summary['resources'] = monitor.summary()
        if monitor.error and summary['status'] == 'complete':
            summary['status'] = 'failed'
            summary['error'] = monitor.error
        summary['process_max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary['total_wall_seconds'] = time.perf_counter() - started
        summary_path.write_text(json.dumps(summary, indent=2))
        print('SPIKE_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
