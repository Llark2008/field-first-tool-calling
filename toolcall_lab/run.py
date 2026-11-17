"""Run one exploratory baseline. Heavy imports and writes happen only in main."""
import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def runtime_root(source_root, configured=None):
    source = Path(source_root).expanduser().resolve()
    value = Path(configured).expanduser() if configured else Path.home() / '.local/share/local-tool-calling'
    value = value.resolve()
    if value.is_relative_to(source) or source.is_relative_to(value):
        raise ValueError('Runtime must be separate from the source tree (not its child or parent).')
    destinations = ('cache', 'cache/huggingface', 'cache/huggingface/hub', 'cache/huggingface/xet',
                    'cache/pip', 'cache/uv', 'cache/pycache', 'tmp', 'runs', 'venv', 'adapters')
    for relative in destinations:
        resolved = (value / relative).resolve()
        if resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError('Runtime destination redirects into or above source: ' + relative)
    # Existing nested cache or environment links can also redirect future writes.
    for parent, directories, files in os.walk(value, followlinks=False):
        for name in directories + files:
            child = Path(parent) / name
            if child.is_symlink():
                resolved = child.resolve()
                if resolved.is_relative_to(source) or source.is_relative_to(resolved):
                    raise ValueError('Runtime symlink redirects into or above source: ' + str(child))
    return value


def configure_runtime(root):
    root = runtime_root(Path(__file__).resolve().parents[1], str(root))
    paths = {'HF_HOME': root / 'cache/huggingface',
             'HF_HUB_CACHE': root / 'cache/huggingface/hub',
             'HF_XET_CACHE': root / 'cache/huggingface/xet',
             'XDG_CACHE_HOME': root / 'cache',
             'PYTHONPYCACHEPREFIX': root / 'cache/pycache', 'PIP_CACHE_DIR': root / 'cache/pip',
             'UV_CACHE_DIR': root / 'cache/uv', 'TMPDIR': root / 'tmp'}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def select_cases(suite, limit=None):
    if suite == 'pilot':
        from .tasks import pilot_cases
        cases = pilot_cases()
    elif suite == 'expanded':
        from .expanded import expanded_cases
        cases = expanded_cases()
    else:
        raise ValueError('Unknown suite')
    if limit is not None and not 1 <= limit <= len(cases):
        raise ValueError(f'limit must be between 1 and {len(cases)}')
    return cases if limit is None else cases[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['pilot', 'expanded'], default='pilot')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--run-name', default=datetime.now(timezone.utc).strftime('pilot-%Y%m%dT%H%M%SZ'))
    parser.add_argument('--model', default='mlx-community/Qwen3-1.7B-4bit')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--prompt-version', choices=['v1', 'v2', 'no_examples', 'checklist'], default='v2')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a simple directory name')
    try:
        cases = select_cases(args.suite, args.limit)
    except ValueError as error:
        parser.error(str(error))
    source = Path(__file__).resolve().parents[1]
    root = runtime_root(source, os.environ.get('TOOLCALL_RUNTIME_DIR'))
    configure_runtime(root)
    run_dir = root / 'runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from .generation_scoring import score_generation
    from .controls import build_control_messages

    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    source_files = ('tasks.py', 'expanded.py', 'evaluation.py', 'generation_scoring.py', 'controls.py', 'run.py')
    source_hashes = {name: hashlib.sha256((source / 'toolcall_lab' / name).read_bytes()).hexdigest() for name in source_files}
    metadata = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'purpose': 'exploratory pilot, not held-out evaluation', 'model': args.model,
                'requested_revision': args.revision, 'case_sha256': fingerprint,
                'suite': args.suite, 'source_sha256': source_hashes,
                'prompt_version': args.prompt_version,
                'platform': platform.platform(), 'machine': platform.machine(),
                'versions': {name: version(name) for name in ('mlx', 'mlx-lm', 'transformers', 'jsonschema')},
                'decoding': {'temperature': 0.0, 'max_tokens': 192, 'enable_thinking': False},
                'case_count': len(cases)}
    meta_path = run_dir / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    (run_dir / 'cases.json').write_text(json.dumps(cases, indent=2))
    print('Downloading/loading model; output:', run_dir, flush=True)
    snapshot = snapshot_download(args.model, revision=args.revision,
                                 allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
    metadata['resolved_revision'] = Path(snapshot).name
    model, tokenizer = load(snapshot)
    metadata['load_peak_gb'] = mx.get_peak_memory() / 1e9
    meta_path.write_text(json.dumps(metadata, indent=2))
    counts = defaultdict(lambda: {'n': 0, 'success': 0, 'parsed': 0, 'schema_valid': 0})
    total_seconds = 0.0
    with (run_dir / 'responses.jsonl').open('w') as output:
        for index, case in enumerate(cases, 1):
            messages = build_control_messages(case, args.prompt_version)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            mx.reset_peak_memory()
            started = time.perf_counter()
            pieces, last = [], None
            for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=192, sampler=make_sampler(temp=0.0)):
                pieces.append(response.text)
                last = response
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            text = ''.join(pieces)
            stats = {key: getattr(last, key, None) for key in ('prompt_tokens', 'generation_tokens', 'prompt_tps', 'generation_tps', 'finish_reason')}
            score = score_generation(case, text, stats['finish_reason'])
            record = {'case_id': case['id'], 'group_id': case['group_id'], 'category': case['category'],
                      'request': case['request'], 'messages': messages, 'gold': case['gold'],
                      'output': text, 'score': score, 'seconds': elapsed,
                      'peak_gb': mx.get_peak_memory() / 1e9, **stats}
            record.update({key: case[key] for key in ('condition', 'order_id', 'target_tool', 'missing_field') if key in case})
            output.write(json.dumps(record) + '\n')
            output.flush()
            count = counts[case['category']]
            count['n'] += 1
            for key in ('success', 'parsed', 'schema_valid'):
                count[key] += int(score[key])
            print(f"{index}/{len(cases)} {case['id']}: {'PASS' if score['success'] else 'FAIL'} {elapsed:.1f}s", flush=True)
    summary = {'categories': dict(counts), 'total': len(cases),
               'success': sum(value['success'] for value in counts.values()),
               'generation_wall_seconds': total_seconds,
               'limitations': ['exploratory cases', 'small hand-authored fixture', 'paired renamed cases are not independent',
                               'no final-test tools used', 'one deterministic decoding run']}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    metadata['status'] = 'complete'
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
