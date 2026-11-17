"""Recompute published metrics on a CPU, without model weights or private runs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from experiments.fields_epochs.report import cell_scores
from experiments.fields_sft.data import audit_case, messages, target, digest
from experiments.fields_sft.scoring import score
from experiments.field_action_diagnostics.run import summarize

ROOT = Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def validate_alignment(cases, rows):
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids) or ids != [r['case_id'] for r in rows]:
        raise ValueError('Missing, duplicate, reordered or extra prediction rows')
    for c, r in zip(cases, rows):
        if not isinstance(r['output'], str) or r['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Invalid generation record')
        if 'prompt_sha256' in r and r['prompt_sha256'] != digest(messages(c, 'F')):
            raise ValueError('Prompt identity changed')

def verify_manifest(root):
    manifest = json.loads((root/'results/manifest.json').read_text())
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Manifest path outside repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Published file changed: ' + name)
    provenance = json.loads((root/'results/provenance.json').read_text())
    for name, expected in provenance['original_source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original experiment source changed: ' + name)
    return manifest

def verify(root=ROOT):
    root = Path(root)
    verify_manifest(root)
    cases = {s: read_rows(root/f'data/{s}-cases.jsonl') for s in ('train', 'dev', 'seen')}
    for split, rows in cases.items():
        expected = {'complete': 300, 'natural': 240, 'unsupported': 60} if split == 'train' else {'complete': 60, 'natural': 48, 'unsupported': 12}
        if Counter(r['condition'] for r in rows) != expected:
            raise ValueError('Split composition changed')
        for c in rows: audit_case(c)
    train_ids = {c['id'] for c in cases['train']}
    if not {c['id'] for c in cases['seen']} <= train_ids or train_ids & {c['id'] for c in cases['dev']}:
        raise ValueError('Split identity changed')
    sft = read_rows(root/'data/sft-F.jsonl')
    if [r['id'] for r in sft] != [c['id'] for c in cases['train']]:
        raise ValueError('Training exposure/order changed')
    for c, r in zip(cases['train'], sft):
        if r['messages'] != messages(c, 'F') + [{'role': 'assistant', 'content': target(c, 'F')[0]}]:
            raise ValueError('Training target or prompt changed')
    expected = json.loads((root/'results/epochs.json').read_text())
    actual = {}
    for arm in ('A', 'B'):
        for epoch in (1, 2, 3):
            for split in ('dev', 'seen'):
                key = f'{arm}-{epoch}-{split}'
                raw = read_rows(root/f'results/predictions/free-{key}.jsonl')
                validate_alignment(cases[split], raw)
                actual[key] = dict(arm=arm, epoch=epoch, split=split, **cell_scores(cases[split], raw))
                if actual[key] != expected['cells'][key]:
                    raise ValueError('Rescoring mismatch: ' + key)
    baselines = {}
    for arm, name in [('J', 'internal'), ('F', 'visible')]:
        raw = read_rows(root/f'results/predictions/dev-{arm}-0.jsonl')
        validate_alignment(cases['dev'], raw)
        rows = [dict(case_id=c['id'], **score(c, r['output'], r['finish_reason'], arm)) for c, r in zip(cases['dev'], raw)]
        baselines[name] = summarize(cases['dev'], rows)
        if arm == 'F': baselines['visible_D'] = cell_scores(cases['dev'], raw)['D']
    baseline_path = root/'results/baselines.json'
    if baselines != json.loads(baseline_path.read_text()):
        raise ValueError('Base-model rescoring mismatch')
    return dict(cells=actual, baselines=baselines)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = verify()
    print('Verified: 600 training examples, fixed seen120/dev120, 1,680 generations, 12 epoch cells and 2 base controls.')
    print('arm epoch split | model C/60 Q/48 F/60 total/120 | converted C/60 Q/48 F/60 total/120')
    for c in result['cells'].values():
        values = [' '.join(str(c[mode][k]) for k in ('C', 'Q', 'F', 'success')) for mode in ('model', 'D')]
        print(f"{c['arm']} {c['epoch']} {c['split']} | " + ' | '.join(values))

if __name__ == '__main__': main()
