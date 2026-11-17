"""Analyze frozen expanded development runs, preserving task-level pairing."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def _count(rows):
    n = len(rows)
    missing = [r for r in rows if r['gold']['action'] == 'clarify']
    actions = [r['score'].get('action', {}).get('action') for r in rows]
    premature = sum(r['score'].get('action', {}).get('action') == 'call' for r in missing)
    success = sum(bool(r['score']['success']) for r in rows)
    return dict(n=n, success=success, success_rate=success / n if n else None,
                schema_valid=sum(bool(r['score']['schema_valid']) for r in rows),
                truncated=sum(bool(r['score'].get('truncated')) for r in rows),
                clarify_actions=actions.count('clarify'), call_actions=actions.count('call'),
                unsupported_actions=actions.count('unsupported'),
                invalid_actions=sum(a not in ('call', 'clarify', 'unsupported') for a in actions),
                premature_calls=premature, missing_n=len(missing),
                premature_call_rate=premature / len(missing) if missing else None)


def _group_counts(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _count(value) for key, value in sorted(groups.items())}


def summarize(rows):
    result = {'by_condition': _group_counts(rows, 'condition')}
    for field, output in (('order_id', 'by_order'), ('target_tool', 'by_tool'), ('missing_field', 'by_missing_field')):
        groups = defaultdict(list)
        for row in rows:
            if field != 'missing_field' or row['gold']['action'] == 'clarify':
                groups[str(row[field])].append(row)
        result[output] = {key: _group_counts(group, 'condition') for key, group in sorted(groups.items())}
    result['wall_seconds'] = sum(r['seconds'] for r in rows)
    result['median_seconds'] = median(r['seconds'] for r in rows)
    result['max_mlx_peak_gb'] = max((r.get('peak_gb', 0) for r in rows), default=0)
    return result


def paired_effect(rows, condition):
    indexed = {}
    for row in rows:
        key = (row['group_id'], row['condition'], row['order_id'])
        if key in indexed:
            raise ValueError('Duplicate case pairing key: ' + str(key))
        indexed[key] = row
    base = {(group, order): row for (group, name, order), row in indexed.items() if name == 'complete_original'}
    changed = {(group, order): row for (group, name, order), row in indexed.items() if name == condition}
    if not base or base.keys() != changed.keys():
        raise ValueError('Missing or unequal original/renamed pairs')
    differences = defaultdict(list)
    orders = defaultdict(set)
    harmful = beneficial = 0
    for key in sorted(base):
        before, after = int(base[key]['score']['success']), int(changed[key]['score']['success'])
        differences[key[0]].append(after - before)
        orders[key[0]].add(key[1])
        harmful += int(before == 1 and after == 0)
        beneficial += int(before == 0 and after == 1)
    if any(value != {0, 1} for value in orders.values()):
        raise ValueError('Each group must include both controlled orders')
    means = [mean(differences[group]) for group in sorted(differences)]
    rng = random.Random(20261112)
    samples = sorted(mean(rng.choices(means, k=len(means))) * 100 for _ in range(5000))
    return {'n_pairs': len(base), 'n_groups': len(means),
            'original_success': sum(r['score']['success'] for r in base.values()),
            'renamed_success': sum(r['score']['success'] for r in changed.values()),
            'delta_pp': mean(means) * 100,
            'cluster_ci95_pp': [samples[125], samples[4874]],
            'correct_to_wrong': harmful, 'wrong_to_correct': beneficial}


def load_run(path):
    from .generation_scoring import score_generation
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    cases = json.loads((path / 'cases.json').read_text())
    rows = [json.loads(line) for line in (path / 'responses.jsonl').read_text().splitlines()]
    ids = [r['case_id'] for r in rows]
    if metadata['status'] != 'complete' or metadata.get('suite') != 'expanded' or len(rows) != 288:
        raise ValueError('Expected a completed 288-input expanded run')
    if len(set(ids)) != len(ids) or set(ids) != {c['id'] for c in cases}:
        raise ValueError('Missing, duplicated or unexpected responses')
    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata['case_sha256']:
        raise ValueError('Saved case manifest fingerprint mismatch')
    case_map = {c['id']: c for c in cases}
    changed = []
    for row in rows:
        case = case_map[row['case_id']]
        for key in ('group_id', 'condition', 'order_id', 'target_tool', 'missing_field', 'gold', 'request'):
            if row[key] != case[key]:
                raise ValueError('Response metadata differs from case manifest: ' + key)
        current_score = score_generation(case, row['output'], row.get('finish_reason'))
        if bool(current_score['success']) != bool(row['score']['success']):
            changed.append(row['case_id'])
        row['score'] = current_score
    metadata['rescored_success_changes'] = changed
    return metadata, rows


def repeated_failures(prompt_rows):
    result = {}
    for condition in ('complete_semantic_a', 'complete_semantic_b', 'missing_natural', 'missing_explicit', 'missing_semantic_a'):
        groups = defaultdict(list)
        for rows in prompt_rows.values():
            index = {(r['group_id'], r['condition'], r['order_id']): r for r in rows}
            for row in rows:
                if row['condition'] != condition:
                    continue
                if condition.startswith('missing'):
                    bad = row['score'].get('action', {}).get('action') == 'call'
                else:
                    original = index[(row['group_id'], 'complete_original', row['order_id'])]
                    bad = original['score']['success'] and not row['score']['success']
                groups[row['group_id']].append(bool(bad))
        result[condition] = {'all_four_groups': sorted(g for g, flags in groups.items() if len(flags) == 4 and all(flags)),
                             'at_least_three_groups': sorted(g for g, flags in groups.items() if sum(flags) >= 3)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', required=True)
    parser.add_argument('--v2', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    first_meta, first = load_run(args.v1)
    second_meta, second = load_run(args.v2)
    if first_meta['prompt_version'] != 'v1' or second_meta['prompt_version'] != 'v2':
        raise ValueError('Run arguments must correspond to v1 and v2')
    for key in ('case_sha256', 'model', 'resolved_revision', 'decoding', 'source_sha256', 'versions'):
        if first_meta[key] != second_meta[key]:
            raise ValueError('Incomparable runs: ' + key)
    result = {'purpose': 'expanded development diagnosis, not held-out method evaluation',
              'case_sha256': first_meta['case_sha256'], 'model_revision': first_meta['resolved_revision'],
              'n_groups': 24, 'n_generations': 576, 'runs': {}}
    result['rescored_success_changes'] = {'v1': first_meta['rescored_success_changes'], 'v2': second_meta['rescored_success_changes']}
    result['analysis_source_sha256'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('analyze_expanded.py', 'generation_scoring.py', 'evaluation.py')}
    for prompt, rows in (('v1', first), ('v2', second)):
        result['runs'][prompt] = summarize(rows)
        result['runs'][prompt]['rename_effects'] = {condition: paired_effect(rows, condition)
            for condition in ('complete_semantic_a', 'complete_semantic_b')}
    result['repeated_failures'] = repeated_failures({'v1': first, 'v2': second})
    output = Path(args.output)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({prompt: value['by_condition'] for prompt, value in result['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
