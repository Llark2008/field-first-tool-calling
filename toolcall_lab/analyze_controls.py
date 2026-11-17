"""Compare prompt-only controls and lexical source blocking on the same inputs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .analyze_expanded import load_run, summarize
from .controls import guard_counts


def analyze(paths):
    result = {'purpose': 'development controls, not held-out evaluation', 'runs': {}}
    reference = None
    for path in paths:
        meta, rows = load_run(path)
        invariants = {key: meta[key] for key in ('model', 'resolved_revision', 'decoding', 'case_sha256', 'versions')}
        if reference is not None and reference != invariants:
            raise ValueError('Control runs do not have identical model, cases, decoding and dependencies')
        reference = invariants
        variant = meta['prompt_version']
        if variant in result['runs']:
            raise ValueError('Duplicate prompt variant')
        cases = {c['id']: c for c in json.loads((Path(path) / 'cases.json').read_text())}
        groups = defaultdict(list)
        fields = defaultdict(list)
        for row in rows:
            groups[row['condition']].append(row)
            if row['condition'] == 'missing_natural':
                fields[row['missing_field']].append(row)
        counts = {key: guard_counts(value, cases) for key, value in sorted(groups.items())}
        result['runs'][variant] = dict(summary=summarize(rows), guard_by_condition=counts,
                                      guard_natural_by_field={k: guard_counts(v, cases) for k, v in sorted(fields.items())},
                                      mean_prompt_tokens=sum(r['prompt_tokens'] for r in rows) / len(rows))
    result['invariants'] = reference
    # Frozen selection rule: balanced complete/clarify accuracy, then prompt cost.
    selected = max(result['runs'], key=lambda name: (
        (result['runs'][name]['guard_by_condition']['complete_original']['raw_success'] +
         result['runs'][name]['guard_by_condition']['missing_natural']['correct_clarifications']) / 96,
        -result['runs'][name]['mean_prompt_tokens']))
    result['selected_training_prompt'] = selected
    result['source_sha256'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ('controls.py', 'analyze_controls.py', 'evaluation.py', 'generation_scoring.py')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.runs)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'selected': result['selected_training_prompt'],
                      'counts': {k: v['guard_by_condition'] for k, v in result['runs'].items()}}, indent=2))


if __name__ == '__main__':
    main()
