"""Draft paired SFT data; template families split before any value expansion.

Synthetic catalogs deliberately differ from the diagnostic fixture. Labels are
structured decisions, not results obtained by executing the evaluation engine.
This is a training/validation draft, not a final benchmark or real experiment data.
"""
import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .controls import build_control_messages
from .run import configure_runtime, runtime_root
from .tasks import BASE_TOOLS, TOOLS


TEMPLATES = {
    'summarize_metric': [
        'For {scope}, {reduce} {metric} over {cohort}.',
        'Working with {scope}: use {cohort} and {reduce} {metric}.',
        'Please {reduce} {metric} across {cohort}, within {scope}.',
        'The request concerns {scope} and {cohort}; {reduce} {metric}.',
        'I need an aggregate for {cohort} in {scope}: {reduce} {metric}.',
    ],
    'rank_runs': [
        'From {scope}{env}, return {count} {ids} sorted by {metric}{order}.',
        'Use {metric} to rank runs from {scope}{env}{order}; give {count} {ids}.',
        'List {count} {ids} from {scope}{env}, ordered by {metric}{order}.',
        'For {scope}{env}, produce {count} {ids} in {metric} order{order}.',
        'My ranking request: {scope}{env}; sort on {metric}{order} and return {count} {ids}.',
    ],
    'list_runs': [
        'List run identifiers from {scope}{filters}.',
        'Find the run IDs belonging to {scope}{filters}.',
        'I need the experiment identifiers in {scope}{filters}.',
        'Retrieve a list of run IDs for {scope}{filters}.',
        'Which run identifiers are registered under {scope}{filters}?',
    ],
    'compare_algorithms': [
        'Compare SAC and PPO using their mean {metric} in {scope}{env}.',
        'For {scope}{env}, contrast SAC versus PPO by mean {metric}.',
        'Show the average {metric} for each of SAC and PPO within {scope}{env}.',
        'Within {scope}{env}, give the mean {metric} separately for SAC and PPO.',
        'How do SAC and PPO compare on mean {metric} for {scope}{env}?',
    ],
    'get_run_config': [
        'Read the saved training settings of {run}.',
        'Which hyperparameters were recorded for {run}?',
        'Display the configuration used to train {run}.',
        'Retrieve the training configuration associated with {run}.',
        'I would like to inspect the saved hyperparameter record for {run}.',
    ],
    'get_metric_series': [
        'Read {points}episode_return observations recorded for {run}.',
        'Fetch {points}values from the episode_return trace of {run}.',
        'Display {points}recorded episode_return readings for {run}.',
        'For {run}, load {points}samples from the episode_return series.',
        'Can you retrieve {points}entries of the episode_return history for {run}?',
    ],
}
MISSING_OPTIONS = {
    'summarize_metric': ['project', 'environment', 'algorithm', 'metric', 'stat'],
    'rank_runs': ['project', 'environment', 'metric', 'order', 'limit'],
    'list_runs': ['project'], 'compare_algorithms': ['project', 'environment', 'metric'],
    'get_run_config': ['run_id'], 'get_metric_series': ['run_id'],
}
UNSUPPORTED = [
    'Delete {run} permanently.', 'Launch another training run for {project}.',
    'Set the learning rate of {run} to 0.02.', 'Send the results for {project} by email.',
    'Erase the stored records for {run}.',
]


def render_request(tool, family, supplied):
    s = supplied
    metric = {'final_return': 'final return', 'peak_return': 'peak return'}.get(s.get('metric'), 'return score')
    stat = {'mean': 'calculate the average of', 'min': 'find the minimum of', 'max': 'find the maximum of'}
    order = {'asc': ', from lowest to highest', 'desc': ', from highest to lowest'}.get(s.get('order'), '')
    env = ' on ' + s['environment'] if 'environment' in s else ''
    cohort = (s['algorithm'] + ' trials' if 'algorithm' in s else 'the trials of my algorithm') + env
    filters = (' with algorithm ' + s['algorithm'] if 'algorithm' in s else '') + env
    values = dict(scope=s.get('project', 'my project'), env=env, metric=metric,
                  reduce=stat.get(s.get('stat'), 'summarize'), cohort=cohort, filters=filters,
                  count={1: 'one', 2: 'two', 3: 'three'}.get(s.get('limit'), 'a limited number of'),
                  ids='run ID' if s.get('limit') == 1 else 'run IDs',
                  order=order, run=s.get('run_id', 'the run I am working on'),
                  points=('the first ' + {1: 'one', 2: 'two', 3: 'three'}[s['max_points']] + ' ')
                  if 'max_points' in s else '')
    return TEMPLATES[tool][family].format(**values)


def _intent(tool, family, index):
    project = f'{("cedar", "delta", "orion", "lyra", "zenith")[family]}-{index}'
    run_id = f'trial-{family}-{index:02d}'
    common = dict(project=project, environment=('Hopper-v4', 'Walker2d-v4')[index % 2],
                  algorithm=('SAC', 'PPO')[(index // 2) % 2])
    metric = ('final_return', 'peak_return')[index % 2]
    if tool == 'summarize_metric':
        args = dict(**common, metric=metric, stat=('mean', 'min', 'max')[index % 3])
    elif tool == 'rank_runs':
        args = dict(project=project, environment=common['environment'], metric=metric,
                    order=('asc', 'desc')[index % 2], limit=index % 3 + 1)
    elif tool == 'list_runs':
        args = {'project': project}
        if index % 3 != 0:
            args['algorithm'] = common['algorithm']
        if index % 3 == 2:
            args['environment'] = common['environment']
    elif tool == 'compare_algorithms':
        args = dict(project=project, environment=common['environment'], metric=metric, algorithms=['SAC', 'PPO'])
    elif tool == 'get_run_config':
        args = dict(run_id=run_id)
    else:
        args = dict(run_id=run_id, metric='episode_return')
        if index % 2:
            args['max_points'] = index % 3 + 1
    return project, run_id, args


def _tools_and_context(target, project, run_id, seed):
    rng = random.Random(seed)
    names = [target] + rng.sample([t for t in BASE_TOOLS if t != target], 2)
    rng.shuffle(names)
    tools = [copy.deepcopy(TOOLS[name]) for name in names]
    for schema in tools:
        if 'project' in schema['parameters']['properties']:
            schema['parameters']['properties']['project']['enum'] = [project, 'reserve-project']
    context = (f'Known projects: {project}, reserve-project. Known run IDs: {run_id}, reserve-run. '
               'Environments: Hopper-v4, Walker2d-v4. Algorithms: SAC, PPO. '
               'These are inventories only, not a selected project, run, environment or algorithm.')
    return tools, context


def build_datasets(prompt_version):
    result = {name: {'train': [], 'valid': []} for name in ('explicit', 'natural')}
    for family in range(5):
        split = 'train' if family < 4 else 'valid'
        for target in BASE_TOOLS:
            for i in range(10):
                project, run_id, arguments = _intent(target, family, i)
                group = f'{target}-f{family}-i{i}'
                tools, context = _tools_and_context(target, project, run_id, group)
                options = MISSING_OPTIONS[target]
                missing = [options[i % len(options)]]
                if i % 4 == 0 and len(options) > 1:
                    missing.append(options[(i + 1) % len(options)])
                for mode in ('complete', 'missing'):
                    supplied = {k: v for k, v in arguments.items() if mode == 'complete' or k not in missing}
                    request = render_request(target, family, supplied)
                    gold = (dict(action='call', name=target, arguments=arguments) if mode == 'complete'
                            else dict(action='clarify', missing_fields=sorted(missing)))
                    for variant in result:
                        row = dict(id=group + '-' + mode, template_group=f'{target}-f{family}',
                                   intent_group=group, target_tool=target, request=request, tools=copy.deepcopy(tools),
                                   context=context, gold=copy.deepcopy(gold), supplied_fields=supplied)
                        if mode == 'missing' and variant == 'explicit':
                            row['request'] += ' I have not provided: ' + ', '.join(sorted(missing)) + '.'
                        row['messages'] = build_control_messages(row, prompt_version) + [
                            {'role': 'assistant', 'content': json.dumps(gold, separators=(',', ':'))}]
                        result[variant][split].append(row)
        for i in range(12):
            project, run_id, _ = _intent('get_run_config', family, i)
            tools, context = _tools_and_context(BASE_TOOLS[i % 6], project, run_id, f'unsupported-{family}-{i}')
            row = dict(id=f'unsupported-f{family}-i{i}', template_group=f'unsupported-f{family}',
                       intent_group=f'unsupported-f{family}-i{i}', target_tool=None, tools=tools, context=context,
                       request=UNSUPPORTED[family].format(project=project, run=run_id),
                       gold={'action': 'unsupported'}, supplied_fields={})
            row['messages'] = build_control_messages(row, prompt_version) + [
                {'role': 'assistant', 'content': '{"action":"unsupported"}'}]
            for variant in result:
                result[variant][split].append(copy.deepcopy(row))
    for variant in result.values():
        for rows in variant.values():
            random.Random(20261113).shuffle(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-version', required=True, choices=['v2', 'no_examples', 'checklist'])
    parser.add_argument('--name', default='sft-draft-20261113')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a simple directory name')
    import os
    root = runtime_root(Path(__file__).resolve().parents[1], os.environ.get('TOOLCALL_RUNTIME_DIR'))
    configure_runtime(root)
    out = root / 'datasets' / args.name
    out.mkdir(parents=True, exist_ok=False)
    data = build_datasets(args.prompt_version)
    manifest = {'purpose': 'draft training/validation only; no final-test data',
                'prompt_version': args.prompt_version, 'seed': 20261113, 'variants': {}}
    for variant, splits in data.items():
        dest = out / variant
        dest.mkdir()
        manifest['variants'][variant] = {}
        for split, rows in splits.items():
            raw = ''.join(json.dumps({'messages': r['messages']}) + '\n' for r in rows)
            (dest / (split + '.jsonl')).write_text(raw)
            (dest / (split + '-manifest.json')).write_text(json.dumps(rows, indent=2))
            manifest['variants'][variant][split] = dict(n=len(rows),
                actions=dict(Counter(r['gold']['action'] for r in rows)),
                template_groups=sorted({r['template_group'] for r in rows}),
                sha256=hashlib.sha256(raw.encode()).hexdigest())
    manifest['source_sha256'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ('training_data.py', 'controls.py', 'tasks.py')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(out), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
