"""Paired formal data; all generated artifacts live in the external runtime."""
import argparse
import copy
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .tasks import TOOLS, BASE_TOOLS, field, tool
from .controls import build_control_messages
from .fixtures import reference_result
from .formal_phrases import PHRASES, UNSUPPORTED

PROJECTS = 'atlas boreal cedar delta orion lyra zenith nova pulsar quartz raven summit'.split()
ENVS = ['Hopper-v4', 'Walker2d-v4']
ALGOS = ['SAC', 'PPO']
METRICS = ['final_return', 'peak_return']
SPLIT_FAMILIES = {'train': list(range(12)), 'dev': list(range(12, 16)), 'test': list(range(16, 20))}
FINAL_TOOLS = ['group_metric', 'count_missing']
DEV_TOOLS = ['count_runs', 'describe_columns']
EXPLICIT = [
    'I have not supplied {fields}.', 'My request leaves {fields} open.',
    'I still need to choose {fields}.', 'There is no choice yet for {fields}.',
    'Please note that {fields} remains undecided.', 'I have not told you {fields}.',
]
FIELD_WORDS = {'project': 'which project to use', 'environment': 'which environment to use',
    'algorithm': 'which algorithm to use', 'run_id': 'which run to read', 'metric': 'which metric to use',
    'stat': 'which summary statistic to compute', 'order': 'which sorting direction to use',
    'limit': 'how many identifiers to return', 'group_by': 'which grouping dimension to use',
    'table': 'which table to inspect', 'column': 'which column to inspect'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def make_fixture(rng, projects, serial):
    runs, configs, series = [], {}, {}
    values = rng.sample(range(20, 10000), 48)
    for i, (p, e, a, seed) in enumerate(itertools.product(projects, ENVS, ALGOS, [1, 2])):
        rid = f'r{serial:04x}{i:02x}'
        runs.append(dict(run_id=rid, project=p, environment=e, algorithm=a,
                         final_return=values[i], peak_return=10000 + values[24+i], seed=seed))
        configs[rid] = dict(learning_rate=rng.choice([0.0001, 0.0003, 0.001]), seed=seed)
        series[rid] = [rng.randrange(10, 200), rng.randrange(200, 400), values[i]]
    rng.shuffle(runs)
    # Both tables have the same usable column names but different missingness.
    tables = {t: [dict(run_id=r['run_id'], score=None if (j + ti) % 3 == 0 else r['final_return'],
                           duration=None if (j + ti) % 4 == 0 else j * 10 + 7)
                  for j, r in enumerate(runs[:8])] for ti, t in enumerate(['experiments', 'evaluations'])}
    return dict(runs=runs, configs=configs, series=series, tables=tables)


def schema_for(name, projects):
    if name == 'group_metric':
        schema = tool(name, 'Group matching runs by one dimension and return the mean scalar metric for each group.',
            {'project': copy.deepcopy(TOOLS['list_runs']['parameters']['properties']['project']),
             'metric': copy.deepcopy(TOOLS['rank_runs']['parameters']['properties']['metric']),
             'group_by': field('Dimension whose distinct values form groups.', enum=['algorithm', 'environment']),
             'environment': copy.deepcopy(TOOLS['list_runs']['parameters']['properties']['environment'])},
            ['project', 'metric', 'group_by'])
    elif name == 'count_missing':
        schema = tool(name, 'Count null (missing) cells in a named column of one table.',
            {'table': field('Table identifier.', enum=['experiments', 'evaluations']),
             'column': field('Column whose null cells are counted.', enum=['score', 'duration'])}, ['table', 'column'])
    else:
        schema = copy.deepcopy(TOOLS[name])
    props = schema['parameters']['properties']
    if 'project' in props: props['project']['enum'] = list(projects)
    if 'table' in props: props['table']['enum'] = ['experiments', 'evaluations']
    return schema


def render(name, args, family):
    a = args
    scope = ', '.join(x for x in [f'in project {a["project"]}' if 'project' in a else '',
        f'on {a["environment"]}' if 'environment' in a else '',
        f'using {a["algorithm"]}' if 'algorithm' in a else ''] if x) or 'in the experiment collection'
    metric = a.get('metric', ['performance', 'return', 'score', 'the performance measure'][family % 4])
    stat = {'mean': 'the mean of', 'min': 'the minimum of', 'max': 'the maximum of'}.get(a.get('stat'),
        ['a summary of', 'an aggregate of', 'a descriptive statistic of', 'a single summary value of'][family % 4])
    rankrule = 'by ' + metric
    if 'order' in a: rankrule += ', ' + ('lowest first' if a['order'] == 'asc' else 'highest first')
    if 'limit' in a: rankrule += f', returning {a["limit"]} identifiers'
    # Entailed limit/direction remains a complete request; do not use it on omissions.
    if 'order' in a and 'limit' in a and family % 3 == 0:
        rankrule = f'with the {a["limit"]} ' + ('lowest' if a['order'] == 'asc' else 'highest') + ' values of ' + metric + ', in that order'
    n = a.get('max_points')
    slots = dict(scope=scope, scope_cap=scope[0].upper()+scope[1:], metric=metric,
        statmetric=stat+' '+metric, rankrule=rankrule, algorithms=' and '.join(a.get('algorithms', ALGOS)),
        run='run '+a['run_id'] if 'run_id' in a else 'the run I want to inspect',
        seriesmetric=a.get('metric', 'the requested metric'), points=f'the first {n}' if n else 'the',
        point_tail=f', limited to the first {n} points' if n else '')
    if name in PHRASES: return PHRASES[name][family].format(**slots)
    # Only dev/test tools use these independently assigned transfer scaffolds.
    variants = {
        'count_runs': [f'Count the recorded runs {scope}.', f'How many runs are present {scope}?',
            f'I need the number of runs {scope}.', f'Give the run count {scope}.'],
        'describe_columns': [f'List the ordered column names in {a.get("table", "the table I am inspecting")}.',
            f'What columns belong to {a.get("table", "the table I need")}, in stored order?',
            f'Return the schema field names of {a.get("table", "my table")}.',
            f'Inspect {a.get("table", "the table of interest")} and enumerate its columns in order.'],
        'group_metric': [f'Group runs {scope} by {a.get("group_by", "a grouping dimension")} and calculate mean {metric} in each group.',
            f'For runs {scope}, provide average {metric} per {a.get("group_by", "group")}.',
            f'Break runs {scope} into {a.get("group_by", "chosen")} groups; return their mean {metric}.',
            f'Show mean {metric} by {a.get("group_by", "grouping dimension")} {scope}.'],
        'count_missing': [f'Count missing cells in column {a.get("column", "of interest")} of {a.get("table", "my table")}.',
            f'How many null values are in {a.get("table", "the table")}, column {a.get("column", "to be checked")}?',
            f'For {a.get("table", "the chosen table")}, return the null-cell count in {a.get("column", "the column I need")}.',
            f'Inspect {a.get("table", "a table")} column {a.get("column", "of interest")} and count its missing entries.'],
    }
    return variants[name][family % 4]


def omission_specs(name, count):
    if name == 'summarize_metric':
        doubles = [('metric', 'project'), ('stat', 'environment'), ('metric', 'algorithm'), ('stat', 'project'), ('metric', 'stat')]
        return [list(doubles[i % 5]) if i < 50 else [['metric'], ['stat']][i % 2] for i in range(count)]
    if name == 'rank_runs':
        doubles = [('order', 'project'), ('limit', 'environment'), ('metric', 'order'), ('metric', 'limit'), ('order', 'limit')]
        return [list(doubles[i % 5]) if i < 50 else [['metric'], ['order'], ['limit']][i % 3] for i in range(count)]
    if name == 'compare_algorithms':
        return [['metric', ['project', 'environment'][i % 2]] if i < 20 else ['metric'] if i < 40 else [['project'], ['environment']][i % 2] for i in range(count)]
    if name in ('list_runs', 'count_runs'): return [['project']] * count
    if name in ('get_run_config', 'get_metric_series'): return [['run_id']] * count
    if name == 'describe_columns': return [['table']] * count
    if name == 'group_metric': return [[['project', 'metric', 'group_by'][i % 3]] for i in range(count)]
    if name == 'count_missing': return [[['table', 'column'][i % 2]] for i in range(count)]
    raise ValueError(name)


def rename(case):
    c = copy.deepcopy(case)
    c['id'] += '-alias'
    c['condition'] = 'renamed_complete' if c['gold']['action'] == 'call' else 'renamed_natural'
    c['bindings'] = {}
    for s in c['tools']:
        old = s['name']
        new = {'list_runs': 'enumerate_trials', 'summarize_metric': 'reduce_trial_measure',
               'rank_runs': 'ordered_trial_subset', 'compare_algorithms': 'contrast_learners',
               'get_run_config': 'read_training_setup', 'get_metric_series': 'fetch_recorded_curve'}[old]
        mapping = {'project': 'study_key', 'environment': 'task_env', 'algorithm': 'learner',
            'metric': 'measure', 'stat': 'reduction', 'order': 'direction', 'limit': 'take_count',
            'algorithms': 'learner_pair', 'run_id': 'trial_key', 'max_points': 'point_cap'}
        p = s['parameters']
        c['bindings'][new] = dict(name=old, arguments={mapping[k]: k for k in p['properties']})
        s['name'] = new
        p['properties'] = {mapping[k]: v for k, v in p['properties'].items()}
        p['required'] = [mapping[k] for k in p['required']]
        if old == c['target_tool']:
            if c['gold']['action'] == 'call':
                c['gold']['name'] = new
                c['gold']['arguments'] = {mapping[k]: v for k, v in c['gold']['arguments'].items()}
            else: c['gold']['missing_fields'] = sorted(mapping[k] for k in c['gold']['missing_fields'])
    return c


def build_formal_datasets(seed=20261113):
    rng = random.Random(seed)
    data = {'A': [], 'B': [], 'dev': [], 'test': []}
    serial = 0
    for split in ['train', 'dev', 'test']:
        dest = []
        tools = BASE_TOOLS + (DEV_TOOLS if split == 'dev' else FINAL_TOOLS if split == 'test' else [])
        counts = [100]*6 if split == 'train' else [17, 17, 17, 17, 16, 16] + ([15, 15] if split == 'dev' else [20, 20])
        for name, count in zip(tools, counts):
            miss_count = {'list_runs': 80, 'summarize_metric': 100, 'rank_runs': 100, 'compare_algorithms': 80,
                          'get_run_config': 60, 'get_metric_series': 60}.get(name, count) if split == 'train' else count
            # Sample the whole omission pool for small dev/test sets; taking its
            # prefix would accidentally make every semantic omission a double.
            pool_count = {'summarize_metric': 100, 'rank_runs': 100, 'compare_algorithms': 80}.get(name, miss_count)
            omit = omission_specs(name, miss_count) if split == 'train' else rng.sample(omission_specs(name, pool_count), miss_count)
            rng.shuffle(omit)
            selected = rng.sample(range(count), miss_count)
            omap = dict(zip(selected, omit))
            families = [SPLIT_FAMILIES[split][i % len(SPLIT_FAMILIES[split])] for i in range(count)]
            rng.shuffle(families)
            positions = [i % 3 for i in range(count)]; rng.shuffle(positions)
            # Separate shuffled balanced decks prevent metric/order and metric/stat coupling.
            rankdeck = list(itertools.islice(itertools.cycle(list(itertools.product(METRICS, ['asc', 'desc']))), count)); rng.shuffle(rankdeck)
            statdeck = list(itertools.islice(itertools.cycle(list(itertools.product(METRICS, ['mean', 'min', 'max']))), count)); rng.shuffle(statdeck)
            limits = [1+i%3 for i in range(count)]; rng.shuffle(limits)
            for i in range(count):
                serial += 1
                family = families[i]
                projects = rng.sample(PROJECTS, 3)
                p = projects[positions[i]]
                fixture = make_fixture(rng, projects, serial)
                run_catalog = rng.sample(list(fixture['configs']), 6)
                run = rng.choice(run_catalog)
                e, a = rng.choice(ENVS), rng.choice(ALGOS)
                metric = rng.choice(METRICS)
                args = {
                    'list_runs': dict(project=p, **({} if i % 2 == 0 else dict(environment=e, algorithm=a))),
                    'summarize_metric': dict(project=p, environment=e, algorithm=a, metric=statdeck[i][0], stat=statdeck[i][1]),
                    'rank_runs': dict(project=p, environment=e, metric=rankdeck[i][0], order=rankdeck[i][1], limit=limits[i]),
                    'compare_algorithms': dict(project=p, environment=e, metric=metric, algorithms=rng.sample(ALGOS, 2)),
                    'get_run_config': dict(run_id=run),
                    'get_metric_series': dict(run_id=run, metric='episode_return', **({} if i % 2 == 0 else dict(max_points=limits[i]))),
                    'count_runs': dict(project=p, **({} if i % 2 == 0 else dict(environment=e))),
                    'describe_columns': dict(table=rng.choice(['experiments', 'evaluations'])),
                    'group_metric': dict(project=p, metric=metric, group_by=rng.choice(['algorithm', 'environment'])),
                    'count_missing': dict(table=rng.choice(['experiments', 'evaluations']), column=rng.choice(['score', 'duration'])),
                }[name]
                candidate_names = [name] + rng.sample([x for x in BASE_TOOLS if x != name], 2)
                rng.shuffle(candidate_names)
                group = f'{split}-{name}-{i:03d}'
                gold = dict(action='call', name=name, arguments=args)
                core = name in BASE_TOOLS
                row = dict(id=group+'-complete', group_id=group, intent_group=group,
                    template_group=f'{split}:{name}:f{family}', split=split, target_tool=name,
                    condition='complete' if core else 'newtool_complete',
                    request=render(name, args, family),
                    context='Known project IDs: '+', '.join(projects)+'. Known run IDs: '+', '.join(run_catalog)+
                        '. Known environments: Hopper-v4, Walker2d-v4. Algorithms: SAC, PPO. Tables: experiments, evaluations. These inventories are not defaults; no user selection has been established.',
                    tools=[schema_for(n, projects) for n in candidate_names], fixture=fixture,
                    gold=gold, expected_result=reference_result(fixture, gold),
                    missing_fields=[], anti_overclarify=name=='get_run_config' or name in ('list_runs', 'get_metric_series') and i%2==0 or name=='rank_runs' and family%3==0,
                    supplied_arguments=args)
                dest.append(row)
                if i in omap:
                    absent = sorted(omap[i])
                    remaining = {k: v for k, v in args.items() if k not in absent}
                    missing = copy.deepcopy(row)
                    missing.update(id=group+'-natural', condition='natural' if core else 'newtool_natural',
                        request=render(name, remaining, family), gold=dict(action='clarify', missing_fields=absent),
                        expected_result=None, missing_fields=absent, anti_overclarify=False, supplied_arguments=remaining)
                    dest.append(missing)
        if split != 'train':
            naturals = [r for r in dest if r['condition']=='natural']
            for original in rng.sample(naturals, 20):
                row = copy.deepcopy(original)
                row['id'] = row['group_id']+'-explicit'
                row['condition'] = 'explicit'
                row['request'] += ' '+EXPLICIT[rng.randrange(len(EXPLICIT))].format(fields=' and '.join(FIELD_WORDS[f] for f in row['missing_fields']))
                dest.append(row)
        for i in range(120 if split == 'train' else 20):
            base = copy.deepcopy(rng.choice([r for r in dest if r['gold']['action']=='call' and r['target_tool'] in BASE_TOOLS]))
            family = SPLIT_FAMILIES[split][i % len(SPLIT_FAMILIES[split])]
            group = f'{split}-unsupported-{i:03d}'
            base.update(id=group, group_id=group, intent_group=group, template_group=f'{split}:unsupported:f{family}',
                condition='unsupported', gold={'action':'unsupported'}, expected_result=None,
                missing_fields=[], supplied_arguments={}, anti_overclarify=False, target_tool=None,
                request=UNSUPPORTED[family].format(project=rng.choice(PROJECTS), run=rng.choice(list(base['fixture']['configs']))))
            dest.append(base)
        if split == 'test':
            complete = rng.sample([r for r in dest if r['condition']=='complete'], 80)
            ids = {r['group_id'] for r in complete}
            naturals = [r for r in dest if r['condition']=='natural' and r['group_id'] in ids]
            dest.extend(rename(r) for r in complete+naturals)
        rng.shuffle(dest)
        if split == 'train':
            data['B'] = dest
            data['A'] = copy.deepcopy(dest)
            for row in data['A']:
                if row['gold']['action']=='clarify':
                    row['request'] += ' '+EXPLICIT[rng.randrange(len(EXPLICIT))].format(fields=' and '.join(FIELD_WORDS[f] for f in row['missing_fields']))
        else: data[split] = dest
    return data


def audit_formal_datasets(data):
    from .evaluation import score_response
    assert {k:len(v) for k,v in data.items()} == dict(A=1200, B=1200, dev=300, test=480)
    assert Counter(r['gold']['action'] for r in data['B']) == dict(call=600, clarify=480, unsupported=120)
    for a, b in zip(data['A'], data['B']):
        aa, bb = copy.deepcopy(a), copy.deepcopy(b)
        if a['gold']['action']=='clarify': assert aa.pop('request') != bb.pop('request')
        assert aa == bb
    group_sets = [{r['template_group'] for r in data[s]} for s in ['B','dev','test']]
    assert not any(a & b for a,b in itertools.combinations(group_sets,2))
    omission_sizes = Counter(str(len(r['missing_fields'])) for r in data['B'] if r['gold']['action']=='clarify')
    assert omission_sizes == {'1':360, '2':120}
    semantic = sum(bool(set(r['missing_fields']) & {'metric','stat','order','limit'}) for r in data['B'])
    assert semantic >= 240
    cross, positions, tool_positions, family_targets = Counter(), Counter(), Counter(), defaultdict(set)
    for split in ['B','dev','test']:
        assert len({r['id'] for r in data[split]}) == len(data[split])
        for row in data[split]:
            if split != 'test': assert not set(FINAL_TOOLS) & {s['name'] for s in row['tools']}
            score = score_response(row, json.dumps(row['gold']))
            assert score['success'], (row['id'], score)
            # Label completeness checked against public schema and supplied slots, not enum position.
            if row['condition'].startswith('renamed'): continue
            if row['gold']['action'] != 'unsupported':
                schema = next(s for s in row['tools'] if s['name']==row['target_tool'])
                required = {k for k in schema['parameters']['required'] if 'default' not in schema['parameters']['properties'][k]}
                assert set(row['missing_fields']) == required - set(row['supplied_arguments'])
            if split=='B' and row['gold']['action']=='call':
                name, args = row['gold']['name'], row['gold']['arguments']
                tool_positions[row['tools'].index(schema)] += 1
                if name=='rank_runs': cross[args['metric']+'/'+args['order']] += 1
                if 'project' in args:
                    positions[schema['parameters']['properties']['project']['enum'].index(args['project'])] += 1
                    family_targets[row['template_group']].add(args['project'])
    assert len(cross)==4 and min(cross.values())==25
    assert set(positions)=={0,1,2} and max(positions.values())<200
    assert min(len(v) for v in family_targets.values())>=3
    anti = sum(r['anti_overclarify'] for r in data['B'] if r['gold']['action']=='call')
    assert anti>=180
    return dict(status='passed', review_attribution='programmatic audit; AI-assisted semantic review tracked separately',
        counts={s:dict(Counter(r['condition'] for r in rows)) for s,rows in data.items()},
        omission_sizes=dict(omission_sizes), semantic_omissions=semantic, anti_overclarify=anti,
        metric_order=dict(cross), project_catalog_positions=dict(positions), candidate_tool_positions=dict(tool_positions),
        min_project_values_per_training_family=min(map(len,family_targets.values())),
        groups={s:dict(intents=len({r['intent_group'] for r in rows}), families=len({r['template_group'] for r in rows})) for s,rows in data.items()},
        training_phrase_families_per_tool=12, generation_note='Authored scaffolds with shared slot grammar; synthetic scoped benchmark, not natural human distribution.')


def training_row(case):
    return dict(id=case['id'], messages=build_control_messages(case, 'v2')+
                [dict(role='assistant', content=json.dumps(case['gold'], separators=(',', ':')))])


def write_dataset(root, data):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    audit = audit_formal_datasets(data)
    manifest = dict(seed=20261113, prompt_version='v2', status='audited_model_free', files={}, audit=audit)
    for split, rows in data.items():
        path = root / (split+'.jsonl')
        path.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
        manifest['files'][path.name] = dict(n=len(rows), sha256=sha(path))
        if split in ('A','B'):
            path = root / (split+'-sft.jsonl')
            path.write_text(''.join(json.dumps(training_row(r), separators=(',', ':'))+'\n' for r in rows))
            manifest['files'][path.name] = dict(n=len(rows), sha256=sha(path))
    (root/'SEALED_TEST.json').write_text(json.dumps(dict(status='sealed', sha256=manifest['files']['test.jsonl']['sha256'], rule='Open only after common-checkpoint preservation gate passes.'), indent=2))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); args=p.parse_args()
    root = Path(args.output).resolve()
    from .run import runtime_root
    import os
    runtime = runtime_root(Path(__file__).resolve().parents[1], os.environ.get('TOOLCALL_RUNTIME_DIR'))
    if not root.is_relative_to(runtime/'datasets'): p.error('Use external runtime datasets directory')
    manifest=write_dataset(root, build_formal_datasets()); print(json.dumps(manifest['audit'], indent=2))


if __name__=='__main__': main()
