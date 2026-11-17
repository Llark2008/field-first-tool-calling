"""Hand-authored exploratory tasks. These are NOT a held-out benchmark."""
import copy
import json
import random


def field(description, kind='string', **kwargs):
    return {'type': kind, 'description': description, **kwargs}


PROJECT = field('Project identifier; never assume a project.', enum=['atlas', 'boreal'])
ENV = field('Environment identifier.', enum=['Hopper-v4', 'Walker2d-v4'])
ALGO = field('Algorithm.', enum=['SAC', 'PPO'])
METRIC = field('final_return is last evaluation return; peak_return is best evaluation return.', enum=['final_return', 'peak_return'])
STAT = field('Statistic across matching runs.', enum=['mean', 'min', 'max'])
RUN_ID = field('Existing run identifier.')


def tool(name, description, properties, required):
    return {'name': name, 'description': description, 'parameters': {
        'type': 'object', 'properties': properties, 'required': required,
        'additionalProperties': False}}


TOOLS = {
    'list_runs': tool('list_runs', 'Return matching run identifiers, sorted by identifier. No aggregation or ranking.',
                      {'project': PROJECT, 'environment': ENV, 'algorithm': ALGO}, ['project']),
    'summarize_metric': tool('summarize_metric', 'Compute one statistic of one scalar metric across matching runs.',
        {'project': PROJECT, 'environment': ENV, 'algorithm': ALGO, 'metric': METRIC, 'stat': STAT},
        ['project', 'environment', 'algorithm', 'metric', 'stat']),
    'rank_runs': tool('rank_runs', 'Return run identifiers ranked by a scalar metric. Order and limit must be specified.',
        {'project': PROJECT, 'environment': ENV, 'metric': METRIC,
         'order': field('Ascending or descending.', enum=['asc', 'desc']),
         'limit': field('Number of results.', 'integer', minimum=1, maximum=5)},
        ['project', 'environment', 'metric', 'order', 'limit']),
    'compare_algorithms': tool('compare_algorithms', 'Return mean scalar metric for each of two algorithms in one environment.',
        {'project': PROJECT, 'environment': ENV, 'metric': METRIC,
         'algorithms': field('Exactly two distinct algorithms to compare.', 'array', items=ALGO, minItems=2, maxItems=2, uniqueItems=True)},
        ['project', 'environment', 'metric', 'algorithms']),
    'get_run_config': tool('get_run_config', 'Read a single run training configuration. Does not start or modify training.',
                          {'run_id': RUN_ID}, ['run_id']),
    'get_metric_series': tool('get_metric_series', 'Read recorded evaluation-return values for one run; no new statistics or plotting.',
        {'run_id': RUN_ID, 'metric': field('Recorded series name.', enum=['episode_return']),
         'max_points': field('Return at most this many recorded points, starting from the first.', 'integer', minimum=1, maximum=3, default=3)},
        ['run_id', 'metric']),
    'count_runs': tool('count_runs', 'Count matching runs. Optional filters restrict the count; absent filters mean all values.',
                      {'project': PROJECT, 'environment': ENV, 'algorithm': ALGO}, ['project']),
    'describe_columns': tool('describe_columns', 'Return the ordered column names of a table; no table rows or statistics.',
                            {'table': field('Table identifier.', enum=['experiments'])}, ['table']),
}

BASE_TOOLS = list(TOOLS)[:6]
CONTEXT = ('Known project IDs: atlas, boreal. Known run IDs: a1, a2, a3, a4, b1. '
           'Known environments: Hopper-v4, Walker2d-v4. Algorithms: SAC, PPO. '
           'Table: experiments. These inventories are not defaults.')


def call(name, **arguments):
    return {'action': 'call', 'name': name, 'arguments': arguments}


def _case(identifier, category, request, gold, expected=None, target=None):
    target = target or gold.get('name', 'summarize_metric')
    others = [name for name in BASE_TOOLS if name != target]
    rng = random.Random(identifier)
    names = [target] + rng.sample(others, 2)
    rng.shuffle(names)
    return {'id': identifier, 'group_id': identifier, 'category': category,
            'request': request, 'context': CONTEXT,
            'tools': [copy.deepcopy(TOOLS[name]) for name in names],
            'gold': gold, 'expected_result': expected}


def rename_case(case):
    renamed = copy.deepcopy(case)
    renamed['id'] = 'renamed_' + case['id']
    renamed['category'] = 'renamed'
    renamed['bindings'] = {}
    for index, schema in enumerate(renamed['tools']):
        original_name = schema['name']
        new_name = f'operation_{index + 11}'
        params = schema['parameters']
        mapping = {key: f'input_{position + 21}' for position, key in enumerate(params['properties'])}
        params['properties'] = {mapping[key]: value for key, value in params['properties'].items()}
        params['required'] = [mapping[key] for key in params['required']]
        schema['name'] = new_name
        renamed['bindings'][new_name] = {'name': original_name, 'arguments': {new: old for old, new in mapping.items()}}
        if renamed['gold'].get('name') == original_name:
            renamed['gold']['name'] = new_name
            renamed['gold']['arguments'] = {mapping[key]: value for key, value in renamed['gold']['arguments'].items()}
    return renamed


def pilot_cases():
    common = dict(project='atlas', environment='Hopper-v4', algorithm='SAC')
    calls = [
        _case('call_mean', 'original', 'In atlas, what is the mean final return of SAC runs on Hopper-v4?', call('summarize_metric', **common, metric='final_return', stat='mean'), 200.0),
        _case('call_peak_max', 'original', 'For atlas SAC runs in Hopper-v4, take each run peak return and report the largest.', call('summarize_metric', **common, metric='peak_return', stat='max'), 330.0),
        _case('call_min', 'original', 'Give the lowest final_return across SAC runs on Hopper-v4 within atlas.', call('summarize_metric', **common, metric='final_return', stat='min'), 100.0),
        _case('call_list', 'original', 'List the identifiers of SAC runs on Hopper-v4 belonging to atlas.', call('list_runs', **common), ['a1', 'a2']),
        _case('call_rank_desc', 'original', 'In atlas on Hopper-v4, return the two run IDs with the greatest final_return, highest first.', call('rank_runs', project='atlas', environment='Hopper-v4', metric='final_return', order='desc', limit=2), ['a2', 'a1']),
        _case('call_rank_asc', 'original', 'From atlas Hopper-v4 runs, give three IDs ordered from lowest to highest final_return.', call('rank_runs', project='atlas', environment='Hopper-v4', metric='final_return', order='asc', limit=3), ['a3', 'a1', 'a2']),
        _case('call_compare', 'original', 'For atlas and Hopper-v4, compare SAC and PPO by their mean final_return.', call('compare_algorithms', project='atlas', environment='Hopper-v4', metric='final_return', algorithms=['SAC', 'PPO']), {'SAC': 200.0, 'PPO': 80.0}),
        _case('call_config', 'original', 'Show the training configuration recorded for run a2.', call('get_run_config', run_id='a2'), {'learning_rate': 0.0001, 'seed': 2}),
        _case('call_series', 'original', 'Read the first two episode_return values recorded for a1.', call('get_metric_series', run_id='a1', metric='episode_return', max_points=2), [10, 40]),
        _case('call_boreal', 'original', 'For boreal rather than atlas, average the final_return of SAC runs in Hopper-v4.', call('summarize_metric', project='boreal', environment='Hopper-v4', algorithm='SAC', metric='final_return', stat='mean'), 1000.0),
    ]
    columns = ['run_id', 'project', 'environment', 'algorithm', 'final_return', 'peak_return', 'seed']
    new = [
        _case('new_count_all', 'new_tool', 'How many runs belong to atlas, across all environments and algorithms?', call('count_runs', project='atlas'), 4),
        _case('new_count_env', 'new_tool', 'Count the atlas runs that use Hopper-v4, regardless of algorithm.', call('count_runs', project='atlas', environment='Hopper-v4'), 3),
        _case('new_count_sac', 'new_tool', 'Count only SAC Hopper-v4 runs in atlas.', call('count_runs', **common), 2),
        _case('new_count_boreal', 'new_tool', 'Give the total number of runs in boreal.', call('count_runs', project='boreal'), 1),
        _case('new_columns_1', 'new_tool', 'What are the column names of experiments, in their stored order?', call('describe_columns', table='experiments'), columns),
        _case('new_columns_2', 'new_tool', 'Inspect the experiments table schema and return its ordered field names.', call('describe_columns', table='experiments'), columns),
        _case('new_columns_3', 'new_tool', 'List columns, not rows, from the experiments table.', call('describe_columns', table='experiments'), columns),
        _case('new_columns_4', 'new_tool', 'Which named columns are available in experiments? Preserve the column order.', call('describe_columns', table='experiments'), columns),
    ]
    missing = [
        ('clarify_project', 'Average the final_return of SAC runs on Hopper-v4.', 'summarize_metric', ['project']),
        ('clarify_metric', 'In atlas, compute the mean score of SAC Hopper-v4 runs. I have not chosen between final and peak return.', 'summarize_metric', ['metric']),
        ('clarify_run', 'Show the training configuration of a run; I have not selected a run ID yet.', 'get_run_config', ['run_id']),
        ('clarify_order', 'Rank atlas Hopper-v4 runs by final_return and return two IDs; the ordering direction is unspecified.', 'rank_runs', ['order']),
        ('clarify_list_project', 'List all run identifiers; I have not selected a project.', 'list_runs', ['project']),
        ('clarify_stat', 'Aggregate final_return across atlas SAC Hopper-v4 runs. The desired statistic is unspecified.', 'summarize_metric', ['stat']),
    ]
    clarifications = [_case(identifier, 'clarify', request, {'action': 'clarify', 'missing_fields': fields}, target=target) for identifier, request, target, fields in missing]
    unsupported_requests = [
        ('unsupported_train', 'Start training a new SAC model in atlas.'),
        ('unsupported_delete', 'Delete every run in atlas permanently.'),
        ('unsupported_email', 'Email the atlas experiment results to a colleague.'),
        ('unsupported_weather', 'Fetch the current weather in Pittsburgh from the internet.'),
        ('unsupported_edit', 'Change the learning rate stored in run a1 to 0.01.'),
        ('unsupported_install', 'Install a new CUDA driver on this computer.'),
    ]
    unsupported = [_case(identifier, 'unsupported', request, {'action': 'unsupported'}, target='get_run_config') for identifier, request in unsupported_requests]
    return calls + [rename_case(case) for case in calls] + new + clarifications + unsupported


SYSTEM = ('You translate requests into one JSON action using ONLY the tools in the current user message. '
          'Read tool and parameter descriptions; names may be unfamiliar. Do not compute results yourself. '
          'If exactly one supported operation is sufficiently specified, output '
          '{"action":"call","name":"TOOL_NAME","arguments":{...}}. '
          'If a supported operation lacks required user information, output '
          '{"action":"clarify","missing_fields":["PARAMETER_NAME"]}. '
          'If no offered tool can fulfill the requested operation, output {"action":"unsupported"}. '
          'Never invent missing values or choose inventory entries as defaults. Do not guess an unspecified statistic or sort direction. '
          'Optional filters may be omitted. Output exactly one JSON object, without markdown, explanations, or extra keys.')


def build_messages(case, prompt_version='v1'):
    example_schema = json.dumps([TOOLS['get_run_config']], separators=(',', ':'))
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'Tools: ' + example_schema + '\nRequest: Read the configuration of run example_run.'},
        {'role': 'assistant', 'content': '{"action":"call","name":"get_run_config","arguments":{"run_id":"example_run"}}'},
        {'role': 'user', 'content': 'Tools: ' + example_schema + '\nRequest: Show a run configuration; I have not selected which run.'},
        {'role': 'assistant', 'content': '{"action":"clarify","missing_fields":["run_id"]}'},
        {'role': 'user', 'content': case['context'] + '\nTools: ' + json.dumps(case['tools'], separators=(',', ':')) + '\nRequest: ' + case['request']},
    ]
    if prompt_version == 'v2':
        messages[0]['content'] += (
            '\nIMPORTANT: The action field is an enum: its value MUST be exactly call, clarify, or unsupported. '
            'A function name NEVER belongs in action; put it in the separate name field. '
            'Before calling, check every required parameter against what the user actually supplied. '
            'An inventory of possible projects, runs or metrics does NOT specify the user choice. '
            'Words such as unspecified or not selected mean information is missing: return clarify with those exact schema parameter names. '
            'For example, if a project is required but the user names only an environment and algorithm, '
            'return {"action":"clarify","missing_fields":["project"]}, even if atlas is listed in the inventory.')
    elif prompt_version != 'v1':
        raise ValueError('Unknown prompt version')
    return messages
