"""Frozen expanded diagnostic suite: 24 task groups, six conditions, two orders."""
import copy
import random

from .tasks import BASE_TOOLS, CONTEXT, TOOLS, call


# Literal answers are specified independently of evaluation.execute_call.
# Each definition includes one complete request and its natural single-slot omission.
DEFINITIONS = [
    ('s01', 'summarize_metric',
     'Across SAC runs for Hopper-v4 in atlas, take the average of final_return.',
     'Across SAC runs for Hopper-v4, take the average of final_return.', 'project',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC', metric='final_return', stat='mean'), 200.0),
    ('s02', 'summarize_metric',
     'Within boreal, find the greatest peak_return among SAC runs in Hopper-v4.',
     'Find the greatest peak_return among SAC runs in Hopper-v4.', 'project',
     dict(project='boreal', environment='Hopper-v4', algorithm='SAC', metric='peak_return', stat='max'), 1100),
    ('s03', 'summarize_metric',
     'For the atlas SAC runs using Hopper-v4, calculate the mean of their peak_return values.',
     'For the atlas SAC runs, calculate the mean of their peak_return values.', 'environment',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC', metric='peak_return', stat='mean'), 235.0),
    ('s04', 'summarize_metric',
     'What is the maximum final_return of SAC runs in atlas on Walker2d-v4?',
     'What is the maximum final_return of SAC runs in atlas?', 'environment',
     dict(project='atlas', environment='Walker2d-v4', algorithm='SAC', metric='final_return', stat='max'), 600),
    ('s05', 'summarize_metric',
     'For PPO in atlas on Hopper-v4, report the minimum final_return across runs.',
     'For my algorithm in atlas on Hopper-v4, report the minimum final_return across runs.', 'algorithm',
     dict(project='atlas', environment='Hopper-v4', algorithm='PPO', metric='final_return', stat='min'), 80),
    ('s06', 'summarize_metric',
     'In atlas, reduce the peak_return values of SAC Hopper-v4 runs by taking the maximum.',
     'In atlas, reduce the return scores of SAC Hopper-v4 runs by taking the maximum.', 'metric',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC', metric='peak_return', stat='max'), 330),
    ('s07', 'summarize_metric',
     'Summarize final_return for atlas SAC runs on Hopper-v4 using the minimum.',
     'Summarize final_return for atlas SAC runs on Hopper-v4.', 'stat',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC', metric='final_return', stat='min'), 100),
    ('s08', 'summarize_metric',
     'Use the minimum to aggregate peak_return over atlas SAC runs on Hopper-v4.',
     'Aggregate peak_return over atlas SAC runs on Hopper-v4.', 'stat',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC', metric='peak_return', stat='min'), 140),
    ('r01', 'rank_runs',
     'From atlas Hopper-v4 experiments, select two run IDs ranked by final_return from largest to smallest.',
     'From Hopper-v4 experiments, select two run IDs ranked by final_return from largest to smallest.', 'project',
     dict(project='atlas', environment='Hopper-v4', metric='final_return', order='desc', limit=2), ['a2', 'a1']),
    ('r02', 'rank_runs',
     'For atlas on Hopper-v4, return two run IDs in ascending order of peak_return.',
     'For atlas, return two run IDs in ascending order of peak_return.', 'environment',
     dict(project='atlas', environment='Hopper-v4', metric='peak_return', order='asc', limit=2), ['a3', 'a1']),
    ('r03', 'rank_runs',
     'Rank atlas Hopper-v4 runs by peak_return, descending, and give three IDs.',
     'Rank atlas Hopper-v4 runs by their return score, descending, and give three IDs.', 'metric',
     dict(project='atlas', environment='Hopper-v4', metric='peak_return', order='desc', limit=3), ['a2', 'a1', 'a3']),
    ('r04', 'rank_runs',
     'Give three atlas Hopper-v4 run IDs sorted by final_return in ascending order.',
     'Give three atlas Hopper-v4 run IDs sorted by final_return.', 'order',
     dict(project='atlas', environment='Hopper-v4', metric='final_return', order='asc', limit=3), ['a3', 'a1', 'a2']),
    ('r05', 'rank_runs',
     'Rank atlas Hopper-v4 run IDs by final_return descending, limiting the result to one ID.',
     'Rank atlas Hopper-v4 run IDs by final_return descending, with a limit on the number of results.', 'limit',
     dict(project='atlas', environment='Hopper-v4', metric='final_return', order='desc', limit=1), ['a2']),
    ('r06', 'rank_runs',
     'In boreal on Hopper-v4, select one run ID ordered by peak_return ascending.',
     'On Hopper-v4, select one run ID ordered by peak_return ascending.', 'project',
     dict(project='boreal', environment='Hopper-v4', metric='peak_return', order='asc', limit=1), ['b1']),
    ('c01', 'get_run_config',
     'Retrieve the saved training settings for a1.',
     'Retrieve the saved training settings for the run I am working on.', 'run_id',
     dict(run_id='a1'), dict(learning_rate=0.0003, seed=1)),
    ('c02', 'get_run_config',
     'I want to inspect the hyperparameter configuration attached to a2.',
     'I want to inspect the hyperparameter configuration attached to my run.', 'run_id',
     dict(run_id='a2'), dict(learning_rate=0.0001, seed=2)),
    ('c03', 'get_run_config',
     'Read the configuration record associated with a3.',
     'Read the configuration record associated with that run.', 'run_id',
     dict(run_id='a3'), dict(learning_rate=0.0003, seed=1)),
    ('c04', 'get_run_config',
     'Look up which training configuration b1 used.',
     'Look up which training configuration the experiment used.', 'run_id',
     dict(run_id='b1'), dict(learning_rate=0.0003, seed=1)),
    ('l01', 'list_runs',
     'Return IDs matching SAC and Hopper-v4 within the atlas project.',
     'Return IDs matching SAC and Hopper-v4 within my project.', 'project',
     dict(project='atlas', environment='Hopper-v4', algorithm='SAC'), ['a1', 'a2']),
    ('l02', 'list_runs',
     'Show the run identifiers for boreal with algorithm SAC and environment Hopper-v4.',
     'Show the run identifiers with algorithm SAC and environment Hopper-v4.', 'project',
     dict(project='boreal', environment='Hopper-v4', algorithm='SAC'), ['b1']),
    ('l03', 'list_runs',
     'List the PPO run IDs in atlas, without restricting the environment.',
     'List the PPO run IDs in the project, without restricting the environment.', 'project',
     dict(project='atlas', algorithm='PPO'), ['a3']),
    ('l04', 'list_runs',
     'Enumerate all run identifiers in atlas across every algorithm and environment.',
     'Enumerate all run identifiers in my project across every algorithm and environment.', 'project',
     dict(project='atlas'), ['a1', 'a2', 'a3', 'a4']),
    ('t01', 'get_metric_series',
     'Fetch the first two recorded episode_return readings for run a1.',
     'Fetch the first two recorded episode_return readings for my run.', 'run_id',
     dict(run_id='a1', metric='episode_return', max_points=2), [10, 40]),
    ('t02', 'get_metric_series',
     'Load the first three episode_return observations from a2.',
     'Load the first three episode_return observations from the run.', 'run_id',
     dict(run_id='a2', metric='episode_return', max_points=3), [30, 120, 300]),
]

ALIASES = {
    'a': {
        'tools': dict(list_runs='search_trials', summarize_metric='aggregate_scores', rank_runs='order_trials',
                      compare_algorithms='contrast_learners', get_run_config='read_trial_settings', get_metric_series='fetch_score_history'),
        'parameters': dict(project='workspace', environment='task', algorithm='learner', metric='measure', stat='reduction',
                           order='direction', limit='count', algorithms='learner_ids', run_id='trial_id', max_points='points'),
    },
    'b': {
        'tools': dict(list_runs='enumerate_experiments', summarize_metric='reduce_run_values', rank_runs='select_ranked_runs',
                      compare_algorithms='compare_methods', get_run_config='lookup_hyperparameters', get_metric_series='load_trace'),
        'parameters': dict(project='dataset', environment='environment_id', algorithm='method', metric='metric_key', stat='operation',
                           order='sort_direction', limit='max_results', algorithms='methods', run_id='experiment_id', max_points='length_limit'),
    },
}

NEIGHBORS = dict(summarize_metric='compare_algorithms', rank_runs='list_runs',
                 get_run_config='get_metric_series', list_runs='rank_runs', get_metric_series='get_run_config')
FIELD_WORDS = dict(project='project', environment='environment', algorithm='algorithm', metric='choice of final versus peak return',
                   stat='aggregation statistic', order='sorting direction', limit='number of results', run_id='run identifier')
CONDITIONS = ('complete_original', 'complete_semantic_a', 'complete_semantic_b',
              'missing_natural', 'missing_explicit', 'missing_semantic_a')


def semantic_rename(case, alias_set):
    result = copy.deepcopy(case)
    aliases = ALIASES[alias_set]
    result['bindings'] = {}
    for schema in result['tools']:
        original = schema['name']
        schema['name'] = aliases['tools'][original]
        params = schema['parameters']
        key_map = {key: aliases['parameters'][key] for key in params['properties']}
        params['properties'] = {key_map[key]: value for key, value in params['properties'].items()}
        params['required'] = [key_map[key] for key in params['required']]
        result['bindings'][schema['name']] = {'name': original, 'arguments': {new: old for old, new in key_map.items()}}
    gold = result['gold']
    if gold['action'] == 'call':
        gold['name'] = aliases['tools'][gold['name']]
        gold['arguments'] = {aliases['parameters'][key]: value for key, value in gold['arguments'].items()}
    elif gold['action'] == 'clarify':
        gold['missing_fields'] = [aliases['parameters'][key] for key in gold['missing_fields']]
    return result


def expanded_cases():
    cases = []
    for group, target, complete, incomplete, missing, arguments, expected in DEFINITIONS:
        neighbor = NEIGHBORS[target]
        rng = random.Random('expanded-v1-' + group)
        third = rng.choice([name for name in BASE_TOOLS if name not in (target, neighbor)])
        candidates = [target, neighbor, third]
        rng.shuffle(candidates)
        for order in (0, 1):
            names = candidates[order:] + candidates[:order]
            base = dict(group_id=group, order_id=order, target_tool=target, missing_field=missing,
                        request=complete, context=CONTEXT, tools=[copy.deepcopy(TOOLS[name]) for name in names],
                        gold=call(target, **arguments), expected_result=copy.deepcopy(expected))
            for condition in CONDITIONS:
                case = copy.deepcopy(base)
                if condition.startswith('missing'):
                    case['request'] = incomplete
                    if condition == 'missing_explicit':
                        case['request'] += f' I have not provided the {FIELD_WORDS[missing]}.'
                    case['gold'] = {'action': 'clarify', 'missing_fields': [missing]}
                    case['expected_result'] = None
                if condition.endswith('semantic_a'):
                    case = semantic_rename(case, 'a')
                elif condition.endswith('semantic_b'):
                    case = semantic_rename(case, 'b')
                case['condition'] = condition
                case['category'] = condition
                case['id'] = f'expanded-{group}-{condition}-o{order}'
                cases.append(case)
    # Fixed interleaving limits condition/order confounding with run progression.
    random.Random(20261112).shuffle(cases)
    return cases
