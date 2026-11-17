"""Strict scoring, separated from prompt construction and tool execution."""
import json
import math
from statistics import mean

from jsonschema import Draft202012Validator


# Synthetic fixture values, independent of the hand-specified expected answers.
RUNS = [
    dict(run_id='a1', project='atlas', environment='Hopper-v4', algorithm='SAC', final_return=100, peak_return=140, seed=1),
    dict(run_id='a2', project='atlas', environment='Hopper-v4', algorithm='SAC', final_return=300, peak_return=330, seed=2),
    dict(run_id='a3', project='atlas', environment='Hopper-v4', algorithm='PPO', final_return=80, peak_return=120, seed=1),
    dict(run_id='a4', project='atlas', environment='Walker2d-v4', algorithm='SAC', final_return=600, peak_return=650, seed=1),
    dict(run_id='b1', project='boreal', environment='Hopper-v4', algorithm='SAC', final_return=1000, peak_return=1100, seed=1),
]
CONFIGS = {'a1': {'learning_rate': 0.0003, 'seed': 1}, 'a2': {'learning_rate': 0.0001, 'seed': 2},
           'a3': {'learning_rate': 0.0003, 'seed': 1}, 'a4': {'learning_rate': 0.0003, 'seed': 1},
           'b1': {'learning_rate': 0.0003, 'seed': 1}}
SERIES = {'a1': [10, 40, 100], 'a2': [30, 120, 300], 'a3': [0, 20, 80], 'a4': [20, 240, 600], 'b1': [100, 300, 1000]}


def _rows(arguments):
    return [row for row in RUNS if all(row[key] == arguments[key] for key in ('project', 'environment', 'algorithm') if key in arguments)]


def execute_call(case, action):
    """Only reads tool bindings and actual fixture data; never reads gold."""
    if 'fixture' in case:
        from .fixtures import execute_fixture
        return execute_fixture(case['fixture'], action, case.get('bindings'))
    name, args = action['name'], dict(action['arguments'])
    binding = case.get('bindings', {}).get(name)
    if binding:
        name = binding['name']
        args = {binding['arguments'][key]: value for key, value in args.items()}
    rows = _rows(args)
    if name == 'list_runs':
        return sorted(row['run_id'] for row in rows)
    if name == 'summarize_metric':
        return {'mean': mean, 'min': min, 'max': max}[args['stat']]([row[args['metric']] for row in rows])
    if name == 'rank_runs':
        rows = sorted(rows, key=lambda row: row[args['metric']], reverse=args['order'] == 'desc')
        return [row['run_id'] for row in rows[:int(args['limit'])]]
    if name == 'compare_algorithms':
        return {algorithm: mean(row[args['metric']] for row in rows if row['algorithm'] == algorithm) for algorithm in args['algorithms']}
    if name == 'get_run_config':
        return dict(CONFIGS[args['run_id']])
    if name == 'get_metric_series':
        return SERIES[args['run_id']][:int(args.get('max_points', 3))]
    if name == 'count_runs':
        return len(rows)
    if name == 'describe_columns':
        return ['run_id', 'project', 'environment', 'algorithm', 'final_return', 'peak_return', 'seed']
    raise ValueError('Unknown tool')


def _canonical_arguments(schema, args):
    result = dict(args)
    for key, field in schema['parameters']['properties'].items():
        if key not in result and 'default' in field:
            result[key] = field['default']
        if field.get('uniqueItems') and isinstance(result.get(key), list):
            result[key] = sorted(result[key])
    return result


def _equivalent(actual, expected):
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equivalent(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_equivalent(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def score_response(case, text):
    result = dict(parsed=False, schema_valid=False, decision_correct=False,
                  tool_correct=False, arguments_correct=False, execution_correct=False,
                  success=False, error=None)
    try:
        action = json.loads(text, parse_constant=_reject_constant)
        if not isinstance(action, dict):
            raise ValueError('Expected a JSON object')
    except (ValueError, TypeError) as error:
        result['error'] = 'parse: ' + str(error)
        return result
    result['parsed'] = True
    result['action'] = action
    gold = case['gold']
    kind = action.get('action')
    result['decision_correct'] = kind == gold['action']
    if kind == 'unsupported':
        result['schema_valid'] = set(action) == {'action'}
        result['success'] = result['schema_valid'] and result['decision_correct']
        return result
    if kind == 'clarify':
        fields = action.get('missing_fields')
        result['schema_valid'] = (set(action) == {'action', 'missing_fields'} and isinstance(fields, list)
                                  and len(fields) > 0 and all(isinstance(value, str) for value in fields)
                                  and len(fields) == len(set(fields)))
        result['arguments_correct'] = result['schema_valid'] and set(fields) == set(gold.get('missing_fields', []))
        result['success'] = result['decision_correct'] and result['arguments_correct']
        return result
    if kind != 'call' or set(action) != {'action', 'name', 'arguments'}:
        result['error'] = 'schema: invalid action shape'
        return result
    schema = next((tool for tool in case['tools'] if tool['name'] == action.get('name')), None)
    if schema is None:
        result['error'] = 'schema: unavailable tool'
        return result
    errors = list(Draft202012Validator(schema['parameters']).iter_errors(action['arguments']))
    if errors:
        result['error'] = 'schema: ' + errors[0].message
        return result
    result['schema_valid'] = True
    result['tool_correct'] = result['decision_correct'] and action['name'] == gold.get('name')
    result['arguments_correct'] = result['tool_correct'] and _canonical_arguments(schema, action['arguments']) == _canonical_arguments(schema, gold['arguments'])
    try:
        result['execution_result'] = execute_call(case, action)
        result['execution_correct'] = gold['action'] == 'call' and _equivalent(result['execution_result'], case['expected_result'])
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        result['error'] = 'execution: ' + str(error)
    result['success'] = result['arguments_correct'] and result['execution_correct']
    return result
