"""Non-training controls. The source guard has no access to answer labels."""
import re

from .tasks import build_messages


def build_control_messages(case, variant):
    if variant in ('v1', 'v2'):
        return build_messages(case, variant)
    if variant not in ('no_examples', 'checklist'):
        raise ValueError('Unknown prompt control: ' + variant)
    old = build_messages(case, 'v2')
    system = old[0]['content'].split('For example, if a project is required')[0].rstrip()
    if variant == 'checklist':
        system += (
            '\nDecision procedure: First identify the supported operation. Then inspect its required fields. '
            'For each field, check whether the request or explicitly established user context determines its value. '
            'A plausible value is not evidence. If any required field without a documented default is undetermined, '
            'return clarify and list all such fields using the current schema names. '
            'If all necessary information is available, call the tool. Optional filters need not be supplied. '
            'Infer unambiguous meanings from the request, but never infer a user choice from the available inventory. '
            'Do this check internally; output only the final JSON action.')
    return [{'role': 'system', 'content': system}, old[-1]]


ENTITY_DESCRIPTIONS = {'Project identifier; never assume a project.', 'Environment identifier.',
                       'Algorithm.', 'Existing run identifier.'}


def source_guard(request, schemas, action):
    """Lexical baseline: only explicit entity values, no numeric/semantic inference.

    A block is a system abstention, never re-labelled as a model clarification.
    Callers apply this only to valid, non-truncated calls. This intentionally
    does not resolve negation, aliases, anaphora, or conversational context.
    """
    fields = []
    if action.get('action') != 'call':
        return dict(blocked=False, ungrounded_fields=fields)
    schema = next((s for s in schemas if s['name'] == action.get('name')), None)
    if schema is None or not isinstance(action.get('arguments'), dict):
        return dict(blocked=False, ungrounded_fields=fields)
    for key, value in action['arguments'].items():
        prop = schema['parameters']['properties'].get(key, {})
        if prop.get('type') == 'array':
            desc, values = prop.get('items', {}).get('description'), value
        else:
            desc, values = prop.get('description'), [value]
        if desc not in ENTITY_DESCRIPTIONS:
            continue
        if any(not isinstance(v, str) or not re.search(r'(?<![\w-])' + re.escape(v) + r'(?![\w-])',
                                                      request, re.IGNORECASE) for v in values):
            fields.append(key)
    return dict(blocked=bool(fields), ungrounded_fields=sorted(fields))


def guard_counts(rows, case_map):
    result = dict(n=len(rows), raw_success=0, correct_clarifications=0, raw_premature_calls=0,
                  valid_calls=0, valid_premature_calls=0, blocked_calls=0, blocked_correct_calls=0,
                  allowed_premature_calls=0, system_success=0)
    for row in rows:
        score = row['score']
        action = score.get('action', {})
        is_call = action.get('action') == 'call'
        premature = is_call and row['gold']['action'] == 'clarify'
        valid = is_call and score['schema_valid'] and not score.get('truncated', False)
        case = case_map[row['case_id']]
        blocked = valid and source_guard(case['request'], case['tools'], action)['blocked']
        result['raw_success'] += bool(score['success'])
        result['correct_clarifications'] += action.get('action') == 'clarify' and bool(score['success'])
        result['raw_premature_calls'] += premature
        result['valid_calls'] += valid
        result['valid_premature_calls'] += valid and premature
        result['blocked_calls'] += blocked
        result['blocked_correct_calls'] += blocked and bool(score['success'])
        result['allowed_premature_calls'] += valid and premature and not blocked
        result['system_success'] += bool(score['success']) and not blocked
    return result
