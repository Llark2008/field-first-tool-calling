"""Matched internal/visible field inspection; labels are used only for scoring."""
import json
from toolcall_lab.controls import build_control_messages,source_guard
from toolcall_lab.generation_scoring import score_generation
from toolcall_lab.evaluate_formal import failure_type

COMMON=('\nField inspection procedure: Identify the one supported tool by its documented operation. '
    'For every required parameter, determine its value from the user request or explicitly established context. '
    'A documented default may be used; an inventory entry is not a user choice. '
    'Represent every undetermined required value as null. Also retain optional parameters explicitly supplied by the user. '
    'If no tool supports the operation, select no tool. '
    'Then form the final action: unsupported if no supported tool; clarify listing all null required fields if any; '
    'otherwise call with the determined arguments. Do not turn null into a guessed value. ')
VISIBLE=('Output exactly two compact JSON objects separated by a newline, with no markdown or prose. '
    'FIRST output {"tool":"CURRENT_TOOL_NAME","fields":{"PARAMETER_NAME":"determined value or null"}}. '
    'The fields object must contain every required parameter (null when undetermined), plus explicitly supplied optional parameters. '
    'Use correctly typed JSON values, not the literal string "null". If unsupported, FIRST output {"tool":null,"fields":{}}. '
    'SECOND output the final action JSON in the call/clarify/unsupported format defined above, using the field inspection.')
INTERNAL=('Do the field inspection internally. Output exactly one final action JSON object in the call/clarify/unsupported format '
    'defined above, without markdown, explanations or intermediate fields.')


def messages(case,arm):
    if arm not in ['I','V']: raise ValueError('Unknown arm')
    result=build_control_messages(case,'no_examples')
    system=result[0]['content'].replace('Output exactly one JSON object, without markdown, explanations, or extra keys.','')
    result[0]['content']=system+COMMON+(INTERNAL if arm=='I' else VISIBLE)
    return result


def field_target(case):
    if case['gold']['action']=='unsupported':return dict(tool=None,fields={})
    tool=next(t for t in case['tools'] if t['name']==case['target_tool'])
    props=tool['parameters']['properties']; supplied=case['supplied_arguments']
    fields={k:supplied.get(k,props[k].get('default')) for k in tool['parameters']['required']}
    fields.update(supplied)
    return dict(tool=tool['name'],fields=fields)


def audit_case(case):
    target=field_target(case)
    if target['tool'] is None:return target
    tool=next(t for t in case['tools'] if t['name']==target['tool'])
    if not set(target['fields'])<=set(tool['parameters']['properties']): raise ValueError('Unknown labeled field')
    missing=sorted(k for k in tool['parameters']['required'] if target['fields'][k] is None)
    if missing!=sorted(case['missing_fields']):raise ValueError('Fields and missing-field label disagree')
    if case['gold']['action']=='call' and case['gold']['arguments']!=case['supplied_arguments']:raise ValueError('Call argument labels disagree')
    if (case['gold']['action']=='clarify')!=bool(missing):raise ValueError('Action and fields disagree')
    return target


def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('Duplicate JSON key')
        result[k]=v
    return result


def parse(raw,arm):
    if arm not in ['I','V']:raise ValueError('Unknown arm')
    decoder=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    remaining=raw.strip(); objects=[]
    for i in range(1 if arm=='I' else 2):
        obj,end=decoder.raw_decode(remaining); objects.append(obj);tail=remaining[end:]
        if arm=='V' and i==0 and '\n' not in tail[:len(tail)-len(tail.lstrip())]:raise ValueError('Field/action objects require newline separator')
        remaining=tail.strip()
    if remaining or not all(isinstance(o,dict) for o in objects):raise ValueError('Wrong object count/type or trailing text')
    if arm=='I':return None,objects[0]
    first,action=objects
    if set(first)!={'tool','fields'} or not (first['tool'] is None or isinstance(first['tool'],str)) or not isinstance(first['fields'],dict):
        raise ValueError('Invalid field-inspection format')
    return first,action


def canonical(fields):
    return {k:sorted(v) if k=='algorithms' and isinstance(v,list) else v for k,v in fields.items()}


def score(case,raw,finish_reason,arm):
    first=None; error=None
    try:
        first,action=parse(raw,arm)
        score=score_generation(case,json.dumps(action),finish_reason)
    except (ValueError,TypeError) as exc:
        error=str(exc);score=score_generation(case,'INVALID_FORMAT',finish_reason)
    valid_call=score['schema_valid'] and not score['truncated'] and score.get('action',{}).get('action')=='call'
    guard=source_guard(case['request'],case['tools'],score['action']) if valid_call else dict(blocked=False,ungrounded_fields=[])
    diagnostics=None
    if arm=='V':
        target=field_target(case); fields=first['fields'] if first else {}; predicted_tool=first['tool'] if first else 'INVALID'
        schema=next((t for t in case['tools'] if t['name']==predicted_tool),None)
        missing=sorted(k for k,v in fields.items() if v is None)
        action=score.get('action',{}); kind=action.get('action')
        full_required=bool(schema and set(schema['parameters']['required'])<=set(fields) and set(fields)<=set(schema['parameters']['properties'])
                           and set(missing)<=set(schema['parameters']['required']))
        consistent=(predicted_tool is None and fields=={} and kind=='unsupported') or (
            full_required and ((missing and kind=='clarify' and sorted(action.get('missing_fields',[]))==missing) or
                              (not missing and kind=='call' and action.get('name')==predicted_tool and
                               all(canonical(action.get('arguments',{})).get(k)==v for k,v in canonical(fields).items()))))
        diagnostics=dict(format_valid=error is None,tool_correct=predicted_tool==target['tool'],
            missing_set_correct=error is None and predicted_tool==target['tool'] and missing==sorted(case['missing_fields']),
            fields_exact=error is None and predicted_tool==target['tool'] and canonical(fields)==canonical(target['fields']),
            self_consistent=bool(error is None and consistent),predicted=first,expected=target)
    row=dict(score=score,guard=guard,gold=case['gold'],format_error=error,field_diagnostics=diagnostics)
    row['failure_type']=failure_type(row)
    return row
