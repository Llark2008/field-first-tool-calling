"""Schema-only conversion. No converter argument can carry private case labels."""
import copy
import json
from jsonschema import Draft202012Validator
from experiments.fields_first_prompt.core import unique_pairs,field_target
from experiments.fields_sft.data import compact
from experiments.fields_sft.scoring import equal

def strict_decoder():
    return json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def field_shape(first):
    if not isinstance(first,dict) or set(first)!={'tool','fields'} or not isinstance(first['fields'],dict) or not (first['tool'] is None or isinstance(first['tool'],str)):
        raise ValueError('Invalid field object shape')

def convert(tools,first):
    field_shape(first)
    tool,fields=first['tool'],copy.deepcopy(first['fields'])
    if tool is None:
        if fields:raise ValueError('Unsupported object has fields')
        return {'action':'unsupported'}
    matches=[t for t in tools if t['name']==tool]
    if len(matches)!=1:raise ValueError('Unknown or ambiguous tool')
    schema=matches[0]['parameters'];props=schema['properties'];required=schema.get('required',[])
    if not set(required)<=set(fields):raise ValueError('Required field omitted from inspection')
    if not set(fields)<=set(props):raise ValueError('Unknown field')
    for k,v in list(fields.items()):
        if v is None:
            if k not in required:raise ValueError('Optional null outside field protocol')
            if 'default' in props[k]:fields[k]=copy.deepcopy(props[k]['default'])
            else:continue
        if not Draft202012Validator(props[k]).is_valid(fields[k]):raise ValueError('Invalid value: '+k)
    missing=sorted(k for k in required if fields[k] is None)
    if missing:return {'action':'clarify','missing_fields':missing}
    if not Draft202012Validator(schema).is_valid(fields):raise ValueError('Arguments violate schema')
    return {'action':'call','name':tool,'arguments':fields}

def first_object(raw,finish_reason):
    if finish_reason=='length':raise ValueError('Truncated parent output')
    first,end=strict_decoder().raw_decode(raw.lstrip());field_shape(first)
    # Intentionally ignore the final action, including a malformed final object.
    return first

def parse_action(raw):
    value=strict_decoder().decode(raw.strip())
    if not isinstance(value,dict):raise ValueError('Action must be an object')
    return value

MAPPING_SYSTEM=('Convert the supplied field inspection into exactly one final action JSON object. '
    'The supplied fields are authoritative. Do not select another tool or infer additional values. '
    'If tool is null, output {"action":"unsupported"}. '
    'Otherwise inspect the selected tool schema. For every required field with value null and no documented default, '
    'include its name in {"action":"clarify","missing_fields":["field"]}. Include ALL such required fields. '
    'Do not ask for an optional field, or for a field whose value is already known. '
    'If there are no missing required fields, output {"action":"call","name":"selected tool","arguments":{...}} '
    'copying the determined values, retaining explicitly supplied optional values, and using only documented defaults. '
    'Output only the action JSON. No field inspection object, markdown or explanations.')

def mapping_messages(tools,fields):
    field_shape(fields)
    return [dict(role='system',content=MAPPING_SYSTEM),dict(role='user',content=compact(dict(tools=tools,inspection=fields)))]

def action_matches(tools,expected,actual):
    if not isinstance(actual,dict) or actual.get('action')!=expected['action']:return False
    kind=expected['action']
    if kind=='unsupported':return actual==expected
    if kind=='clarify':
        xs=actual.get('missing_fields')
        return set(actual)=={'action','missing_fields'} and isinstance(xs,list) and all(isinstance(x,str) for x in xs) and len(xs)==len(set(xs)) and set(xs)==set(expected['missing_fields'])
    if set(actual)!={'action','name','arguments'} or actual['name']!=expected['name']:return False
    schema=next(t['parameters'] for t in tools if t['name']==expected['name'])
    if not Draft202012Validator(schema).is_valid(actual['arguments']):return False
    def defaults(v):return {**{k:p['default'] for k,p in schema['properties'].items() if 'default'in p},**v}
    return equal(defaults(actual['arguments']),defaults(expected['arguments']))

def probes(cases):
    result=[]
    for tool in sorted({c['target_tool'] for c in cases if c['condition']=='complete'}):
        c=min((c for c in cases if c['condition']=='complete' and c['target_tool']==tool),key=lambda c:c['id'])
        first=field_target(c);schema=next(t['parameters'] for t in c['tools'] if t['name']==tool)
        def add(kind,obj,gold):
            result.append(dict(id=tool+':'+kind,source_id=c['id'],kind=kind,tools=c['tools'],first=obj,gold=gold))
        add('original',copy.deepcopy(first),copy.deepcopy(c['gold']))
        reversed_first=copy.deepcopy(first);reversed_first['fields']=dict(reversed(list(first['fields'].items())))
        add('reverse_order',reversed_first,copy.deepcopy(c['gold']))
        keys=[k for k in schema['required'] if 'default' not in schema['properties'][k]]
        for n in [1,2]:
            if len(keys)<n:continue
            obj=copy.deepcopy(first)
            for k in keys[:n]:obj['fields'][k]=None
            add('null_'+str(n),obj,{'action':'clarify','missing_fields':sorted(keys[:n])})
        optional=[k for k in first['fields'] if k not in schema['required']]
        if optional:
            obj=copy.deepcopy(first);gold=copy.deepcopy(c['gold'])
            for k in optional:del obj['fields'][k];del gold['arguments'][k]
            add('omit_optional',obj,gold)
    return result
