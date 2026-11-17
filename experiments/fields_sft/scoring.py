"""Strict output scoring and type-safe field diagnostics with all-slot denominators."""
import json
from experiments.fields_first_prompt.core import parse,field_target
from toolcall_lab.generation_scoring import score_generation
from toolcall_lab.controls import source_guard
from toolcall_lab.evaluate_formal import failure_type

def normalized(value,key=None):
    if isinstance(value,dict):return ('object',tuple(sorted((k,normalized(v,k)) for k,v in value.items())))
    if isinstance(value,list):
        items=[normalized(v) for v in value]
        return ('array',tuple(sorted(items,key=repr) if key=='algorithms' else items))
    if isinstance(value,bool):return ('boolean',value)
    if isinstance(value,(int,float)):return ('number',value)
    return (type(value).__name__,value)

def equal(a,b,key=None):return normalized(a,key)==normalized(b,key)

def inspect_fields(case,first):
    target=field_target(case);schema=next((t['parameters'] for t in case['tools'] if t['name']==target['tool']),None)
    fields=first['fields'] if first else {};right_tool=bool(first is not None and first['tool']==target['tool'])
    required=schema['required'] if schema else []
    missing=[k for k in required if target['fields'][k] is None];known=[k for k in required if k not in missing]
    return dict(format_valid=first is not None,tool_correct=right_tool,fields_exact=right_tool and equal(fields,target['fields']),
        missing_set_correct=right_tool and sorted(k for k,v in fields.items() if v is None)==sorted(case['missing_fields']),
        required_missing_n=len(missing),required_missing_correct=sum(right_tool and k in fields and fields[k] is None for k in missing),
        required_known_n=len(known),required_known_correct=sum(right_tool and k in fields and equal(fields[k],target['fields'][k],k) for k in known),
        false_null_known=sum(right_tool and k in fields and fields[k] is None for k in known),
        required_missing_invented=sum(right_tool and k in fields and fields[k] is not None for k in missing),
        required_missing_omitted=sum(right_tool and k not in fields for k in missing),
        required_missing_wrong_tool=len(missing) if not right_tool else 0,
        required_known_wrong_value=sum(right_tool and k in fields and fields[k] is not None and not equal(fields[k],target['fields'][k],k) for k in known),
        required_known_omitted=sum(right_tool and k not in fields for k in known),
        required_known_wrong_tool=len(known) if not right_tool else 0,
        required_slots_covered=sum(right_tool and k in fields for k in required),predicted=first,expected=target)

def consistent(first,action,case):
    if first is None:return False
    tool,fields=first['tool'],first['fields'];kind=action.get('action')
    if tool is None:return not fields and kind=='unsupported'
    schema=next((t['parameters'] for t in case['tools'] if t['name']==tool),None)
    if not schema or not set(schema['required'])<=set(fields) or not set(fields)<=set(schema['properties']):return False
    missing=sorted(k for k,v in fields.items() if v is None)
    if not set(missing)<=set(schema['required']):return False
    if missing:return kind=='clarify' and sorted(action.get('missing_fields',[]))==missing
    args=action.get('arguments',{})
    def defaults(values):
        return {**{k:p['default'] for k,p in schema['properties'].items() if 'default' in p},**values}
    return kind=='call' and action.get('name')==tool and isinstance(args,dict) and equal(defaults(args),defaults(fields))

def score(case,raw,finish_reason,arm):
    if arm not in ['J','F']:raise ValueError('Unknown arm')
    first=None;error=None
    try:
        first,action=parse(raw,{'J':'I','F':'V'}[arm]);result=score_generation(case,json.dumps(action),finish_reason)
    except (ValueError,TypeError) as exc:
        first=None;error=str(exc);result=score_generation(case,'INVALID_FORMAT',finish_reason)
    valid=result['schema_valid'] and not result['truncated']
    guard=source_guard(case['request'],case['tools'],result['action']) if valid and result['action'].get('action')=='call' else dict(blocked=False,ungrounded_fields=[])
    diagnostics=None
    if arm=='F':
        diagnostics=inspect_fields(case,first if not result['truncated'] else None)
        diagnostics['eligible_output']=valid and first is not None
        diagnostics.update(self_consistent=valid and consistent(first,result.get('action',{}),case),pipeline_success=result['success'] and diagnostics['fields_exact'])
    row=dict(score=result,guard=guard,gold=case['gold'],format_error=error,field_diagnostics=diagnostics)
    row['failure_type']=failure_type(row)
    return row
