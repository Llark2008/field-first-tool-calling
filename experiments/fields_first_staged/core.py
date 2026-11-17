"""Two model calls; stage two receives raw model output, never oracle fields."""
from toolcall_lab.controls import build_control_messages
from experiments.fields_first_prompt.core import COMMON,INTERNAL

FIRST=('This is the FIELD INSPECTION turn only. Output exactly one compact JSON object with keys tool and fields. '
       'tool is the current supported tool name, or null if unsupported. fields maps every required parameter to '
       'its determined JSON value or null if missing, and also includes optional parameters explicitly supplied. '
       'For an unsupported request use {"tool":null,"fields":{}}. Do not output the final action on this turn. '
       'No markdown, explanations, or extra objects.')
SECOND=('Now use your field inspection above to produce the final action for the original request. '
        'If the selected supported tool has any null required field without a default, clarify with all such field names; '
        'otherwise call it with the determined arguments. If unsupported, return unsupported. '
        'Output exactly one action JSON object only; do not output another field object.')


def messages(case,stage,raw_fields=None):
    if stage not in [1,2] or (stage==2 and not isinstance(raw_fields,str)):raise ValueError('Invalid stage or missing raw field output')
    base=build_control_messages(case,'no_examples')
    system=base[0]['content'].replace('Output exactly one JSON object, without markdown, explanations, or extra keys.','')
    base[0]['content']=system+COMMON+(FIRST if stage==1 else INTERNAL)
    if stage==2:base += [dict(role='assistant',content=raw_fields),dict(role='user',content=SECOND)]
    return base
