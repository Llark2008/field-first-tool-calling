import copy
import importlib
import unittest


class FieldsTests(unittest.TestCase):
    def setUp(self):
        try:self.c=importlib.import_module('experiments.fields_first_prompt.core')
        except ModuleNotFoundError:self.fail('Fields-first spike not implemented')
        self.case=dict(request='Read the curve for r123',context='Known runs: r123,r456. Not defaults.',
            tools=[dict(name='read',description='Read series',parameters=dict(type='object',properties=dict(run_id=dict(type='string'),metric=dict(type='string'),max_points=dict(type='integer',default=3)),required=['run_id','metric'],additionalProperties=False))],
            target_tool='read',gold=dict(action='clarify',missing_fields=['metric']),missing_fields=['metric'],supplied_arguments=dict(run_id='r123'))

    def test_inference_messages_never_read_private_labels(self):
        changed=copy.deepcopy(self.case); changed['gold']={'action':'unsupported'};changed['missing_fields']=[];changed['supplied_arguments']={'metric':'secret-label'};changed['target_tool']='secret-tool'
        for arm in ['I','V']:self.assertEqual(self.c.messages(self.case,arm),self.c.messages(changed,arm))

    def test_field_targets_follow_required_and_documented_defaults(self):
        self.assertEqual(self.c.field_target(self.case),dict(tool='read',fields=dict(run_id='r123',metric=None)))
        self.case['tools'][0]['parameters']['required'].append('max_points')
        self.assertEqual(self.c.field_target(self.case)['fields']['max_points'],3)
        self.case['missing_fields']=[]
        with self.assertRaises(ValueError):self.c.audit_case(self.case)

    def test_parser_requires_exact_order_count_and_no_duplicate_keys(self):
        raw='{"tool":"read","fields":{"run_id":"r123","metric":null}}\n{"action":"clarify","missing_fields":["metric"]}'
        first,action=self.c.parse(raw,'V')
        self.assertIsNone(first['fields']['metric']);self.assertEqual(action['action'],'clarify')
        for bad in [raw+' {}','```json\n'+raw+'\n```',raw.replace('\n',''),raw.replace('"tool":"read"','"tool":"read","tool":"other"'),'{"action":"call"}\n'+raw]:
            with self.assertRaises(ValueError):self.c.parse(bad,'V')
        with self.assertRaises(ValueError):self.c.parse(raw,'I')

    def test_invented_field_is_not_self_consistent(self):
        raw='{"tool":"read","fields":{"run_id":"r123","metric":null,"invented":null}}\n{"action":"clarify","missing_fields":["metric","invented"]}'
        self.assertFalse(self.c.score(self.case,raw,'stop','V')['field_diagnostics']['self_consistent'])


if __name__=='__main__':unittest.main()
