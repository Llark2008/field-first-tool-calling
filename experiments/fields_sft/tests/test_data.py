import copy
import importlib
import json
import unittest


def pair(name='lookup',group='g'):
    tool=dict(name=name,description='Read a run',parameters=dict(type='object',properties=dict(run_id=dict(type='string'),limit=dict(type='integer',default=3),flag=dict(type='boolean')),required=['run_id','limit'],additionalProperties=False))
    base=dict(group_id=group,intent_group=group,target_tool=name,tools=[tool],context='Known runs: r1,r2. Not defaults.',template_group='train:family',split='train')
    a=dict(base,id=group+'-complete',condition='complete',request='Read r1 with flag false',supplied_arguments=dict(run_id='r1',flag=False),missing_fields=[],gold=dict(action='call',name=name,arguments=dict(run_id='r1',limit=3,flag=False)))
    b=copy.deepcopy(a);b.update(id=group+'-natural',condition='natural',request='Read a run with flag false',supplied_arguments=dict(flag=False),missing_fields=['run_id'],gold=dict(action='clarify',missing_fields=['run_id']))
    return a,b


class DataTests(unittest.TestCase):
    def setUp(self):
        try:self.m=importlib.import_module('experiments.fields_sft.data')
        except ModuleNotFoundError:self.fail('Paired fields data pipeline not implemented')

    def test_null_default_optional_values_and_final_action_share_one_answer(self):
        a,b=pair();content,spans=self.m.target(b,'F')
        self.assertEqual(content,'{"tool":"lookup","fields":{"run_id":null,"limit":3,"flag":false}}\n{"action":"clarify","missing_fields":["run_id"]}')
        values={s['name']:content[s['start']:s['end']] for s in spans if s['kind'] in ['known','null']}
        self.assertEqual(values,dict(run_id='null',limit='3',flag='false'))
        self.assertEqual(json.loads(content.splitlines()[1]),b['gold'])
        changed=copy.deepcopy(b);changed['gold']={'action':'unsupported'};changed['supplied_arguments']={'secret':'oracle'};changed['missing_fields']=[]
        for arm in ['J','F']:self.assertEqual(self.m.messages(b,arm),self.m.messages(changed,arm))

    def test_pair_audit_rejects_changed_known_value_or_fake_missing_field(self):
        a,b=pair();self.m.audit_pair(a,b)
        bad=copy.deepcopy(b);bad['supplied_arguments']['flag']=True
        with self.assertRaises(ValueError):self.m.audit_pair(a,bad)
        bad=copy.deepcopy(b);bad['missing_fields']=['limit'];bad['gold']['missing_fields']=['limit']
        with self.assertRaises(ValueError):self.m.audit_pair(a,bad)
        bad=copy.deepcopy(a);bad['supplied_arguments']['flag']=0
        with self.assertRaises(ValueError):self.m.audit_case(bad)

    def test_selection_preserves_pairs_and_tool_quotas_independent_of_source_order(self):
        rows=[]
        for name in ['lookup','read']:
            for i in range(3):rows.extend(pair(name,name+str(i)))
            extra=pair(name,name+'-extra')[0];rows.append(extra)
        for i in range(2):
            row=copy.deepcopy(rows[0]);row.update(id='unsupported'+str(i),group_id='unsupported'+str(i),intent_group='unsupported'+str(i),condition='unsupported',target_tool=None,gold={'action':'unsupported'},supplied_arguments={},missing_fields=[]);rows.append(row)
        selected=self.m.select_cases(rows,pairs_per_tool=1,extras_per_tool=1,unsupported=2)
        self.assertEqual(len(selected),8)
        self.assertEqual([r['id'] for r in selected],[r['id'] for r in self.m.select_cases(list(reversed(rows)),pairs_per_tool=1,extras_per_tool=1,unsupported=2)])
        for name in ['lookup','read']:
            part=[r for r in selected if r['target_tool']==name]
            self.assertEqual(sorted(r['condition'] for r in part),['complete','complete','natural'])
            natural=next(r for r in part if r['condition']=='natural')
            self.assertEqual(sum(r['group_id']==natural['group_id'] for r in part),2)

    def test_extra_complete_can_come_from_unselected_pairs_when_tool_has_no_unpaired_rows(self):
        rows=[r for i in range(3) for r in pair('lookup',str(i))]
        result=self.m.select_cases(rows,pairs_per_tool=1,extras_per_tool=1,unsupported=0)
        self.assertEqual(len(result),3)
        natural=next(r for r in result if r['condition']=='natural')
        self.assertEqual(sum(r['group_id']==natural['group_id'] for r in result),2)

    def test_call_target_materializes_required_default_and_rejects_schema_invalid_gold(self):
        a,b=pair();a['gold']['arguments']['limit']=3
        self.m.audit_case(a)
        del a['gold']['arguments']['limit']
        with self.assertRaises(ValueError):self.m.audit_case(a)


if __name__=='__main__':unittest.main()
