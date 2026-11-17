import importlib
import unittest
from test_data import pair


class ScoringTests(unittest.TestCase):
    def setUp(self):
        try:self.m=importlib.import_module('experiments.fields_sft.scoring')
        except ModuleNotFoundError:self.fail('Strict field diagnostics not implemented')

    def test_typed_field_comparison_distinguishes_false_zero_and_survives_mixed_arrays(self):
        self.assertFalse(self.m.equal(False,0))
        self.assertTrue(self.m.equal(['PPO','SAC'],['SAC','PPO'],'algorithms'))
        self.assertFalse(self.m.equal(['PPO',{}],['PPO','SAC'],'algorithms'))

    def test_null_recall_and_false_null_use_all_required_slots(self):
        a,b=pair();d=self.m.inspect_fields(b,{'tool':'lookup','fields':{'run_id':None,'limit':3,'flag':False}})
        self.assertEqual((d['required_missing_n'],d['required_missing_correct']),(1,1))
        self.assertEqual((d['required_known_n'],d['required_known_correct']),(1,1))
        bad=self.m.inspect_fields(a,{'tool':'lookup','fields':{'run_id':None,'limit':3,'flag':False}})
        self.assertEqual(bad['false_null_known'],1)
        absent=self.m.inspect_fields(b,None)
        self.assertEqual((absent['required_missing_n'],absent['required_missing_correct']),(1,0))
        self.assertFalse(self.m.inspect_fields(a,{'tool':'lookup','fields':{'run_id':'r1','limit':3,'flag':0}})['fields_exact'])

    def test_only_first_object_duplicate_keys_or_truncation_never_pass(self):
        a,b=pair();first='{"tool":"lookup","fields":{"run_id":null,"limit":3,"flag":false}}';final='{"action":"clarify","missing_fields":["run_id"]}'
        good=self.m.score(b,first+'\n'+final,'stop','F')
        self.assertTrue(good['score']['success']);self.assertTrue(good['field_diagnostics']['fields_exact'])
        for raw,finish in [(first,'stop'),(first+'\n'+final+' {}','stop'),(first+'\n'+final,'length'),(first.replace('"limit":3','"limit":3,"limit":4')+'\n'+final,'stop')]:
            self.assertFalse(self.m.score(b,raw,finish,'F')['score']['success'])

    def test_truncated_complete_json_cannot_inflate_field_success(self):
        from experiments.fields_sft.data import target
        a,b=pair();r=self.m.score(b,target(b,'F')[0],'length','F');d=r['field_diagnostics']
        self.assertFalse(d['format_valid']);self.assertFalse(d['fields_exact'])
        self.assertEqual(d['required_missing_correct'],0);self.assertEqual(d['required_missing_n'],1)

    def test_defaulted_values_and_optional_defaults_are_self_consistent(self):
        from experiments.fields_sft.data import field_target
        a,b=pair();action=dict(action='call',name='lookup',arguments=dict(run_id='r1',flag=False))
        self.assertTrue(self.m.consistent(field_target(a),action,a))

    def test_legal_field_errors_separate_invention_omission_and_false_null(self):
        a,b=pair();d=self.m.inspect_fields(b,dict(tool='lookup',fields=dict(run_id='r2',flag=False)))
        self.assertEqual(d['required_missing_invented'],1)
        self.assertEqual(d['required_known_omitted'],1)
        self.assertEqual(d['required_missing_n'],1);self.assertEqual(d['required_known_n'],1)


if __name__=='__main__':unittest.main()
