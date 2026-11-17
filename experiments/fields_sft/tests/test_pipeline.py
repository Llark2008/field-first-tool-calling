import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

class PipelineTests(unittest.TestCase):
    def test_recipe_identity_changes_with_arm_data_or_sequence_cap(self):
        try:m=importlib.import_module('experiments.fields_sft.train')
        except ModuleNotFoundError:self.fail('Training identity not implemented')
        a=m.recipe('J','fingerprint','data',1601)
        self.assertEqual(a['checkpoints'],[200,400,600]);self.assertEqual(a['normalization'],'sample')
        for key,value in [('arm','F'),('sft_sha256','other'),('max_seq_length',1633)]:
            b=dict(a,**{key:value});self.assertNotEqual(m.identity(a),m.identity(b))

    def test_partial_or_resource_failure_cannot_get_joint_verdict(self):
        try:m=importlib.import_module('experiments.fields_sft.report')
        except ModuleNotFoundError:self.fail('Report completion gate not implemented')
        metric={k:dict(count=v,n=n) for k,v,n in [('C',56,60),('Q',20,48),('F',3,60),('boundary',27,30),('U',11,12)]}
        grid={f'{a}-{s}':copy.deepcopy(metric) for a in ['J','F'] for s in [0,200,400,600]};grid['J-600']['Q']['count']=16
        self.assertTrue(m.verdict(grid,{'J','F'},True)['joint_pass'])
        for g,replays,clean in [(dict(grid,**{'F-600':None}),{'J','F'},True),(grid,{'J'},True),(grid,{'J','F'},False)]:
            with self.assertRaises(ValueError):m.verdict(g,replays,clean)

    def test_raw_outputs_are_bound_to_case_labels_and_fresh_rescoring(self):
        try:m=importlib.import_module('experiments.fields_sft.evaluate')
        except ModuleNotFoundError:self.fail('Output validation not implemented')
        from test_data import pair
        from experiments.fields_sft.scoring import score
        a,b=pair();raw='{"action":"clarify","missing_fields":["run_id"]}'
        row=dict(case_id=b['id'],condition=b['condition'],template_group=b['template_group'],target_tool=b['target_tool'],missing_fields=b['missing_fields'],output=raw,finish_reason='stop',**score(b,raw,'stop','J'))
        m.validate_rows([b],[row],'J')
        bad=copy.deepcopy(row);bad['score']['success']=False
        with self.assertRaises(ValueError):m.validate_rows([b],[bad],'J')
        bad=copy.deepcopy(row);bad['missing_fields']=['limit']
        with self.assertRaises(ValueError):m.validate_rows([b],[bad],'J')

    def test_real_tokenizer_supervises_both_objects_then_eos_and_null_spans(self):
        from mlx_lm.utils import load_tokenizer
        from experiments.fields_sft.data import encoded_target
        from test_data import pair
        root=Path.home()/'.local/share/local-tool-calling/runs/fields-first-prompt-20261115'
        snapshot=Path(json.loads((root/'frozen.json').read_text())['snapshot'])
        tok=load_tokenizer(snapshot);a,b=pair()
        row,tokens,offset,groups=encoded_target(b,'F',tok)
        self.assertTrue(groups['known']);self.assertTrue(groups['null']);self.assertTrue(groups['action'])
        self.assertTrue(all(offset<=p<len(tokens) for v in groups.values() for p in v))
        self.assertLess(max(groups['null']),min(groups['action']))
        self.assertTrue(set(tokens[max(groups['action'])+1:])&set(tok.eos_token_ids))

if __name__=='__main__':unittest.main()
