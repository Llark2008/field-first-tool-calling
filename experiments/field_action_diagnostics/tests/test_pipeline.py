import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from experiments.field_action_diagnostics import run

class PipelineTests(unittest.TestCase):
    def test_completed_report_is_not_overwritten_when_rescore_differs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json';p.write_text('{"success":86}')
            before=p.read_bytes()
            with self.assertRaises(ValueError):run.publish_report(p,{'success':98},expected={'success':86})
            self.assertEqual(p.read_bytes(),before)

    def test_freeze_covers_imported_scoring_and_resource_code(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ['experiments/field_action_diagnostics/core.py','toolcall_lab/evaluation.py','toolcall_lab/resource_monitor.py','scripts/run_train_replay.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
            with patch.object(run,'SOURCE',root),patch.object(run,'DOCS',[]):
                before=run.sources()
                self.assertIn('toolcall_lab/evaluation.py',before)
                self.assertIn('toolcall_lab/resource_monitor.py',before)
                (root/'toolcall_lab/evaluation.py').write_text('modified')
                f={'sources':before,'inputs':{},'local_files':{}};f['fingerprint']=run.digest(f)
                with self.assertRaisesRegex(ValueError,'Frozen source changed'):run.verify(f)

    def test_scoring_real_case_preserves_gold_without_duplicate_metadata(self):
        from pathlib import Path
        import json
        p=Path.home()/'.local/share/local-tool-calling/runs/fields-sft-20261115/dev-cases.jsonl'
        c=json.loads(p.read_text().splitlines()[0])
        row=run.score_action(c,json.dumps(c['gold']))
        self.assertTrue(row['score']['success'])
        self.assertEqual(row['gold'],c['gold'])

    def test_resume_rejects_changed_prompt_order_and_tampered_score(self):
        jobs=[{'id':'a','arm':'G','prompt_sha256':'one'}]
        row={'id':'a','arm':'G','prompt_sha256':'one','output':'{}','finish_reason':'stop'}
        self.assertEqual(run.validate_raw(jobs,[row],partial=False),[row])
        for key,value in [('id','b'),('prompt_sha256','changed'),('arm','S')]:
            with self.assertRaises(ValueError):run.validate_raw(jobs,[dict(row,**{key:value})])
        with self.assertRaises(ValueError):run.validate_raw(jobs,[])

    def test_oracle_prefix_has_no_action_tokens_with_real_tokenizer(self):
        # Real frozen tokenizer and target, not a fake tokenizer that mirrors the implementation.
        from pathlib import Path
        import json
        from mlx_lm.utils import load_tokenizer
        from experiments.fields_sft.data import encoded_target
        p=Path.home()/'.local/share/local-tool-calling/runs/fields-sft-20261115'
        c=json.loads((p/'dev-cases.jsonl').read_text().splitlines()[0])
        f=json.loads((p/'frozen.json').read_text());tok=load_tokenizer(Path(f['snapshot']))
        job=run.make_job(c,'G',tok)
        _,ids,offset,groups=encoded_target(c,'F',tok)
        self.assertEqual(job['prompt_ids'],ids[:len(job['prompt_ids'])])
        self.assertEqual(len(job['prompt_ids']),min(groups['action']))
        altered=copy.deepcopy(c);altered['expected_result']='PRIVATE';altered['gold']['arguments']['run_id']='SECRET_ACTION'
        self.assertEqual(run.make_job(c,'G',tok),run.make_job(altered,'G',tok))

    def test_summary_counts_rejections_in_denominator(self):
        cases=[{'id':'c','condition':'complete','intent_group':'p'}, {'id':'n','condition':'natural','intent_group':'p'}, {'id':'u','condition':'unsupported','intent_group':'u'}]
        rows=[{'case_id':'c','score':{'success':False,'schema_valid':False},'rejection':'bad'},
              {'case_id':'n','score':{'success':True,'schema_valid':True,'action':{'action':'clarify'}}},
              {'case_id':'u','score':{'success':True,'schema_valid':True,'action':{'action':'unsupported'}}}]
        got=run.summarize(cases,rows)
        self.assertEqual((got['n'],got['success'],got['rejected'],got['C'],got['Q'],got['J']),(3,2,1,0,1,0))

if __name__=='__main__':unittest.main()
