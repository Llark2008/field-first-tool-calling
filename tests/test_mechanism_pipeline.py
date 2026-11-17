import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from toolcall_lab.run_mechanism import verify_reused,verify_training,finalize_resources,rescore,load_available,main
from toolcall_lab.resource_monitor import ResourceMonitor
from toolcall_lab.formal_data import sha


class MechanismPipelineTests(unittest.TestCase):
    def test_reused_raw_meta_metrics_and_parent_freeze_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);names=['frozen.json','dev.jsonl','dev.meta.json','dev.metrics.json']
            for n in names:(p/n).write_text('original')
            frozen=dict(parent_run=str(p),parent_files={n:sha(p/n) for n in names})
            verify_reused(frozen)
            for n in names:
                (p/n).write_text('changed')
                with self.assertRaises(ValueError):verify_reused(frozen)
                (p/n).write_text('original')

    def test_summary_cannot_claim_unfinished_or_mismatched_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen={'fingerprint':'f','dataset':str(root)}
            (root/'D0-sft.jsonl').write_text('data')
            train=root/'train-M1';train.mkdir()
            (train/'summary.json').write_text(json.dumps(dict(status='complete',arm='M2',microsteps=600,optimizer_updates=75)))
            with self.assertRaises((ValueError,FileNotFoundError)):verify_training(root,frozen,'M1')

    def test_final_resource_failure_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as d,patch('toolcall_lab.resource_monitor.sample_resources',side_effect=[dict(rss_bytes=1,swap_used_bytes=0),dict(rss_bytes=1,swap_used_bytes=3*1024**3)]):
            r=ResourceMonitor(Path(d)/'r.jsonl');r.start()
            with self.assertRaises(RuntimeError):finalize_resources(r)

    def test_complete_raw_eval_recovers_missing_metrics_without_new_generation(self):
        from toolcall_lab.ablation_data import build_dev
        from toolcall_lab.controls import source_guard
        from toolcall_lab.generation_scoring import score_generation
        from toolcall_lab.evaluate_formal import failure_type,summarize
        cases=build_dev();rows=[]
        for c in cases:
            text=json.dumps(c['gold']);score=score_generation(c,text,'stop')
            guard=source_guard(c['request'],c['tools'],score['action']) if c['gold']['action']=='call' else dict(blocked=False,ungrounded_fields=[])
            r=dict(case_id=c['id'],condition=c['condition'],gold=c['gold'],output=text,finish_reason='stop',score=score,guard=guard,seconds=1)
            r['failure_type']=failure_type(r);rows.append(r)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'eval.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));before=sha(p)
            result=rescore(cases,rows,p,repair_missing_metrics=True)
            self.assertEqual(result['rows'],120);self.assertEqual(sha(p),before)
            self.assertEqual(json.loads(p.with_suffix('.metrics.json').read_text()),summarize(rows))
            p.with_suffix('.metrics.json').write_text('{}')
            with self.assertRaises(ValueError):rescore(cases,rows,p,repair_missing_metrics=True)

    def test_uncertified_partial_run_does_not_publish_M0(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'dev.jsonl').write_text('')
            frozen={'dataset':str(root),'fingerprint':'f'}
            with patch('toolcall_lab.run_mechanism.reused_rows',return_value=({('M0',200):[]},{'base_v2':[]})):
                cases,outputs,refs,partial=load_available(root,frozen)
                self.assertEqual(outputs,{})
                self.assertIn('base_v2',refs)
                (root/'control-equivalence.json').write_text(json.dumps({'status':'failed','fingerprint':'f'}))
                self.assertEqual(load_available(root,frozen)[1],{})

    def test_rerunning_complete_experiment_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=Path(d).resolve();root=runtime/'runs/new';parent=runtime/'runs/old';root.mkdir(parents=True);parent.mkdir()
            (root/'frozen.json').write_text(json.dumps({'parent_run':str(parent),'fingerprint':'f'}))
            originals={'completed.json':'{"status":"complete"}','report-data.json':'original report','results.html':'original html','progress.json':'{"status":"complete"}'}
            for n,text in originals.items():(root/n).write_text(text)
            args=['run','--parent-run',str(parent),'--run-root',str(root),'--stage','run']
            with patch('sys.argv',args),patch('toolcall_lab.run_mechanism.runtime_root',return_value=runtime),patch('toolcall_lab.run_mechanism.configure_runtime'),patch('toolcall_lab.run_mechanism.verify'),patch('toolcall_lab.run_mechanism.run') as runner,patch('toolcall_lab.run_mechanism.finish_report') as report:
                main();runner.assert_not_called();report.assert_not_called()
            for n,text in originals.items():self.assertEqual((root/n).read_text(),text)
