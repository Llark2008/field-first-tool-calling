import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.fields_epochs import run,train,evaluate,progress

class ReviewTests(unittest.TestCase):
    def test_inherited_finite_teacher_and_training_loss_changes_are_rejected(self):
        for name in ['teacher-F.jsonl','train-F/events.jsonl']:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"loss":0.1}\n')
                run.write(root/'frozen.json',{'fingerprint':'parent'})
                run.write(root/'report-data.json',{'artifacts':{name:run.sha(p)}})
                run.write(root/'completed.json',{'fingerprint':'parent','report_sha256':run.sha(root/'report-data.json')})
                run.verify_signed_parent(root)
                p.write_text('{"loss":0.2}\n')
                with self.assertRaises(ValueError):run.verify_signed_parent(root)

    def test_interrupted_journal_replacement_preserves_committed_history(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'events.jsonl';p.write_text('{"original":true}\n');before=p.read_bytes()
            with patch.object(Path,'replace',side_effect=InterruptedError('simulated')):
                with self.assertRaises(InterruptedError):train.atomic_events(p,[{'microstep':8}])
            self.assertEqual(p.read_bytes(),before)
            train.atomic_events(p,[{'microstep':8}]);self.assertEqual(json.loads(p.read_text()),{'microstep':8})

    def test_only_torn_uncommitted_tail_can_be_discarded(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'events.jsonl';p.write_text('{"microstep":8}\n{"micro')
            self.assertEqual(train.load_events(p,8),[{'microstep':8}])
            with self.assertRaises(ValueError):train.load_events(p,16)
            p.write_text('{"microstep":8}\nBROKEN\n')
            with self.assertRaises(ValueError):train.load_events(p,8)

    def test_generation_surfaces_latched_resource_error_before_next_token(self):
        class Resource:
            flag=False
            def check(self):
                if self.flag:raise RuntimeError('limit')
        resource=Resource()
        def stream():
            yield SimpleNamespace(text='a')
            resource.flag=True
            yield SimpleNamespace(text='b')
            raise AssertionError('Continued after resource failure')
        with self.assertRaisesRegex(RuntimeError,'limit'):evaluate.consume_generation(stream(),resource)

    def test_browser_script_parses_before_serving_progress(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'page.js';p.write_text(progress.HTML.split('<script>')[1].split('</script>')[0])
            result=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
