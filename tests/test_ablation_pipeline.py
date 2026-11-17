"""Reject stale provenance and mixed-generation evaluation files before reuse."""
import json
import tempfile
import unittest
from pathlib import Path
from toolcall_lab.run_clarification_ablation import validate_evaluation, verify_file_map, verify_checkpoint_bundle, verify_historical_model, finalize_resources
from unittest.mock import patch
from toolcall_lab.resource_monitor import ResourceMonitor
from toolcall_lab.formal_data import sha


class AblationPipelineTests(unittest.TestCase):
    def test_wrong_metadata_or_row_identity_rejected(self):
        cases=[dict(id='a',condition='complete',gold={'action':'call'},template_group='f')]
        rows=[dict(case_id='a',condition='complete',gold={'action':'call'},template_group='f')]
        meta=dict(dataset_fingerprint='one',prompt='v2',adapter_sha256='adapter',name='R1-600')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'eval.jsonl';p.write_text(json.dumps(rows[0])+'\n');p.with_suffix('.meta.json').write_text(json.dumps(meta))
            self.assertEqual(len(validate_evaluation(p,cases,meta)),1)
            for key,value in [('dataset_fingerprint','two'),('prompt','checklist'),('adapter_sha256','other')]:
                changed=dict(meta);changed[key]=value
                with self.assertRaises(ValueError):validate_evaluation(p,cases,changed)
            p.write_text(json.dumps(dict(rows[0],case_id='b'))+'\n')
            with self.assertRaises(ValueError):validate_evaluation(p,cases,meta)

    def test_changed_frozen_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'a';p.write_text('before');hashes={'a':sha(p)}
            verify_file_map(root,hashes);p.write_text('after')
            with self.assertRaises(ValueError):verify_file_map(root,hashes)

    def test_historical_model_and_later_checkpoint_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir();(model/'weights').write_text('original')
            historic={'model_files':{'weights':sha(model/'weights')}}
            verify_historical_model(model,historic);(model/'weights').write_text('changed')
            with self.assertRaises(ValueError):verify_historical_model(model,historic)
            cp=root/'step-0800';cp.mkdir()
            for n in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:(cp/n).write_text('original')
            state=dict(microstep=800,optimizer_updates=100,fingerprint='old',hashes={n:sha(cp/n) for n in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']})
            (cp/'state.json').write_text(json.dumps(state));verify_checkpoint_bundle(cp,'old',800)
            (cp/'adapters.safetensors').write_text('changed')
            with self.assertRaises(ValueError):verify_checkpoint_bundle(cp,'old',800)
            with self.assertRaises(ValueError):verify_checkpoint_bundle(cp,'different',800)

    def test_late_resource_sample_prevents_success(self):
        low=dict(rss_bytes=1024,swap_used_bytes=0);high=dict(rss_bytes=1024,swap_used_bytes=3*1024**3)
        with tempfile.TemporaryDirectory() as d,patch('toolcall_lab.resource_monitor.sample_resources',side_effect=[low,high]):
            resources=ResourceMonitor(Path(d)/'resources.jsonl');resources.start()
            with self.assertRaises(RuntimeError):finalize_resources(resources)

if __name__=='__main__':unittest.main()
