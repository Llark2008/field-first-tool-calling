import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

class RecoveryTests(unittest.TestCase):
    def test_interruption_before_checkpoint_publish_keeps_prior_checkpoint_resumable(self):
        from experiments.fields_sft import train as m
        from toolcall_lab.formal_data import sha
        config=m.recipe('F','fp','sft',1217)
        def fake_save(path,model,opt,micro,fingerprint):
            path=Path(path);path.mkdir(exist_ok=True)
            for name in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:(path/name).write_text('weights')
            (path/'adapter_config.json').write_text(json.dumps(dict(fine_tune_type='lora',num_layers=8,lora_parameters=m.LORA)))
            (path/'state.json').write_text(json.dumps(dict(fingerprint=fingerprint,microstep=micro,optimizer_updates=micro//8,hashes={n:sha(path/n) for n in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']})))
            if micro==400:raise InterruptedError('Killed after old state published')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(m,'save_training_state',side_effect=fake_save):
                m.save_checkpoint(root/'step-0200',None,None,200,config)
                with self.assertRaises(InterruptedError):m.save_checkpoint(root/'step-0400',None,None,400,config)
            self.assertEqual([p.name for p in m.checkpoints(root,config)],['step-0200'])
            self.assertFalse((root/'step-0400').exists())

    def test_orphaned_resource_violation_cannot_be_hidden_by_clean_summary(self):
        try:m=importlib.import_module('experiments.fields_sft.resources')
        except ModuleNotFoundError:self.fail('Cross-session resource verification not implemented')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);initial=dict(rss_bytes=1,swap_used_bytes=100)
            (root/'resource-baseline.json').write_text(json.dumps(initial))
            (root/'resources-1.jsonl').write_text(json.dumps(dict(initial,swap_used_bytes=100+3*1024**3))+'\n')
            (root/'resources-2.jsonl').write_text(json.dumps(initial)+'\n')
            (root/'resource-summary-2.json').write_text(json.dumps(dict(error=None,session_log='resources-2.jsonl')))
            with self.assertRaises(ValueError):m.verify_sessions(root)

if __name__=='__main__':unittest.main()
