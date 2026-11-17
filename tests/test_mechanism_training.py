"""Catch wrong token denominators, optimizer flags, and unsafe state reuse."""
import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from toolcall_lab import train_mechanism as tm
from toolcall_lab.formal_data import sha
from toolcall_lab.train_ablation import validate_config


class MechanismRecipeTests(unittest.TestCase):
    def test_only_one_factor_differs_and_resume_rejects_factor_change(self):
        a=tm.recipe('M1',600,'f','d');b=tm.recipe('M2',600,'f','d')
        self.assertEqual((a['normalization'],a['bias_correction']),('token',False))
        self.assertEqual((b['normalization'],b['bias_correction']),('sample',True))
        for key in ['learning_rate','seed','batch_size','grad_accumulation_steps','lora_parameters','sft_sha256']:
            self.assertEqual(a[key],b[key])
        with self.assertRaises(ValueError):validate_config(a,dict(a,normalization='sample'))
        with self.assertRaises(ValueError):validate_config(a,dict(a,bias_correction=True))
        for arm,limit in [('M1',1200),('M2',200),('M3',600),('M0',600)]:
            with self.assertRaises(ValueError):tm.recipe(arm,limit,'f','d')

    def test_checkpoint_cannot_cross_mechanism_even_with_same_run_fingerprint(self):
        config=tm.recipe('M1',600,'f','d')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for name in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:(p/name).write_bytes(name.encode())
            state=dict(fingerprint='f',microstep=200,optimizer_updates=25,hashes={f.name:sha(f) for f in p.iterdir()},mechanism=tm.checkpoint_identity(config))
            (p/'adapter_config.json').write_text(json.dumps(dict(fine_tune_type='lora',num_layers=8,lora_parameters=tm.LORA)))
            state['adapter_config_sha256']=sha(p/'adapter_config.json')
            (p/'state.json').write_text(json.dumps(state));tm.verify_checkpoint(p,config,200)
            state['mechanism']['arm']='M2';(p/'state.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):tm.verify_checkpoint(p,config,200)

    def test_inference_lora_scale_cannot_change_without_rejection(self):
        config=tm.recipe('M1',600,'f','d')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for n in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:(p/n).write_text('tensor')
            state=dict(fingerprint='f',microstep=200,optimizer_updates=25,hashes={f.name:sha(f) for f in p.iterdir()},mechanism=tm.checkpoint_identity(config))
            (p/'state.json').write_text(json.dumps(state))
            adapter=dict(fine_tune_type='lora',num_layers=8,lora_parameters=tm.LORA)
            (p/'adapter_config.json').write_text(json.dumps(adapter));state['adapter_config_sha256']=sha(p/'adapter_config.json')
            (p/'state.json').write_text(json.dumps(state));tm.verify_checkpoint(p,config,200)
            adapter['lora_parameters']=dict(tm.LORA,scale=200)
            (p/'adapter_config.json').write_text(json.dumps(adapter))
            with self.assertRaises(ValueError):tm.verify_checkpoint(p,config,200)


@unittest.skipUnless(os.environ.get('TOOLCALL_TEST_MLX')=='1','requires Metal runtime')
class MechanismGradientTests(unittest.TestCase):
    def test_token_accumulation_matches_full_batch_with_unequal_answer_lengths(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.weight=mx.arange(35,dtype=mx.float32).reshape(7,5)/20
            def __call__(self,x):return self.weight[x]
        batch=mx.array([[0,1,2,3,4,0,1],[1,2,3,0,0,0,0],[2,3,4,0,1,2,0],[3,4,0,1,0,0,0],
                        [4,0,1,2,3,4,0],[0,2,4,0,0,0,0],[1,3,0,2,4,0,0],[2,4,1,3,0,0,0]])
        lengths=mx.array([[2,7],[1,3],[2,6],[1,4],[3,7],[1,3],[2,5],[1,4]])
        # 5+2+4+3+4+2+3+3 = 26 supervised tokens, independent full-batch reference.
        mask=mx.array([[0,1,1,1,1,1],[1,1,0,0,0,0],[0,1,1,1,1,0],[1,1,1,0,0,0],
                       [0,0,1,1,1,1],[1,1,0,0,0,0],[0,1,1,1,0,0],[1,1,1,0,0,0]])
        def objective(model):return (nn.losses.cross_entropy(model(batch[:,:-1]),batch[:,1:])*mask).sum()/26
        reference=Tiny();refopt=tm.make_optimizer('M1')
        _,expected=nn.value_and_grad(reference,objective)(reference)
        refopt.update(reference,expected);mx.eval(reference.parameters(),refopt.state)
        model=Tiny();optimizer=tm.make_optimizer('M1');step,state=tm.build_step(model,optimizer,'token');prev=None
        for i in range(8):
            loss,ntokens,prev=step(batch[i:i+1],lengths[i:i+1],prev,26,i==7)
            mx.eval(state,loss,ntokens,prev)
            if i<7:self.assertEqual(int(optimizer.step.item()),0)
        self.assertEqual(int(optimizer.step.item()),1);self.assertIsNone(prev)
        for (_,a),(_,b) in zip(tree_flatten(optimizer.state),tree_flatten(refopt.state)):
            self.assertTrue(bool(mx.allclose(a,b,rtol=1e-5,atol=1e-7).item()))
        self.assertTrue(bool(mx.allclose(model.weight,reference.weight,rtol=1e-6,atol=1e-7).item()))
        with self.assertRaises(ValueError):step(batch[:1],lengths[:1],None,0,True)

    def test_loss_sum_excludes_prompt_and_padding_targets(self):
        import mlx.core as mx
        import mlx.nn as nn
        class Table(nn.Module):
            def __init__(self):super().__init__();self.logits=mx.zeros((1,6,3))
            def __call__(self,x):return self.logits
        model=Table();batch=mx.zeros((1,7),dtype=mx.int32);lengths=mx.array([[3,5]])
        (loss,n),grad=nn.value_and_grad(model,tm.token_sum_loss)(model,batch,lengths)
        mx.eval(loss,n,grad)
        self.assertEqual(int(n.item()),2);self.assertAlmostEqual(float(loss.item()),2*math.log(3),places=6)
        for i in [0,1,4,5]:self.assertEqual(float(mx.abs(grad['logits'][0,i]).sum().item()),0)
        for i in [2,3]:self.assertGreater(float(mx.abs(grad['logits'][0,i]).sum().item()),0)

    def test_sample_mode_preserves_mean_of_example_objectives(self):
        import mlx.core as mx
        import mlx.nn as nn
        class Tiny(nn.Module):
            def __init__(self):super().__init__();self.weight=mx.array([[.2,-.3],[.7,-.1]])
            def __call__(self,x):return self.weight[x]
        model=Tiny();optimizer=tm.make_optimizer('M2');step,state=tm.build_step(model,optimizer,'sample')
        batch=mx.array([[0,1,0,1],[1,0,1,0]]);lengths=mx.array([[1,4],[2,3]])
        def reference(m):
            ce=nn.losses.cross_entropy(m(batch[:,:-1]),batch[:,1:])
            return (ce[0].sum()/3+ce[1,1])*4/8
        _,expected=nn.value_and_grad(model,reference)(model)
        clone=Tiny();refopt=tm.make_optimizer('M2');refopt.update(clone,expected);mx.eval(clone.parameters(),refopt.state)
        prev=None
        for i in range(8):
            j=i%2;loss,n,prev=step(batch[j:j+1],lengths[j:j+1],prev,16,i==7);mx.eval(state,loss,n,prev)
        self.assertTrue(bool(mx.allclose(optimizer.state['weight']['m'],refopt.state['weight']['m'],rtol=1e-6,atol=1e-7).item()))

    def test_bias_correction_flag_changes_real_first_and_second_updates(self):
        import mlx.core as mx
        import mlx.nn as nn
        for arm,expected in [('M0',[.000316227716,.000741186856]),('M1',[.000316227716,.000741186856]),('M2',[.0001,.0002])]:
            model=nn.Linear(1,1,bias=False);model.weight=mx.array([[1.0]]);optimizer=tm.make_optimizer(arm)
            for want in expected:
                optimizer.update(model,{'weight':mx.array([[2.]])});mx.eval(model.parameters(),optimizer.state)
                self.assertAlmostEqual(1-model.weight.item(),want,places=6)


if __name__=='__main__':unittest.main()
