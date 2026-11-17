import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest


class CoreTests(unittest.TestCase):
    def setUp(self):
        try:
            self.core = importlib.import_module('experiments.action_weighting.core')
        except ModuleNotFoundError:
            self.fail('Action-weighted objective and checkpoint validation have not been implemented')

    def test_decision_is_the_first_discriminating_token_not_its_continuation(self):
        # Prompt ids 90,91; answer prefix 10,11,12; clarify = 21,22.
        ids = {'call': 20, 'clarify': 21, 'unsupported': 23}
        self.assertEqual(self.core.decision_index([90,91,10,11,12,21,22,20],2,[10,11,12],ids,'clarify'),5)
        with self.assertRaises(ValueError):
            self.core.decision_index([90,91,10,11,12,21,22],2,[10,11,12],ids,'call')
        with self.assertRaises(ValueError):
            self.core.decision_index([90,91,10,99,12,21,22],2,[10,11,12],ids,'clarify')

    def test_changed_recipe_cannot_resume_existing_checkpoint(self):
        from toolcall_lab.formal_data import sha
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for name in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']:
                (path/name).write_text(name)
            config=self.core.recipe('fingerprint','dataset-hash')
            adapter=dict(fine_tune_type='lora',num_layers=8,lora_parameters=config['lora_parameters'])
            (path/'adapter_config.json').write_text(json.dumps(adapter))
            state=dict(fingerprint='fingerprint',microstep=200,optimizer_updates=25,
                hashes={n:sha(path/n) for n in ['adapters.safetensors','optimizer.safetensors','rng.safetensors']},
                action_weighting=self.core.checkpoint_identity(config),adapter_config_sha256=sha(path/'adapter_config.json'))
            (path/'state.json').write_text(json.dumps(state))
            self.core.verify_checkpoint(path,config,200)
            for key,value in [('action_weight',.3),('bias_correction',True),('sft_sha256','changed')]:
                with self.assertRaises(ValueError):
                    self.core.verify_checkpoint(path,dict(config,**{key:value}),200)
            (path/'adapters.safetensors').write_text('changed')
            with self.assertRaises(ValueError):self.core.verify_checkpoint(path,config,200)


@unittest.skipUnless(os.environ.get('TOOLCALL_TEST_MLX')=='1','requires local Metal')
class GradientTests(CoreTests):
    def test_group_weights_are_exact_and_prompt_padding_gradients_are_zero(self):
        import mlx.core as mx
        import mlx.nn as nn
        class Table(nn.Module):
            def __init__(self):
                super().__init__();self.logits=mx.array([[[0.,0.],[0.,0.],[0.,math.log(3.)],[0.,0.],[0.,0.],[0.,0.]]])
            def __call__(self,x):return self.logits
        m=Table();batch=mx.zeros((1,7),dtype=mx.int32)
        fn=lambda model:self.core.weighted_loss(model,batch,mx.array([[2,5]]),mx.array([3]),.2)
        (loss,aux),grad=nn.value_and_grad(m,fn)(m);mx.eval(loss,aux,grad)
        self.assertAlmostEqual(float(loss.item()),.2*math.log(4)+.8*math.log(2),places=6)
        self.assertAlmostEqual(float(aux[0].item()),(math.log(4)+2*math.log(2))/3,places=6)
        self.assertEqual(int(aux[3].item()),3)
        for i in [0,4,5]:self.assertEqual(float(mx.abs(grad['logits'][0,i]).sum().item()),0)
        # Rest positions each receive .4 weight; action receives .2.
        self.assertAlmostEqual(float(grad['logits'][0,1,0].item()),-.2,places=6)
        self.assertAlmostEqual(float(grad['logits'][0,2,0].item()),-.15,places=6)

    def test_eight_unequal_answers_match_independent_full_batch_gradient(self):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        class Tiny(nn.Module):
            def __init__(self):super().__init__();self.weight=mx.arange(25,dtype=mx.float32).reshape(5,5)/20
            def __call__(self,x):return self.weight[x]
        batch=mx.array([[0,1,2,3,4,0],[1,2,3,4,0,0],[2,3,4,0,1,0],[3,4,0,1,0,0],
                        [4,0,1,2,3,4],[0,2,4,0,0,0],[1,3,0,2,4,0],[2,4,1,3,0,0]])
        lengths=mx.array([[1,6],[2,4],[1,5],[1,4],[2,6],[1,3],[2,5],[1,4]])
        decisions=mx.array([2,2,3,1,4,2,3,2])
        # Explicit weights per shifted target position, independent of production masks.
        weights=mx.array([[.2,.2,.2,.2,.2],[0,.2,.8,0,0],[.8/3,.8/3,.2,.8/3,0],
                          [.2,.4,.4,0,0],[0,.8/3,.8/3,.2,.8/3],[.8,.2,0,0,0],
                          [0,.4,.2,.4,0],[.4,.2,.4,0,0]])
        reference=Tiny();refopt=optim.Adam(learning_rate=1e-4,bias_correction=False)
        objective=lambda m:(nn.losses.cross_entropy(m(batch[:,:-1]),batch[:,1:])*weights).sum()/8
        _,g=nn.value_and_grad(reference,objective)(reference);refopt.update(reference,g);mx.eval(reference.parameters(),refopt.state)
        m=Tiny();opt=optim.Adam(learning_rate=1e-4,bias_correction=False);step,state=self.core.build_step(m,opt,.2);prev=None
        for i in range(8):
            loss,aux,prev,norm=step(batch[i:i+1],lengths[i:i+1],decisions[i:i+1],prev,i==7)
            mx.eval(state,loss,aux,prev,norm)
            if i<7:self.assertEqual(int(opt.step.item()),0)
        self.assertEqual(int(opt.step.item()),1);self.assertIsNone(prev)
        for (_,a),(_,b) in zip(tree_flatten(opt.state),tree_flatten(refopt.state)):
            self.assertTrue(bool(mx.allclose(a,b,rtol=1e-5,atol=1e-7).item()))
        self.assertTrue(bool(mx.allclose(m.weight,reference.weight,rtol=1e-6,atol=1e-7).item()))
        self.assertGreater(float(norm.item()),0)


if __name__=='__main__':unittest.main()
