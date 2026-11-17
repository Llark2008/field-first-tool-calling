"""Guard actual optimizer rate, fixed exposure, and incompatible resume rejection."""
import os
import unittest
from toolcall_lab.train_ablation import recipe, validate_config, checkpoint_steps, make_optimizer
from toolcall_lab.train_formal import exposure_schedule


class AblationTrainingTests(unittest.TestCase):
    def test_one_epoch_and_checkpoint_boundaries(self):
        self.assertEqual(checkpoint_steps(1200),(200,400,600,800,1200))
        self.assertEqual(checkpoint_steps(200),(200,))
        with self.assertRaises(ValueError): checkpoint_steps(201)
        self.assertEqual(exposure_schedule(1200,1,20261113),exposure_schedule(1200,2,20261113)[:1200])
        self.assertEqual(1200//recipe(5e-5,1200,(200,400,600,800,1200),'x','data')['grad_accumulation_steps'],150)

    def test_changed_recipe_rejects_resume(self):
        a=recipe(1e-4,1200,(200,400,600,800,1200),'x','data')
        validate_config(a,dict(a))
        for key,value in [('learning_rate',5e-5),('sft_sha256','other'),('checkpoints',[200]),('fingerprint','other')]:
            changed=dict(a);changed[key]=value
            with self.assertRaises(ValueError):validate_config(a,changed)

    @unittest.skipUnless(os.environ.get('TOOLCALL_TEST_MLX')=='1','requires Metal runtime')
    def test_lower_rate_changes_actual_optimizer_update(self):
        import mlx.core as mx
        import mlx.nn as nn
        changes=[]
        for rate in [1e-4,5e-5]:
            model=nn.Linear(1,1,bias=False);model.weight=mx.array([[1.0]])
            optimizer=make_optimizer(rate)
            optimizer.update(model,{'weight':mx.array([[2.0]])});mx.eval(model.parameters(),optimizer.state)
            changes.append(1.0-model.weight.item())
        # Historical MLX Adam defaults to bias_correction=False:
        # first m=0.2, v=0.004, delta=lr*0.2/(sqrt(0.004)+1e-8).
        self.assertAlmostEqual(changes[0],0.0003162277,places=6)
        self.assertAlmostEqual(changes[1],0.0001581139,places=6)
        self.assertGreater(changes[0],changes[1]*1.9)

if __name__=='__main__': unittest.main()
