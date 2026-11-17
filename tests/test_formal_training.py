import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from toolcall_lab.train_formal import exposure_schedule, select_checkpoint, verify_frozen, save_training_state, restore_training_state
from toolcall_lab.evaluate_formal import summarize, failure_type


def row(condition, success=True, action='call', valid=True):
    return dict(condition=condition, score=dict(success=success, schema_valid=valid,
                truncated=False, action={'action':action}), gold={'action':'call' if condition=='complete' else 'clarify'})


class FormalTrainingTests(unittest.TestCase):
    def test_matched_schedule_has_every_example_once_per_epoch(self):
        s=exposure_schedule(1200, 2, 20261113)
        self.assertEqual(len(s), 2400)
        self.assertEqual(set(s[:1200]), set(range(1200)))
        self.assertEqual(set(s[1200:]), set(range(1200)))
        self.assertEqual(s, exposure_schedule(1200, 2, 20261113))
        self.assertNotEqual(s[:1200], s[1200:])

    def test_shared_step_gate_and_earlier_tie(self):
        baseline={'complete':dict(n=100,success=90)}
        def metrics(c,n,u):
            return {'complete':dict(n=100,success=c),'natural':dict(n=100,success=n),'unsupported':dict(n=20,success=u)}
        a={600:metrics(84,95,20),1200:metrics(90,70,18),2400:metrics(90,70,18)}
        b={600:metrics(95,95,20),1200:metrics(85,80,18),2400:metrics(85,80,18)}
        self.assertEqual(select_checkpoint(baseline,a,b)['selected_step'],1200)
        a={k:metrics(80,100,20) for k in a}
        self.assertIsNone(select_checkpoint(baseline,a,b)['selected_step'])

    def test_invalid_and_false_clarify_are_separate(self):
        rows=[row('complete'),row('complete',False,'clarify'),row('complete',False,'call',False),row('natural',False,'call')]
        m=summarize(rows)
        self.assertEqual(m['complete']['n'],3)
        self.assertEqual(m['complete']['false_clarify'],1)
        self.assertEqual(m['complete']['invalid'],1)
        self.assertEqual(m['natural']['premature_call'],1)
        self.assertEqual(failure_type(rows[2]),'invalid')

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'x').write_text('changed')
            manifest={'files':{'x':{'sha256':'wrong'}}}
            with self.assertRaises(ValueError): verify_frozen(root,manifest)

    def test_block_is_not_a_correct_clarification(self):
        r=row('natural',False,'call'); r['guard']={'blocked':True}
        m=summarize([r])['natural']
        self.assertEqual((m['blocked_calls'],m['correct_clarifications'],m['system_success'],m['allowed_premature_calls']),(1,0,0,0))

    @unittest.skipUnless(os.environ.get('TOOLCALL_TEST_MLX')=='1','requires Metal runtime')
    def test_full_optimizer_rng_resume_matches_uninterrupted(self):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from functools import partial
        model=nn.Linear(2,1); optimizer=optim.Adam(learning_rate=1e-4)
        mx.random.seed(44)
        def compiled_step(m,o):
            state=[m.state,o.state,mx.random.state]
            value_grad=nn.value_and_grad(m,lambda m:mx.sum(m(mx.array([[1.0,2.0]]))**2))
            @partial(mx.compile,inputs=state,outputs=state)
            def run():
                value,grad=value_grad(m); o.update(m,grad); return value
            def evaluated():
                value=run(); mx.eval(value,state)
            return evaluated
        step=compiled_step(model,optimizer)
        step()
        with tempfile.TemporaryDirectory() as tmp:
            save_training_state(Path(tmp),model,optimizer,8,'fingerprint')
            rng_expected=mx.random.uniform(shape=(5,)); mx.eval(rng_expected)
            step()
            expected=dict(tree_flatten(model.parameters()))
            other=nn.Linear(2,1); other_opt=optim.Adam(learning_rate=1e-4)
            self.assertEqual(restore_training_state(Path(tmp),other,other_opt,'fingerprint'),8)
            rng_actual=mx.random.uniform(shape=(5,)); mx.eval(rng_actual)
            self.assertTrue(bool(mx.array_equal(rng_expected,rng_actual)))
            compiled_step(other,other_opt)()
            for key,value in tree_flatten(other.parameters()):
                self.assertTrue(bool(mx.array_equal(expected[key],value)))
            with self.assertRaises(ValueError): restore_training_state(Path(tmp),other,other_opt,'changed')


if __name__=='__main__': unittest.main()
