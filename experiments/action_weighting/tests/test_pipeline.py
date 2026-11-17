import importlib
import copy
import unittest


class PipelineTests(unittest.TestCase):
    def setUp(self):
        try: self.report=importlib.import_module('experiments.action_weighting.report')
        except ModuleNotFoundError: self.fail('Weighted comparison pipeline not implemented')

    def test_partial_grid_never_receives_success_or_failure_verdict(self):
        good={k:dict(count=v,n=n) for k,v,n in [('C',56,60),('Q',17,48),('F',3,60),('U',11,12),('boundary',27,30)]}
        self.assertIsNone(self.report.verdict({'W20-600':good},replay_complete=True))
        grid={f'{a}-{s}':good for a in ['C0','W20'] for s in [200,400,600]}
        grid['C0-600']=dict(good,Q=dict(count=13,n=48))
        self.assertIsNone(self.report.verdict(grid,replay_complete=False))
        self.assertTrue(self.report.verdict(grid,replay_complete=True)['joint_pass'])
        grid['W20-600']=dict(good,F=dict(count=4,n=60))
        self.assertFalse(self.report.verdict(grid,replay_complete=True)['joint_pass'])

    def test_html_escapes_raw_examples_and_preserves_data(self):
        data=dict(status='partial',development={},replay={},verdict=None,examples=[dict(request='<script>alert(1)</script>')])
        page=self.report.render(data)
        self.assertNotIn('<script>alert(1)</script>',page)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;',page)
        self.assertIn('尚未完成',page)

    def test_completion_requires_training_and_clean_final_resource_record(self):
        clean=dict(error=None,n_samples=2,initial_swap_used_bytes=100,peak_sampled_rss_bytes=1000,peak_swap_growth_bytes=0)
        training=dict(status='complete',microsteps=600,optimizer_updates=75)
        for tr,rs in [(None,[clean]),(training,[]),(training,[dict(clean,error='too much swap')]),
                      (training,[clean,dict(clean,initial_swap_used_bytes=200)])]:
            with self.assertRaises(ValueError):self.report.require_completion(tr,rs)
        # A recovered run keeps the original baseline; an earlier stop is retained.
        self.report.require_completion(training,[dict(clean,error='interrupted'),clean])

    def test_first_decision_accuracy_does_not_require_continuation_correct(self):
        rows=[dict(gold_action='clarify',action_positions=[0,1],token_details=[dict(correct=True,loss=.3),dict(correct=False,loss=.7)],supervised_tokens=2,loss_sum=1.,correct_tokens=1),
              dict(gold_action='call',action_positions=[0],token_details=[dict(correct=False,loss=.8),dict(correct=True,loss=.2)],supervised_tokens=2,loss_sum=1.,correct_tokens=1)]
        result=self.report.decision_metrics(rows)
        self.assertEqual(result['all']['correct'],1);self.assertEqual(result['all']['n'],2)
        self.assertEqual(result['clarify']['accuracy'],1.)
        self.assertAlmostEqual(result['all']['loss'],.55)

    def test_curve_series_preserves_raw_and_optimized_values(self):
        events={'C0':[dict(optimizer_updates=1,loss=.5)],'W20':[dict(optimizer_updates=1,loss=.2,objective_loss=.8,decision_loss=3.2)]}
        result=self.report.curve_series(events)
        self.assertEqual(result['C0 原始loss'],[[1,.5]])
        self.assertEqual(result['W20 原始loss'],[[1,.2]])
        self.assertEqual(result['W20 加权目标'],[[1,.8]])
        broken=copy.deepcopy(events);broken['W20'][0]['loss']=float('nan')
        with self.assertRaises(ValueError):self.report.curve_series(broken)

    def test_completion_marker_requires_matching_report_identity(self):
        marker=dict(status='complete',fingerprint='one',report_sha256='hash')
        self.report.verify_marker(marker,'one','hash')
        for f,h in [('two','hash'),('one','changed')]:
            with self.assertRaises(ValueError):self.report.verify_marker(marker,f,h)


if __name__=='__main__':unittest.main()
