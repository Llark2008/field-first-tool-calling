import copy
import unittest
from toolcall_lab.ablation_data import build_dev
from toolcall_lab.mechanism_analysis import analyze,ARMS,STEPS,REFERENCES
from test_ablation_analysis import outputs


class MechanismAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cases=build_dev();self.rows=outputs(self.cases)
        self.grid={(a,s):copy.deepcopy(self.rows) for a in ARMS for s in STEPS}
        self.refs={a:self.rows for a in REFERENCES}
        for a,n in [('M0',10),('M1',6),('M2',8)]:
            for r in [r for r in self.grid[a,600] if r['condition']=='natural'][:n]:
                r['score'].update(success=False,action={'action':'call'})
        for r in [r for r in self.grid['M0',600] if r['condition']=='complete'][:2]:
            r['score'].update(success=False,action={'action':'clarify','missing_fields':['order']})

    def test_primary_contrasts_require_joint_capability_improvement(self):
        a=analyze(self.cases,self.grid,self.refs,replicates=30)
        self.assertEqual(set(a['effects']),{'M1-M0','M2-M0'})
        self.assertEqual(a['effects']['M1-M0']['Q']['delta_count'],4)
        self.assertEqual(a['effects']['M1-M0']['F']['delta_count'],-2)
        self.assertTrue(a['verdicts']['M1']['joint_goal'])
        self.assertFalse(a['verdicts']['M2']['joint_goal'])
        for r in [r for r in self.grid['M1',600] if r['condition']=='complete'][:12]:
            r['score'].update(success=False,action={'action':'clarify','missing_fields':['order']})
        a=analyze(self.cases,self.grid,self.refs,replicates=30)
        self.assertFalse(a['verdicts']['M1']['joint_goal'])
        self.assertFalse(a['verdicts']['M1']['capability_gate_passed'])

    def test_incomplete_grid_withholds_verdict_even_if_primary_finished(self):
        del self.grid['M2',200]
        with self.assertRaises(ValueError):analyze(self.cases,self.grid,self.refs,replicates=10)
        a=analyze(self.cases,self.grid,self.refs,complete=False,replicates=10)
        self.assertEqual(a['status'],'incomplete');self.assertEqual(a['missing_cells'],['M2-200'])
        self.assertEqual(a['verdicts'],{});self.assertEqual(a['effects']['M1-M0']['Q']['delta_count'],4)

    def test_wrong_denominators_rejected_even_in_partial_report(self):
        cases=[c for c in self.cases if c['condition']!='unsupported']
        rows=[r for r in self.rows if r['condition']!='unsupported']
        with self.assertRaises(ValueError):analyze(cases,{('M0',200):rows},{},complete=False)
