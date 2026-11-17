import copy
import unittest
from toolcall_lab.ablation_data import build_dev
from toolcall_lab.analyze_clarification_ablation import (metrics_for,analyze_pilot,analyze_available,select_candidate,gate_candidate,regression_decision,historical_complete_counts)


def outputs(cases):
    return [dict(case_id=c['id'],group_id=c['group_id'],template_group=c['template_group'],condition=c['condition'],
        target_tool=c['target_tool'],missing_fields=c['missing_fields'],gold=c['gold'],
        score=dict(success=True,schema_valid=True,truncated=False,action=c['gold']),guard={'blocked':False}) for c in cases]


class AblationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=build_dev();cls.rows=outputs(cls.cases)

    def test_pair_requires_both_correct(self):
        rows=copy.deepcopy(self.rows)
        miss=next(r for r in rows if r['condition']=='natural')
        complete=next(r for r in rows if r['group_id']==miss['group_id'] and r['condition']=='complete')
        complete['score'].update(success=False,action=dict(action='clarify',missing_fields=['order']))
        m=metrics_for(self.cases,rows)
        self.assertEqual((m['C']['count'],m['Q']['count'],m['J']['count'],m['F']['count']),(59,48,47,1))
        miss['score'].update(success=False,action=dict(action='call'));miss['guard']['blocked']=True
        m=metrics_for(self.cases,rows)
        self.assertEqual((m['Q']['count'],m['J']['count'],m['P']['count']),(47,47,1))

    def test_gate_uses_integer_denominators(self):
        def metric(n,d):return dict(count=n,n=d,rate=n/d)
        m={k:metric(n,d) for k,n,d in [('C',54,60),('Q',20,48),('F',3,60),('J',20,48),('U',11,12),('boundary',27,30)]}
        reference=dict(m,C=metric(57,60),U=metric(12,12),boundary=metric(30,30))
        r0=dict(m,C=metric(56,60))
        g=gate_candidate(m,[reference],r0)
        self.assertEqual(g['complete_floor_count'],54);self.assertTrue(g['eligible'])
        self.assertFalse(gate_candidate(dict(m,C=metric(53,60)),[reference],r0)['eligible'])
        self.assertFalse(gate_candidate(dict(m,F=metric(4,60)),[reference],r0)['eligible'])

    def test_ineligible_all_clarify_is_rejected(self):
        def candidate(arm,eligible,q,j,f,c,step):
            return dict(arm=arm,microstep=step,gate={'eligible':eligible},metrics={k:{'count':v} for k,v in dict(Q=q,J=j,F=f,C=c).items()})
        rows=[candidate('R0',True,10,9,1,58,600),candidate('R1',False,48,0,60,0,1200),candidate('R2',True,12,11,1,58,600)]
        self.assertEqual(select_candidate({'candidates':rows})['selected']['arm'],'R2')
        self.assertIsNone(select_candidate({'candidates':[rows[1]]})['selected'])

    def test_same_step_factor_deltas_and_interaction(self):
        grid={(a,s):copy.deepcopy(self.rows) for a in ['R0','R1','R2','R3'] for s in [200,400,600,800,1200]}
        for a,nwrong in [('R0',10),('R1',6),('R2',4),('R3',2)]:
            for r in [r for r in grid[a,600] if r['condition']=='natural'][:nwrong]:r['score'].update(success=False,action={'action':'call'})
        refs={k:self.rows for k in ['base_v2','base_no_examples','base_checklist','A75']}
        a=analyze_pilot(self.cases,grid,refs,replicates=20)
        self.assertEqual(a['effects']['R1-R0']['Q']['delta_count'],4)
        self.assertEqual(a['effects']['R2-R0']['Q']['delta_count'],6)
        self.assertEqual(a['effects']['R3-R2']['Q']['delta_count'],2)
        self.assertEqual(a['effects']['R3-R1']['Q']['delta_count'],4)
        self.assertEqual(a['effects']['interaction']['Q']['delta_count'],-2)
        self.assertTrue(a['effects']['R1-R0']['practical_change']['higher_Q'])
        del grid['R3',1200]
        with self.assertRaises(ValueError):analyze_pilot(self.cases,grid,refs,replicates=20)

    def test_regression_failure_does_not_search_runner_up(self):
        counts={'original':38,'semantic_a':39,'semantic_b':38}
        r=regression_decision(counts)
        self.assertFalse(r['passed']);self.assertEqual(r['next_action'],'stop_promotion')
        self.assertTrue(regression_decision(dict(counts,semantic_b=39))['passed'])

    def test_historical_real_condition_names(self):
        from toolcall_lab.expanded import expanded_cases
        from collections import Counter
        counts=Counter(c['condition'] for c in expanded_cases())
        summary={k:dict(n=v,success=v) for k,v in counts.items()}
        self.assertEqual(historical_complete_counts(summary),dict(original=48,semantic_a=48,semantic_b=48))

    def test_incomplete_grid_reports_only_common_points_without_selection(self):
        grid={(a,s):copy.deepcopy(self.rows) for a in ['R0','R1','R2','R3'] for s in [200,400,600,800,1200]}
        del grid['R3',1200]
        refs={k:self.rows for k in ['base_v2','base_no_examples','base_checklist','A75']}
        result=analyze_available(self.cases,grid,refs,replicates=20)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['common_updates'],[25,50,75,100])
        self.assertEqual(result['missing_cells'],['R3-1200'])
        self.assertIsNone(result['selection']['selected'])
        self.assertEqual(result['effects']['R1-R0']['C']['delta_count'],0)

    def test_duplicate_missing_and_mismatched_rows_rejected(self):
        for rows in [self.rows[:-1],self.rows+[self.rows[0]]]:
            with self.assertRaises(ValueError):metrics_for(self.cases,rows)
        rows=copy.deepcopy(self.rows);rows[0]['template_group']='other'
        with self.assertRaises(ValueError):metrics_for(self.cases,rows)

if __name__=='__main__':unittest.main()
