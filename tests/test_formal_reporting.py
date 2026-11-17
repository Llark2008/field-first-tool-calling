import unittest
from toolcall_lab.report_formal import paired_interval, aligned, transition_counts, curated_cases, transfer_results, clarification_field_errors, PAGE


def row(i,ok,group=None):
    return dict(case_id=str(i),group_id=str(i),template_group=group or str(i),condition='natural',
        missing_fields=['project'],gold={'action':'clarify'},score={'success':ok,'schema_valid':True},failure_type='correct' if ok else 'premature_call')


class FormalReportingTests(unittest.TestCase):
    def test_paired_alignment_rejects_missing_or_duplicate(self):
        a=[row(1,True),row(2,False)]
        self.assertEqual(len(aligned(a,list(reversed(a)))),2)
        with self.assertRaises(ValueError): aligned(a,a[:1])
        with self.assertRaises(ValueError): aligned(a,[a[0],a[0]])

    def test_family_bootstrap_preserves_related_inputs(self):
        a=[row(i,False,'family'+str(i//2)) for i in range(8)]
        b=[row(i,True,'family'+str(i//2)) for i in range(8)]
        ci=paired_interval(a,b,replicates=100)
        self.assertEqual(ci['n'],8); self.assertEqual(ci['families'],4)
        self.assertEqual(ci['delta_pp'],100); self.assertEqual(ci['ci95_pp'],[100,100])
        b[0]['template_group']='wrong'
        with self.assertRaises(ValueError): paired_interval(a,b,replicates=10)

    def test_all_four_transitions_and_curated_provenance(self):
        a=[row(0,False),row(1,True),row(2,False),row(3,True)]
        b=[row(0,True),row(1,False),row(2,False),row(3,True)]
        self.assertEqual(transition_counts(a,b),{'wrong→right':1,'right→wrong':1,'wrong→wrong':1,'right→right':1})
        cases=[dict(id=str(i),condition='natural',target_tool='x',request='example',tools=[],gold={'action':'clarify'},missing_fields=['project']) for i in range(4)]
        chosen=curated_cases(cases,{'base_v2':a,'B_v2':b})
        self.assertEqual({r['id'] for r in chosen},{'0','1','2','3'})
        self.assertTrue(all(r['selection_reason'] for r in chosen))

    def test_transfer_tools_with_opposite_effects_are_not_pooled(self):
        base=[row(i,i>=2,'tool'+str(i//2)) for i in range(4)]
        adapted=[row(i,i<2,'tool'+str(i//2)) for i in range(4)]
        for rows in [base,adapted]:
            for r in rows:
                r['target_tool']='group_metric' if int(r['case_id'])<2 else 'count_missing'
                r['condition']='newtool_complete'
        result=transfer_results({'base_v2':base,'A_v2':base,'B_v2':adapted})
        self.assertEqual(result['group_metric']['comparisons']['B_vs_base_v2']['newtool_complete']['delta_pp'],100)
        self.assertEqual(result['count_missing']['comparisons']['B_vs_base_v2']['newtool_complete']['delta_pp'],-100)
        self.assertEqual(result['count_missing']['metrics']['B_v2']['newtool_complete']['n'],2)

    def test_field_errors_do_not_absorb_invalid_or_nonclarify_outputs(self):
        rows=[]
        for i,(gold,pred,valid,action,ok) in enumerate([
            (['project','metric'],['project'],True,'clarify',False),
            (['project'],['project','run_id'],True,'clarify',False),
            (['project'],[],False,'clarify',False),
            (['project'],[],True,'unsupported',False),
            (['project'],['project'],True,'clarify',True)]):
            r=row(i,ok);r['gold']['missing_fields']=gold;r['missing_fields']=gold
            r['score'].update(schema_valid=valid,action={'action':action,'missing_fields':pred}); rows.append(r)
        result=clarification_field_errors(rows)
        self.assertEqual(result['missing_required_field_counts'],{'metric':1})
        self.assertEqual(result['extra_field_counts'],{'run_id':1})
        self.assertEqual((result['valid_clarify_outputs'],result['invalid_outputs'],result['valid_nonclarify_outputs']),(3,1,1))
        self.assertEqual(result['conditional_whole_action_accuracy']['project']['exact_whole_action_correct'],1)

    def test_html_exposes_aggregate_transfer_guard_and_field_panels(self):
        for label in ['两个具体新工具分别评测','原始与来源校验后成功率','澄清字段遗漏与额外字段','擅自调用变化（越低越好）']:
            self.assertIn(label,PAGE)


if __name__=='__main__': unittest.main()
