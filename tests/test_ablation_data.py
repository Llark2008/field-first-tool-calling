"""Catch changed labels/exposures, bad omission quotas and semantic boundary errors."""
import copy
import unittest
from collections import Counter, defaultdict

from toolcall_lab.formal_data import build_formal_datasets, training_row
from toolcall_lab.ablation_data import rewrite_rows, build_dev, audit_rows
from toolcall_lab.ablation_phrases import render_train, render_dev


class AblationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d0=build_formal_datasets()['B']
        cls.d1,cls.audit=rewrite_rows(cls.d0)
        cls.dev=build_dev()

    def test_only_selected_pair_requests_change(self):
        self.assertEqual([r['id'] for r in self.d0],[r['id'] for r in self.d1])
        changed=[(a,b) for a,b in zip(self.d0,self.d1) if a['request']!=b['request']]
        self.assertEqual(len(changed),360)
        self.assertEqual(len({a['intent_group'] for a,b in changed}),180)
        self.assertEqual(Counter(a['target_tool'] for a,b in changed if a['condition']=='complete'),
            dict(rank_runs=60,summarize_metric=50,compare_algorithms=30,list_runs=20,get_run_config=10,get_metric_series=10))
        for a,b in zip(self.d0,self.d1):
            self.assertEqual({k:v for k,v in a.items() if k!='request'},{k:v for k,v in b.items() if k!='request'})

    def test_targets_and_exposure_ids_match(self):
        self.assertEqual([training_row(x)['messages'][-1] for x in self.d0],
                         [training_row(x)['messages'][-1] for x in self.d1])
        self.assertEqual(Counter(r['gold']['action'] for r in self.d1),dict(call=600,clarify=480,unsupported=120))
        d1,review=rewrite_rows(self.d0)
        self.assertEqual((d1,review),(self.d1,self.audit))

    def test_dev_pair_and_omission_quotas(self):
        self.assertEqual(Counter(r['condition'] for r in self.dev),dict(complete=60,natural=48,unsupported=12))
        self.assertEqual(Counter(r['difficulty'] for r in self.dev if r['condition']=='complete'),dict(routine=30,boundary=30))
        missing=[r for r in self.dev if r['condition']=='natural']
        self.assertEqual(Counter(len(r['missing_fields']) for r in missing),{1:36,2:12})
        self.assertEqual(sum(bool(set(r['missing_fields'])&{'metric','stat','order','limit'}) for r in missing),20)
        groups=defaultdict(list)
        for r in self.dev: groups[r['intent_group']].append(r)
        self.assertEqual(sum(len(v)==2 for v in groups.values()),48)
        for v in groups.values():
            self.assertEqual(len({r['template_group'] for r in v}),1)
        self.assertEqual(len({r['template_group'] for r in self.dev}),28)
        self.assertFalse({r['template_group'] for r in self.dev}&{r['template_group'] for r in self.d0})
        self.assertEqual(audit_rows(self.d0,self.d1,self.dev,self.audit)['status'],'passed')

    def test_entailed_optional_default_fields_are_not_missing(self):
        args=dict(project='atlas',environment='Hopper-v4',metric='final_return',order='desc',limit=2)
        self.assertIn('two runs with highest',render_train('rank_runs',args,0))
        missing=dict(args); del missing['order']
        text=render_train('rank_runs',missing,0).lower()
        self.assertNotIn('highest',text); self.assertNotIn('lowest',text)
        self.assertNotIn('mean',render_train('summarize_metric',dict(project='atlas',metric='final_return'),0).lower())
        config=[r for r in self.dev if r['target_tool']=='get_run_config' and r['condition']=='complete']
        self.assertTrue(all(set(r['supplied_arguments'])=={'run_id'} for r in config))
        series=[r for r in self.dev if r['target_tool']=='get_metric_series' and r['condition']=='complete']
        self.assertTrue(any('max_points' not in r['supplied_arguments'] for r in series))
        self.assertTrue(all(r['gold']['action']=='call' for r in config+series))

    def test_corrupt_label_and_changed_nonrequest_are_rejected(self):
        bad=copy.deepcopy(self.d1); bad[0]['context']+=' changed'
        with self.assertRaises(ValueError): audit_rows(self.d0,bad,self.dev,self.audit)
        dev=copy.deepcopy(self.dev); r=next(r for r in dev if r['condition']=='natural');r['gold']['missing_fields']=[]
        with self.assertRaises(ValueError): audit_rows(self.d0,self.d1,dev,self.audit)

if __name__=='__main__': unittest.main()
