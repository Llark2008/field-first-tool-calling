import copy
import unittest
from experiments.fields_epochs import core

class CoreTests(unittest.TestCase):
    def test_schedule_preserves_original_order_and_exact_three_exposures(self):
        ids=['case-'+str(i) for i in range(600)]
        got=core.schedule(ids)
        self.assertEqual(len(got),1800)
        for i in range(3):self.assertEqual(got[600*i:600*(i+1)],ids)
        with self.assertRaises(ValueError):core.schedule(ids[:-1]+[ids[0]])

    def test_learning_rate_is_only_numerical_recipe_difference(self):
        a=core.recipe('A','fp','data',1217);b=core.recipe('B','fp','data',1217)
        self.assertEqual(a['learning_rate'],1e-4);self.assertEqual(b['learning_rate'],5e-5)
        self.assertEqual({k:v for k,v in a.items() if k not in ['arm','learning_rate']},{k:v for k,v in b.items() if k not in ['arm','learning_rate']})
        self.assertEqual(a['max_microsteps'],1800);self.assertEqual(a['checkpoints'],list(range(200,1801,200)))
        self.assertFalse(a['bias_correction']);self.assertEqual(a['normalization'],'sample')

    def test_inherited_optimizer_lr_cannot_silently_override_requested_rate(self):
        core.check_lr(9.999999747378752e-5,1e-4)
        with self.assertRaises(ValueError):core.check_lr(1e-4,5e-5)

    def test_partial_grid_cannot_be_called_complete(self):
        cells={f'{a}-{e}-{s}':{'n':120} for a in ['A','B'] for e in [1,2,3] for s in ['dev','seen']}
        core.require_grid(cells)
        partial=copy.deepcopy(cells);del partial['B-3-seen']
        with self.assertRaises(ValueError):core.require_grid(partial)
        bad=copy.deepcopy(cells);bad['A-1-dev']['n']=119
        with self.assertRaises(ValueError):core.require_grid(bad)

if __name__=='__main__':unittest.main()
