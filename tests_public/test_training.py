import tempfile
import unittest
from pathlib import Path
from fieldfirst.train import make_config, exposure, output_root

class TrainingTests(unittest.TestCase):
    def test_recipe_and_exposure_match_frozen_experiment(self):
        a, b = make_config('A', False), make_config('B', False)
        self.assertEqual(a['learning_rate'], 1e-4)
        self.assertEqual(b['learning_rate'], 5e-5)
        self.assertEqual(a['max_microsteps'], 1800)
        self.assertEqual(a['bias_correction'], False)
        self.assertEqual(a['normalization'], 'sample')
        self.assertEqual(exposure(1800), list(range(600))*3)
        self.assertEqual(make_config('A', True)['checkpoints'], [8])
        with self.assertRaises(ValueError): make_config('C', False)

    def test_runtime_rejects_source_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir()
            external=Path(temp)/'runtime'
            for name in ['../escape','/absolute','..','']:
                with self.assertRaises(ValueError): output_root(source,external,name)
            with self.assertRaises(ValueError): output_root(source,source/'runs','run')
            (external/'runs').mkdir(parents=True)
            (external/'runs'/'link').symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError): output_root(source,external,'link')

if __name__ == '__main__': unittest.main()
