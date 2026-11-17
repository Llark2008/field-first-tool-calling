import json
import unittest
import tempfile
import shutil
from pathlib import Path
from fieldfirst.verify import verify, validate_alignment
from experiments.fields_sft.data import messages

ROOT = Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_published_predictions_reproduce_every_cell(self):
        result = verify(ROOT)
        self.assertEqual(len(result['cells']), 12)
        self.assertEqual(result['cells']['A-3-dev']['model']['success'], 107)
        self.assertEqual(result['cells']['B-3-dev']['D']['success'], 111)
        self.assertEqual(result['baselines']['internal']['success'], 66)

    def test_incomplete_reordered_and_duplicate_predictions_fail(self):
        cases = [{'id': 'a'}, {'id': 'b'}]
        good = [{'case_id': 'a', 'output': '{}', 'finish_reason': 'stop'},
                {'case_id': 'b', 'output': '{}', 'finish_reason': 'stop'}]
        validate_alignment(cases, good)
        for bad in [good[:1], good[::-1], [good[0], good[0]]]:
            with self.assertRaises(ValueError): validate_alignment(cases, bad)

    def test_missing_published_baseline_metrics_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp)/'release'
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns('.git', '__pycache__'))
            (copy/'results/baselines.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify(copy)

    def test_prompt_cannot_read_labels(self):
        case = json.loads((ROOT/'data/dev-cases.jsonl').read_text().splitlines()[0])
        public = {k: case[k] for k in ['request', 'context', 'tools']}
        self.assertEqual(messages(public, 'F'), messages(case, 'F'))

if __name__ == '__main__': unittest.main()
