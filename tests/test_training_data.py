import json
import unittest
from collections import Counter

from jsonschema import Draft202012Validator
from toolcall_lab.training_data import build_datasets


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_datasets('no_examples')

    def test_template_families_never_cross_splits(self):
        train = {r['template_group'] for r in self.data['natural']['train']}
        valid = {r['template_group'] for r in self.data['natural']['valid']}
        self.assertTrue(train)
        self.assertTrue(valid)
        self.assertFalse(train & valid)
        requests = [{r['request'] for r in self.data['natural'][split]} for split in ('train', 'valid')]
        self.assertFalse(requests[0] & requests[1])

    def test_variants_share_labels_and_nonmissing_examples(self):
        for split in ('train', 'valid'):
            a, b = self.data['explicit'][split], self.data['natural'][split]
            self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
            self.assertEqual(Counter(r['gold']['action'] for r in a), Counter(r['gold']['action'] for r in b))
            for left, right in zip(a, b):
                self.assertEqual(left['gold'], right['gold'])
                self.assertEqual(left['tools'], right['tools'])
                if left['gold']['action'] != 'clarify':
                    self.assertEqual(left['messages'], right['messages'])
                else:
                    self.assertTrue(left['request'].startswith(right['request']))
                    self.assertNotEqual(left['request'], right['request'])

    def test_labels_follow_schema_and_preserve_optional_defaults(self):
        saw_default = saw_optional = saw_multiple = False
        for variant in self.data.values():
            for rows in variant.values():
                for row in rows:
                    self.assertEqual(json.loads(row['messages'][-1]['content']), row['gold'])
                    self.assertTrue(all(t['name'] in ('list_runs', 'summarize_metric', 'rank_runs',
                                                     'compare_algorithms', 'get_run_config', 'get_metric_series') for t in row['tools']))
                    if row['gold']['action'] == 'unsupported':
                        continue
                    schema = next(t for t in row['tools'] if t['name'] == row['target_tool'])
                    if row['gold']['action'] == 'call':
                        Draft202012Validator(schema['parameters']).validate(row['gold']['arguments'])
                        if row['target_tool'] == 'get_metric_series':
                            saw_default |= 'max_points' not in row['gold']['arguments']
                        if row['target_tool'] == 'list_runs':
                            saw_optional |= set(row['gold']['arguments']) == {'project'}
                    else:
                        missing = row['gold']['missing_fields']
                        self.assertTrue(set(missing) <= set(schema['parameters']['required']))
                        self.assertTrue(all('default' not in schema['parameters']['properties'][f] for f in missing))
                        saw_multiple |= len(missing) > 1
        self.assertTrue(saw_default and saw_optional and saw_multiple)

    def test_hand_checked_call_and_omission_pair(self):
        rows = {r['id']: r for r in self.data['natural']['train']}
        complete = rows['summarize_metric-f0-i0-complete']
        missing = rows['summarize_metric-f0-i0-missing']
        self.assertEqual(complete['gold'], {'action': 'call', 'name': 'summarize_metric', 'arguments': {
            'project': 'cedar-0', 'environment': 'Hopper-v4', 'algorithm': 'SAC', 'metric': 'final_return', 'stat': 'mean'}})
        self.assertEqual(missing['gold'], {'action': 'clarify', 'missing_fields': ['environment', 'project']})
        self.assertNotIn('cedar-0', missing['request'])
        self.assertNotIn('Hopper-v4', missing['request'])
        self.assertIn('average', missing['request'])


if __name__ == '__main__':
    unittest.main()
