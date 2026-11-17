import copy
import json
import unittest
from collections import Counter

from toolcall_lab.formal_data import build_formal_datasets, audit_formal_datasets
from toolcall_lab.evaluation import score_response
from toolcall_lab.fixtures import execute_fixture, reference_result


class FormalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_formal_datasets(20261113)

    def test_quotas_and_matched_arms(self):
        d = self.data
        self.assertEqual({k: len(v) for k, v in d.items()}, {'A': 1200, 'B': 1200, 'dev': 300, 'test': 480})
        self.assertEqual(Counter(r['gold']['action'] for r in d['B']), {'call': 600, 'clarify': 480, 'unsupported': 120})
        for a, b in zip(d['A'], d['B']):
            aa, bb = copy.deepcopy(a), copy.deepcopy(b)
            if a['gold']['action'] == 'clarify':
                self.assertNotEqual(aa.pop('request'), bb.pop('request'))
            self.assertEqual(aa, bb)

    def test_shortcuts_removed(self):
        calls = [r for r in self.data['B'] if r['gold']['action'] == 'call']
        ranks = [r['gold']['arguments'] for r in calls if r['gold']['name'] == 'rank_runs']
        self.assertEqual(set((x['metric'], x['order']) for x in ranks),
                         {(m, o) for m in ['final_return', 'peak_return'] for o in ['asc', 'desc']})
        positions = Counter()
        for row in calls:
            args = row['gold']['arguments']
            if 'project' in args:
                schema = next(s for s in row['tools'] if s['name'] == row['gold']['name'])
                positions[schema['parameters']['properties']['project']['enum'].index(args['project'])] += 1
        self.assertEqual(set(positions), {0, 1, 2})
        self.assertLess(max(positions.values()), 200)

    def test_split_isolation_and_final_only_tools(self):
        groups = {s: {r['template_group'] for r in self.data[s]} for s in ['B', 'dev', 'test']}
        self.assertFalse(groups['B'] & groups['dev'] or groups['B'] & groups['test'] or groups['dev'] & groups['test'])
        for split in ['B', 'dev']:
            self.assertFalse({'group_metric', 'count_missing'} & {t['name'] for r in self.data[split] for t in r['tools']})
        self.assertEqual(sum(r['condition'] == 'renamed_complete' for r in self.data['test']), 80)
        self.assertEqual(sum(r['condition'] == 'renamed_natural' for r in self.data['test']), 80)

    def test_optional_default_and_all_reference_results(self):
        self.assertGreaterEqual(sum(r.get('anti_overclarify', False) for r in self.data['B'] if r['gold']['action'] == 'call'), 180)
        for split in ['B', 'dev', 'test']:
            for row in self.data[split]:
                score = score_response(row, json.dumps(row['gold']))
                self.assertTrue(score['success'], (row['id'], score))
                if row['gold']['action'] == 'call':
                    self.assertEqual(execute_fixture(row['fixture'], row['gold'], row.get('bindings')),
                                     reference_result(row['fixture'], row['gold'], row.get('bindings')))

    def test_audit_rejects_corrupted_label_and_reports_coverage(self):
        audit = audit_formal_datasets(self.data)
        self.assertEqual(audit['omission_sizes'], {'1': 360, '2': 120})
        self.assertGreaterEqual(audit['semantic_omissions'], 240)
        bad = copy.deepcopy(self.data)
        row = next(r for r in bad['B'] if r['gold']['action'] == 'call')
        row['expected_result'] = 'corrupted'
        with self.assertRaises(AssertionError):
            audit_formal_datasets(bad)

    def test_executor_only_needs_public_fixture(self):
        fixture = {'runs': [dict(run_id='r1', project='p', environment='e', algorithm='a', final_return=4, peak_return=8),
                            dict(run_id='r2', project='p', environment='e', algorithm='b', final_return=2, peak_return=9)],
                   'configs': {'r1': {'seed': 42}}, 'series': {'r1': [5, 6, 7]},
                   'tables': {'experiments': [{'x': None}, {'x': 3}]}}
        self.assertEqual(execute_fixture(fixture, {'action': 'call', 'name': 'rank_runs', 'arguments': {'project': 'p', 'environment': 'e', 'metric': 'final_return', 'order': 'asc', 'limit': 1}}), ['r2'])
        self.assertEqual(execute_fixture(fixture, {'action': 'call', 'name': 'count_missing', 'arguments': {'table': 'experiments', 'column': 'x'}}), 1)

    def test_small_splits_cover_single_and_double_semantic_omissions(self):
        for split in ['dev', 'test']:
            for name in ['rank_runs', 'summarize_metric', 'compare_algorithms']:
                sizes = {len(r['missing_fields']) for r in self.data[split]
                         if r['condition']=='natural' and r['target_tool']==name}
                self.assertEqual(sizes, {1, 2})


if __name__ == '__main__': unittest.main()
