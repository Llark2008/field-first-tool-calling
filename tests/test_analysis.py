import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from toolcall_lab.analyze_expanded import summarize, paired_effect, load_run
from toolcall_lab.expanded import expanded_cases
from toolcall_lab.run import select_cases


def row(group, condition, order, success, action='call', missing=False):
    return {'case_id': f'{group}-{condition}-{order}', 'group_id': group, 'condition': condition,
            'order_id': order, 'target_tool': 'summarize_metric', 'missing_field': 'project',
            'gold': {'action': 'clarify' if missing else 'call'},
            'score': {'success': success, 'schema_valid': True, 'parsed': True, 'action': {'action': action}},
            'seconds': 1.0}


class AnalysisTests(unittest.TestCase):
    def test_suite_selection_does_not_silently_keep_pilot_limit(self):
        self.assertEqual(len(select_cases('expanded')), 288)
        self.assertEqual(len(select_cases('pilot')), 40)
        self.assertEqual(len(select_cases('expanded', 3)), 3)
        with self.assertRaises(ValueError):
            select_cases('expanded', 0)

    def test_pairing_uses_group_and_order_not_list_position(self):
        rows = [row('a', 'complete_original', i, True) for i in (0, 1)]
        rows += [row('b', 'complete_original', i, True) for i in (0, 1)]
        rows += [row('b', 'complete_semantic_a', i, True) for i in (0, 1)]
        rows += [row('a', 'complete_semantic_a', i, False) for i in (0, 1)]
        result = paired_effect(rows, 'complete_semantic_a')
        self.assertEqual(result['n_pairs'], 4)
        self.assertEqual(result['n_groups'], 2)
        self.assertEqual(result['delta_pp'], -50.0)
        self.assertEqual(result['correct_to_wrong'], 2)
        self.assertEqual(result['wrong_to_correct'], 0)
        self.assertEqual(result['cluster_ci95_pp'], [-100.0, 0.0])
        self.assertEqual(result, paired_effect(list(reversed(rows)), 'complete_semantic_a'))

    def test_missing_or_duplicate_pairs_are_rejected(self):
        rows = [row('a', 'complete_original', i, True) for i in (0, 1)]
        rows += [row('a', 'complete_semantic_a', 0, False)]
        with self.assertRaises(ValueError):
            paired_effect(rows, 'complete_semantic_a')
        rows.append(row('a', 'complete_semantic_a', 1, False))
        rows.append(copy.deepcopy(rows[-1]))
        with self.assertRaises(ValueError):
            paired_effect(rows, 'complete_semantic_a')

    def test_guessing_is_not_conflated_with_any_wrong_answer(self):
        rows = [row('a', 'missing_natural', 0, False, 'call', True),
                row('b', 'missing_natural', 0, False, 'unsupported', True),
                row('c', 'missing_natural', 0, False, 'clarify', True),
                row('d', 'missing_natural', 0, True, 'clarify', True)]
        result = summarize(rows)['by_condition']['missing_natural']
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['premature_calls'], 1)
        self.assertEqual(result['clarify_actions'], 2)
        self.assertEqual(result['success'], 1)
        self.assertEqual(result['premature_call_rate'], 0.25)

    def test_loader_rescores_token_limit_without_overwriting_raw_scores(self):
        cases = expanded_cases()
        records = []
        for case in cases:
            record = {key: case[key] for key in ('group_id', 'condition', 'order_id', 'target_tool', 'missing_field', 'gold', 'request')}
            record.update(case_id=case['id'], output=json.dumps(case['gold']), finish_reason='stop', score={'success': True})
            records.append(record)
        records[0]['finish_reason'] = 'length'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'cases.json').write_text(json.dumps(cases))
            (path / 'metadata.json').write_text(json.dumps({'status': 'complete', 'suite': 'expanded',
                'case_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}))
            (path / 'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            metadata, rescored = load_run(path)
            self.assertFalse(rescored[0]['score']['success'])
            self.assertEqual(metadata['rescored_success_changes'], [records[0]['case_id']])
            self.assertTrue(json.loads((path / 'responses.jsonl').read_text().splitlines()[0])['score']['success'])


if __name__ == '__main__':
    unittest.main()
