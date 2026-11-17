import copy
import json
import unittest
from collections import Counter

from toolcall_lab.expanded import expanded_cases
from toolcall_lab.tasks import build_messages, pilot_cases
from toolcall_lab.evaluation import execute_call, score_response


class ExpandedTests(unittest.TestCase):
    def setUp(self):
        self.cases = expanded_cases()
        self.index = {(c['group_id'], c['condition'], c['order_id']): c for c in self.cases}

    def test_balanced_crossed_design_has_unique_inputs_and_twenty_four_groups(self):
        self.assertEqual(len(self.cases), 288)
        self.assertEqual(len({c['id'] for c in self.cases}), 288)
        self.assertEqual(len({c['group_id'] for c in self.cases}), 24)
        self.assertEqual(set(Counter(c['condition'] for c in self.cases).values()), {48})
        self.assertEqual(set(Counter(c['group_id'] for c in self.cases).values()), {12})

    def test_gold_answers_match_independent_fixture_execution(self):
        original = self.index[('s01', 'complete_original', 0)]
        self.assertEqual(execute_call(original, original['gold']), 200.0)
        ranking = self.index[('r02', 'complete_original', 0)]
        self.assertEqual(execute_call(ranking, ranking['gold']), ['a3', 'a1'])
        for case in self.cases:
            with self.subTest(case=case['id']):
                self.assertTrue(score_response(case, json.dumps(case['gold']))['success'])

    def test_rename_keeps_request_description_constraints_and_result(self):
        for group in {c['group_id'] for c in self.cases}:
            base = self.index[(group, 'complete_original', 0)]
            for condition in ('complete_semantic_a', 'complete_semantic_b'):
                renamed = self.index[(group, condition, 0)]
                self.assertEqual(base['request'], renamed['request'])
                self.assertEqual(execute_call(base, base['gold']), execute_call(renamed, renamed['gold']))
                for old, new in zip(base['tools'], renamed['tools']):
                    self.assertNotEqual(old['name'], new['name'])
                    self.assertEqual(old['description'], new['description'])
                    binding = renamed['bindings'][new['name']]
                    for renamed_key, old_key in binding['arguments'].items():
                        self.assertEqual(new['parameters']['properties'][renamed_key], old['parameters']['properties'][old_key])

    def test_missing_information_and_candidate_order_remain_paired(self):
        base = self.index[('s01', 'complete_original', 0)]
        missing = self.index[('s01', 'missing_natural', 0)]
        self.assertNotIn('atlas', missing['request'])
        self.assertEqual(missing['gold'], {'action': 'clarify', 'missing_fields': ['project']})
        self.assertEqual(missing['tools'], base['tools'])
        renamed = self.index[('s01', 'missing_semantic_a', 0)]
        self.assertEqual(renamed['gold']['missing_fields'], ['workspace'])
        for group in {c['group_id'] for c in self.cases}:
            zero = self.index[(group, 'complete_original', 0)]
            one = self.index[(group, 'complete_original', 1)]
            names_zero, names_one = ([t['name'] for t in c['tools']] for c in (zero, one))
            self.assertEqual(set(names_zero), set(names_one))
            self.assertNotEqual(names_zero, names_one)

    def test_no_pilot_request_reuse_final_test_tool_or_gold_prompt_leak(self):
        pilot_requests = {c['request'] for c in pilot_cases()}
        for case in self.cases:
            if case['condition'] == 'complete_original':
                self.assertNotIn(case['request'], pilot_requests)
            self.assertTrue(all(t['name'] not in ('group_metric', 'count_missing') for t in case['tools']))
        case = copy.deepcopy(self.cases[0])
        case['gold'] = {'marker': 'DO_NOT_EXPOSE_GOLD'}
        case['expected_result'] = 'DO_NOT_EXPOSE_RESULT'
        for prompt in ('v1', 'v2'):
            visible = json.dumps(build_messages(case, prompt))
            self.assertNotIn('DO_NOT_EXPOSE_GOLD', visible)
            self.assertNotIn('DO_NOT_EXPOSE_RESULT', visible)


if __name__ == '__main__':
    unittest.main()
