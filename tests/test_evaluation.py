import copy
import json
import unittest

from toolcall_lab.tasks import pilot_cases, build_messages
from toolcall_lab.evaluation import execute_call, score_response


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case['id']: case for case in pilot_cases()}
        self.case = self.cases['call_mean']

    def test_mean_execution_matches_hand_computed_fixture(self):
        self.assertEqual(execute_call(self.case, self.case['gold']), 200.0)
        result = score_response(self.case, json.dumps(self.case['gold']))
        self.assertTrue(result['success'])

    def test_valid_wrong_environment_is_not_semantic_success(self):
        wrong = copy.deepcopy(self.case['gold'])
        wrong['arguments']['environment'] = 'Walker2d-v4'
        result = score_response(self.case, json.dumps(wrong))
        self.assertTrue(result['schema_valid'])
        self.assertFalse(result['arguments_correct'])
        self.assertFalse(result['success'])
        self.assertEqual(result['execution_result'], 600.0)

    def test_missing_required_argument_and_unknown_keys_fail_schema(self):
        for mutation in ('missing', 'extra'):
            action = copy.deepcopy(self.case['gold'])
            if mutation == 'missing':
                del action['arguments']['project']
            else:
                action['arguments']['invented_parameter'] = 3
            result = score_response(self.case, json.dumps(action))
            self.assertFalse(result['schema_valid'])
            self.assertFalse(result['success'])

    def test_malformed_or_nonfinite_json_is_rejected(self):
        for text in ('not JSON', '{', '[]', '{"action":"call","arguments":NaN}'):
            self.assertFalse(score_response(self.case, text)['success'])

    def test_renamed_interface_preserves_computation_but_rejects_old_name(self):
        renamed = self.cases['renamed_call_mean']
        self.assertEqual(execute_call(renamed, renamed['gold']), 200.0)
        self.assertTrue(score_response(renamed, json.dumps(renamed['gold']))['success'])
        self.assertFalse(score_response(renamed, json.dumps(self.case['gold']))['success'])

    def test_wrong_clarification_fields_fail(self):
        case = self.cases['clarify_project']
        self.assertTrue(score_response(case, '{"action":"clarify","missing_fields":["project"]}')['success'])
        self.assertFalse(score_response(case, '{"action":"clarify","missing_fields":["metric"]}')['success'])
        self.assertFalse(score_response(case, '{"action":"unsupported"}')['success'])

    def test_schema_valid_integer_valued_floats_execute_correctly(self):
        for identifier, key, expected in (
            ('call_rank_desc', 'limit', ['a2', 'a1']),
            ('call_series', 'max_points', [10, 40]),
        ):
            case = self.cases[identifier]
            action = copy.deepcopy(case['gold'])
            action['arguments'][key] = 2.0
            with self.subTest(case=identifier):
                score = score_response(case, json.dumps(action))
                self.assertTrue(score['success'], score)
                self.assertEqual(score['execution_result'], expected)

    def test_unsupported_and_over_refusal_are_distinguished(self):
        self.assertTrue(score_response(self.cases['unsupported_train'], '{"action":"unsupported"}')['success'])
        self.assertFalse(score_response(self.case, '{"action":"unsupported"}')['success'])

    def test_model_messages_exclude_gold_and_expected_results(self):
        case = copy.deepcopy(self.case)
        case['gold'] = {'secret': 'PRIVATE_GOLD_SENTINEL'}
        case['expected_result'] = 'PRIVATE_RESULT_SENTINEL'
        messages = json.dumps(build_messages(case))
        self.assertNotIn('PRIVATE_GOLD_SENTINEL', messages)
        self.assertNotIn('PRIVATE_RESULT_SENTINEL', messages)
        self.assertIn(case['request'], messages)

    def test_pilot_has_no_final_test_only_tools_and_all_gold_scores(self):
        self.assertEqual(len(self.cases), 40)
        for case in self.cases.values():
            for tool in case['tools']:
                self.assertNotIn(tool['name'], ('group_metric', 'count_missing'))
            with self.subTest(case=case['id']):
                self.assertTrue(score_response(case, json.dumps(case['gold']))['success'])


if __name__ == '__main__':
    unittest.main()
