import copy
import unittest

from toolcall_lab.tasks import TOOLS, build_messages
from toolcall_lab.expanded import expanded_cases, semantic_rename
from toolcall_lab.controls import build_control_messages, source_guard, guard_counts


class ControlsTests(unittest.TestCase):
    def test_no_examples_preserves_task_without_demonstration_values(self):
        case = next(c for c in expanded_cases() if c['id'] == 'expanded-s03-missing_natural-o0')
        old = build_messages(case, 'v2')
        new = build_control_messages(case, 'no_examples')
        self.assertEqual(new[-1], old[-1])
        self.assertEqual([m['role'] for m in new], ['system', 'user'])
        self.assertNotIn('example_run', new[0]['content'])
        self.assertNotIn('atlas', new[0]['content'])
        self.assertIn('Never invent', new[0]['content'])

    def test_inventory_cannot_justify_invented_environment(self):
        action = dict(action='call', name='summarize_metric', arguments=dict(
            project='atlas', environment='Walker2d-v4', algorithm='SAC', metric='peak_return', stat='mean'))
        result = source_guard('Average peak_return for atlas SAC experiments.', [TOOLS['summarize_metric']], action)
        self.assertEqual(result['ungrounded_fields'], ['environment'])
        self.assertTrue(result['blocked'])

    def test_identifier_boundaries_and_case(self):
        action = dict(action='call', name='get_run_config', arguments={'run_id': 'a1'})
        self.assertTrue(source_guard('Read a10 settings.', [TOOLS['get_run_config']], action)['blocked'])
        self.assertFalse(source_guard('Read A1 settings.', [TOOLS['get_run_config']], action)['blocked'])

    def test_semantic_parameters_and_optional_defaults_are_not_entity_checks(self):
        schema = TOOLS['rank_runs']
        action = dict(action='call', name='rank_runs', arguments=dict(project='atlas', environment='Hopper-v4',
                      metric='final_return', order='desc', limit=2))
        self.assertFalse(source_guard('The top two atlas Hopper-v4 trials by final return.', [schema], action)['blocked'])
        action = dict(action='call', name='get_metric_series', arguments=dict(run_id='a1', metric='episode_return', max_points=3))
        self.assertFalse(source_guard('Read the episode_return trace of a1.', [TOOLS['get_metric_series']], action)['blocked'])

    def test_renamed_fields_use_visible_descriptions(self):
        case = next(c for c in expanded_cases() if c['id'] == 'expanded-c01-complete_original-o0')
        renamed = semantic_rename(case, 'a')
        action = renamed['gold']
        self.assertEqual(source_guard('Read my trial settings.', renamed['tools'], action)['ungrounded_fields'], ['trial_id'])
        self.assertFalse(source_guard(renamed['request'], renamed['tools'], action)['blocked'])

    def test_blocked_wrong_call_is_not_a_correct_clarification(self):
        case = next(c for c in expanded_cases() if c['id'] == 'expanded-c01-missing_natural-o0')
        row = dict(case_id=case['id'], gold=case['gold'], score=dict(schema_valid=True, success=False,
                   action=dict(action='call', name='get_run_config', arguments={'run_id': 'a1'})))
        counts = guard_counts([row], {case['id']: case})
        self.assertEqual(counts['raw_premature_calls'], 1)
        self.assertEqual(counts['allowed_premature_calls'], 0)
        self.assertEqual(counts['correct_clarifications'], 0)
        self.assertEqual(counts['blocked_calls'], 1)


if __name__ == '__main__':
    unittest.main()
