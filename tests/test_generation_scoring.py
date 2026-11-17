import json
import unittest

from toolcall_lab.generation_scoring import score_generation
from toolcall_lab.tasks import pilot_cases


class GenerationScoringTests(unittest.TestCase):
    def test_token_limit_is_failure_even_if_final_json_happens_to_be_complete(self):
        case = pilot_cases()[0]
        text = json.dumps(case['gold'])
        regular = score_generation(case, text, 'stop')
        limited = score_generation(case, text, 'length')
        self.assertTrue(regular['success'])
        self.assertFalse(regular['truncated'])
        self.assertTrue(limited['schema_valid'])
        self.assertTrue(limited['truncated'])
        self.assertFalse(limited['success'])
        self.assertIn('token limit', limited['error'])


if __name__ == '__main__':
    unittest.main()
