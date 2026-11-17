import importlib
import unittest


class StagedTests(unittest.TestCase):
    def test_second_stage_receives_raw_model_fields_and_no_labels(self):
        try:c=importlib.import_module('experiments.fields_first_staged.core')
        except ModuleNotFoundError:self.fail('Staged protocol not implemented')
        case=dict(request='Show a run configuration',context='Known runs: r1,r2',tools=[],gold={'secret':'gold'},supplied_arguments={'secret':'oracle'})
        raw='{"tool":"guessed_tool","fields":{"run_id":"hallucinated"}}'
        messages=c.messages(case,2,raw)
        self.assertEqual(messages[-2],dict(role='assistant',content=raw))
        self.assertNotIn('oracle',str(messages));self.assertNotIn('"secret"',str(messages))
        with self.assertRaises(ValueError):c.messages(case,2,None)
        self.assertNotIn(raw,str(c.messages(case,1)))


if __name__=='__main__':unittest.main()
