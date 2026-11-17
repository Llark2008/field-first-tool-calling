import unittest
from experiments.fields_epochs import report

class ReportTests(unittest.TestCase):
    def test_teacher_diagnostic_rejects_changed_tokens_or_nonfinite_loss(self):
        target={'tokens':[1,2,3],'offset':1,'groups':{'known':[],'null':[],'action':[1]}}
        good={'case_id':'x',**target,'losses':[.1,.2],'correct':[True,False]}
        report.check_teacher([{'id':'x'}],[good],{'x':target})
        for bad in [dict(good,tokens=[1,2,4]),dict(good,losses=[float('nan'),.2]),dict(good,correct=[True])]:
            with self.assertRaises(ValueError):report.check_teacher([{'id':'x'}],[bad],{'x':target})

    def test_summary_keeps_model_and_converter_independent(self):
        from pathlib import Path
        import json
        root=Path.home()/'.local/share/local-tool-calling/runs/fields-sft-20261115'
        cases=[json.loads(s) for s in (root/'dev-cases.jsonl').read_text().splitlines()]
        raw=[json.loads(s) for s in (root/'dev-F-600.jsonl').read_text().splitlines()]
        cell=report.cell_scores(cases,raw)
        self.assertEqual((cell['model']['C'],cell['model']['Q'],cell['model']['F']),(44,34,13))
        self.assertEqual((cell['D']['C'],cell['D']['Q'],cell['D']['F']),(54,37,3))
        self.assertEqual(cell['fields']['exact'],95)

if __name__=='__main__':unittest.main()
