import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from toolcall_lab.report_mechanism import render_html,write_report


class MechanismReportTests(unittest.TestCase):
    def payload(self):
        return dict(status='incomplete',fingerprint='test',reason='stopped',candidates=[],references={},effects={},verdicts={},
            cases=[dict(request='</script><script>bad()</script>',models={})],training={},resource_summaries=[],token_audit={},consumed={},
            certification={'status':'pending'},limitations=['development only'],missing_cells=['M1-200'],partial_evaluations=[],
            verification=dict(status='completed_rows_verified',counts={'reused':0,'new':0},total_generations=0,new_generation_seconds=0))

    def test_embedded_payload_and_export_copies_are_identical_and_escape_html(self):
        p=self.payload()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';source=root/'source';run.mkdir();source.mkdir()
            write_report(p,run,source)
            html=(run/'results.html').read_text();embedded=re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1)
            self.assertEqual(json.loads(embedded),p);self.assertNotIn('</script><script>bad()',html)
            self.assertEqual((run/'report-data.json').read_bytes(),(source/'reports/training-mechanism-metrics.json').read_bytes())
            self.assertEqual((run/'results.html').read_bytes(),(source/'reports/training-mechanism-results.html').read_bytes())
            self.assertIn('incomplete',(source/'reports/training-mechanism-results.md').read_text())
            self.assertIn('未完成',html)

    def test_result_and_progress_javascript_parse(self):
        from toolcall_lab.mechanism_progress import HTML as progress
        node=shutil.which('node');self.assertIsNotNone(node)
        scripts=[]
        for html in [render_html(self.payload()),progress]:
            scripts.extend(re.findall(r'<script>(.*?)</script>',html,re.S))
        self.assertEqual(len(scripts),2)
        for script in scripts:
            result=subprocess.run([node,'--check'],input=script,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
