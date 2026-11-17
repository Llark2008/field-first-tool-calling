import json
import re
import shutil
import subprocess
import tempfile
import unittest
from toolcall_lab.report_clarification_ablation import render_html


class AblationReportingTests(unittest.TestCase):
    def test_embedded_payload_matches_and_escapes_script_content(self):
        payload={'kind':'development pilot','cases':[{'request':'</script><img src=x>'}], 'candidates':[]}
        html=render_html(payload)
        raw=re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1)
        self.assertEqual(json.loads(raw),payload)
        self.assertNotIn('</script><img',html)

    @unittest.skipUnless(shutil.which('node'),'Node required for browser script syntax')
    def test_browser_script_parses(self):
        html=render_html({})
        script=re.findall(r'<script>(.*?)</script>',html,re.S)[0]
        with tempfile.NamedTemporaryFile(suffix='.js',mode='w') as f:
            f.write(script);f.flush()
            result=subprocess.run(['node','--check',f.name],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
