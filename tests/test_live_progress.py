import json
import tempfile
import unittest
from pathlib import Path
from toolcall_lab.live_progress import update_state, read_state


class ProgressTests(unittest.TestCase):
    def test_updates_preserve_previous_metrics_and_detect_stale_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_state(root, stage='training_A', status='running', step=80, total_steps=2400)
            update_state(root, metrics={'loss': 0.5})
            state = read_state(root)
            self.assertEqual(state['step'], 80)
            self.assertEqual(state['metrics']['loss'], 0.5)
            self.assertFalse(state['stale'])
            state['updated_at'] = 1
            (root/'progress.json').write_text(json.dumps(state))
            self.assertTrue(read_state(root)['stale'])


if __name__ == '__main__': unittest.main()
