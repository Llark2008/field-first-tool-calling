import unittest
import os

from toolcall_lab.training_spike import validate_tokens, training_budget
from toolcall_lab.training_loss import completion_mask
from toolcall_lab.resource_monitor import parse_swap, parse_vm_stat, resource_violation


class TrainingSpikeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TOOLCALL_TEST_MLX') == '1', 'Opt in on a host with accessible Metal')
    def test_actual_loss_counts_only_assistant_suffix(self):
        import math
        import mlx.core as mx
        from toolcall_lab.training_loss import assistant_loss
        batch = mx.array([[10, 20, 30, 40, 50, 0, 0, 0]])
        lengths = mx.array([[3, 5]])
        def uniform_logits(inputs):
            return mx.zeros((inputs.shape[0], inputs.shape[1], 64))
        loss, ntokens = assistant_loss(uniform_logits, batch, lengths)
        self.assertEqual(ntokens.item(), 2)
        self.assertAlmostEqual(loss.item(), math.log(64), places=5)

    def test_completion_mask_excludes_first_padding_target(self):
        import numpy as np
        # Full sequence [10,20,30,40,50], prefix length 3, padded to 8.
        # Shifted targets are [20,30,40,50,0,0,0]; only 40 and 50 are answers.
        positions = np.arange(1, 8)
        mask = completion_mask(positions, np.array([[3]]), np.array([[5]]))
        self.assertEqual(mask.tolist(), [[False, False, True, True, False, False, False]])

    def test_prefix_mask_excludes_prompt_and_keeps_answer(self):
        tokens, offset = validate_tokens([10, 20, 30, 40, 50], [10, 20, 30], 8)
        self.assertEqual(offset, 3)
        self.assertEqual(tokens[offset:], [40, 50])

    def test_rejects_misalignment_empty_answer_and_truncation(self):
        for full, prefix, cap in (([1, 2, 3], [1, 4], 8), ([1, 2], [1, 2], 8), ([1, 2, 3], [1], 2)):
            with self.subTest(full=full, prefix=prefix), self.assertRaises(ValueError):
                validate_tokens(full, prefix, cap)

    def test_training_budget_counts_microsteps_not_updates(self):
        result = training_budget(64, 1, 8)
        self.assertEqual(result, {'microsteps': 64, 'sample_exposures': 64, 'optimizer_updates': 8})
        with self.assertRaises(ValueError):
            training_budget(63, 1, 8)

    def test_memory_units_and_thresholds(self):
        self.assertEqual(parse_swap('total = 1024.00M used = 512.00M free = 512.00M'), 512 * 1024**2)
        self.assertEqual(parse_swap('total = 2.0G used = 1.5G free = 0.5G'), int(1.5 * 1024**3))
        stats = parse_vm_stat('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages occupied by compressor: 3.\n')
        self.assertEqual(stats['free_bytes'], 163840)
        self.assertEqual(stats['compressor_bytes'], 49152)
        self.assertIsNone(resource_violation({'rss_bytes': 1024, 'swap_used_bytes': 100}, 100))
        self.assertIn('RSS', resource_violation({'rss_bytes': 11 * 1024**3, 'swap_used_bytes': 100}, 100))
        self.assertIn('swap', resource_violation({'rss_bytes': 1024, 'swap_used_bytes': 3 * 1024**3}, 0))


if __name__ == '__main__':
    unittest.main()
