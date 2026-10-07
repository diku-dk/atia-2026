"""``block_postfix`` (``src/eomt/training/lightning_module.py``): with the instance module's single final-layer
metric, the training losses still get one key per block (upstream's deep supervision), the metric none."""

import unittest
from types import SimpleNamespace

from src.eomt.training.lightning_module import LightningModule


class BlockPostfixTest(unittest.TestCase):
    def test_per_block_loss_keys_with_one_metric(self):
        module = SimpleNamespace(network=SimpleNamespace(masked_attn_enabled=True, num_blocks=3), metrics=[None])
        postfixes = [LightningModule.block_postfix(module, i) for i in range(4)]
        self.assertEqual(postfixes, ["_block_-3", "_block_-2", "_block_-1", ""])

    def test_no_postfix_without_masked_attention(self):
        module = SimpleNamespace(network=SimpleNamespace(masked_attn_enabled=False, num_blocks=3), metrics=[None])
        self.assertEqual(LightningModule.block_postfix(module, 0), "")


if __name__ == "__main__":
    unittest.main()
