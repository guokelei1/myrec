from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.decoder_reranker_training import train_decoder_cross_encoder


class DecoderRerankerTrainingTest(unittest.TestCase):
    def test_rejects_nonpositive_training_budget_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            train_decoder_cross_encoder(
                "/not/read",
                "unused",
                "/not/write",
                input_mode="qc",
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
