from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.cect import CECTModel, counterfactual_upper_quantile  # noqa: E402
from train.data import assert_candidate_manifest  # noqa: E402
from train.integrity import (  # noqa: E402
    assert_source_isolation,
    load_config,
    verify_proposal_lock,
)
from train.smoke import run_smoke, tiny_batch  # noqa: E402


class CECTTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = tiny_batch()
        self.model = CECTModel(dropout=0.0)
        self.model.eval()

    def test_quantile_uses_finite_sample_index(self) -> None:
        value = counterfactual_upper_quantile(
            torch.tensor([4.0, 1.0, 3.0, 2.0]), alpha=0.25
        )
        self.assertEqual(float(value), 4.0)

    def test_exact_atom_is_protected_above_certificate(self) -> None:
        self.model.set_certificate_threshold(1_000_000.0)
        output = self.model(self.batch)
        self.assertGreater(float(output.exact_scores[0, 0]), 0.0)
        self.assertEqual(float(output.transfer_scores[0, 0]), 0.0)
        self.assertTrue(bool(output.request_evidence_present[0]))

    def test_empty_history_is_exact_base_fallback(self) -> None:
        output = self.model(self.batch)
        self.assertTrue(torch.equal(output.scores[1], self.batch["base_scores"][1]))
        self.assertFalse(bool(output.request_evidence_present[1]))

    def test_masked_candidate_and_history_padding(self) -> None:
        batch = copy.deepcopy(self.batch)
        batch["candidate_mask"][0, 2] = False
        output = self.model(batch)
        self.assertEqual(float(output.scores[0, 2]), 0.0)
        self.assertTrue(torch.equal(output.energies[1], torch.zeros_like(output.energies[1])))

    def test_counterfactual_twins_change_only_declared_evidence(self) -> None:
        true = self.model._condition_batch(self.batch, "true")
        wrong = self.model._condition_batch(self.batch, "wrong")
        shuffled = self.model._condition_batch(self.batch, "shuffled")
        coarse = self.model._condition_batch(self.batch, "coarse")
        self.assertTrue(torch.equal(true["query"], wrong["query"]))
        self.assertTrue(torch.equal(wrong["history"], self.batch["wrong_history"]))
        self.assertTrue(
            torch.equal(shuffled["history_indices"][0, :3], torch.tensor([32, 11, 31]))
        )
        self.assertEqual(int(torch.count_nonzero(coarse["history"])), 0)
        self.assertTrue(torch.equal(coarse["history_categories"], true["history_categories"]))

    def test_parameter_matched_plain_control_and_finite_scores(self) -> None:
        plain = CECTModel(dropout=0.0, mode="plain")
        plain.load_state_dict(self.model.state_dict())
        self.assertEqual(self.model.parameter_count(), plain.parameter_count())
        output = plain(self.batch)
        self.assertTrue(bool(torch.isfinite(output.scores).all()))

    def test_candidate_manifest_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate manifest hash mismatch"):
                assert_candidate_manifest(path, "0" * 64)

    def test_lock_and_forbidden_source_scan(self) -> None:
        config = load_config()
        lock = verify_proposal_lock(config)
        source = assert_source_isolation()
        self.assertEqual(lock["candidate_id"], "c01")
        self.assertEqual(source["forbidden_references"], 0)

    def test_cpu_smoke_has_attention_and_head_gradients(self) -> None:
        config = copy.deepcopy(load_config())
        config["model"].update(
            {
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "dim_feedforward": 64,
                "dropout": 0.0,
            }
        )
        result = run_smoke(config, "cpu")
        self.assertTrue(result["finite_loss"])
        self.assertTrue(result["optimizer_step_completed"])


if __name__ == "__main__":
    unittest.main()
