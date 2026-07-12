"""Operator and degeneration contracts for CHHT."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.chht import CHHTRanker, masked_zscore, multi_positive_listwise_loss
from train.train_screen import corruption_loss, preservation_loss


class CHHTTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = CHHTRanker(
            input_dim=12,
            hidden_dim=16,
            heads=4,
            ffn_dim=32,
            rank=4,
            history_layers=1,
            pair_layers=1,
            dropout=0.0,
            max_history=4,
            max_skew_norm=0.25,
            max_score_residual=1.0,
            variant="chht",
        ).eval()

    def inputs(self, *, history: bool = True) -> dict[str, torch.Tensor]:
        batch, candidates, events, dimension = 2, 3, 3, 12
        history_mask = torch.tensor(
            [[history, history, False], [history, False, False]], dtype=torch.bool
        )
        return {
            "query_embeddings": torch.randn(batch, dimension),
            "candidate_embeddings": torch.randn(batch, candidates, dimension),
            "history_embeddings": torch.randn(batch, events, dimension),
            "base_scores": torch.randn(batch, candidates, dtype=torch.float64),
            "candidate_mask": torch.tensor(
                [[True, True, True], [True, True, False]], dtype=torch.bool
            ),
            "history_mask": history_mask,
            "history_event_weight": history_mask.float(),
            "repeat_mask": torch.zeros(
                batch, candidates, events, dtype=torch.bool
            ),
        }

    def test_no_history_is_exact_base_for_every_variant(self) -> None:
        values = self.inputs(history=False)
        for variant in sorted(self.model.VALID_VARIANTS):
            output = self.model(**values, variant=variant)
            valid = values["candidate_mask"]
            self.assertTrue(torch.equal(output.scores[valid], values["base_scores"][valid]))
            self.assertEqual(float(output.core.abs().max()), 0.0)
            self.assertEqual(float(output.residual.abs().max()), 0.0)

    def test_skew_cayley_radius_and_orthogonality(self) -> None:
        output = self.model(**self.inputs())
        skew_error = (output.core + output.core.transpose(-1, -2)).abs().max()
        diagonal = output.core.diagonal(dim1=-2, dim2=-1).abs().max()
        self.assertLess(float(skew_error), 1e-6)
        self.assertLess(float(diagonal), 1e-7)
        self.assertLessEqual(float(output.core_norm.max()), 0.250001)
        self.assertLess(float(output.cayley_orthogonality_error.max()), 1e-5)

    def test_update_matrix_rank_is_bounded(self) -> None:
        output = self.model(**self.inputs())
        basis = torch.linalg.qr(self.model.rotation_basis.float(), mode="reduced").Q
        identity = torch.eye(self.model.rank)
        cayley = torch.linalg.solve(identity + output.core[0, 0], identity - output.core[0, 0])
        update = basis @ (cayley - identity) @ basis.T
        self.assertLessEqual(int(torch.linalg.matrix_rank(update)), self.model.rank)

    def test_candidate_conditioning_changes_core(self) -> None:
        values = self.inputs()
        output = self.model(**values)
        self.assertGreater(float((output.core[0, 0] - output.core[0, 1]).abs().max()), 1e-7)

    def test_history_only_core_excludes_query_candidate_and_repeat(self) -> None:
        values = self.inputs()
        values["repeat_mask"][0, 0, 0] = True
        first = self.model(**values, variant="history_only")
        altered = {key: value.clone() for key, value in values.items()}
        altered["query_embeddings"] = torch.randn_like(values["query_embeddings"]) * 100
        altered["candidate_embeddings"] = torch.randn_like(values["candidate_embeddings"]) * 100
        altered["repeat_mask"] = ~values["repeat_mask"]
        second = self.model(**altered, variant="history_only")
        self.assertTrue(torch.allclose(first.core, second.core, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(first.core[:, :1], first.core[:, 1:2], atol=1e-6))

    def test_gradients_are_finite(self) -> None:
        self.model.train()
        output = self.model(**self.inputs())
        loss = output.scores[self.inputs()["candidate_mask"]].sum()
        # Use the output's own valid shape; a fresh mask has the same frozen shape.
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_masked_zscore_and_listwise_loss(self) -> None:
        values = torch.tensor([[1.0, 2.0, 99.0]])
        mask = torch.tensor([[True, True, False]])
        z = masked_zscore(values, mask)
        self.assertTrue(torch.allclose(z, torch.tensor([[-1.0, 1.0, 0.0]]), atol=1e-5))
        scores = torch.tensor([[2.0, 1.0, -8.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        loss = multi_positive_listwise_loss(scores, labels, mask)
        expected = torch.log(torch.exp(torch.tensor(2.0)) + torch.exp(torch.tensor(1.0))) - 2.0
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_all_no_history_full_training_loss_is_finite(self) -> None:
        self.model.train()
        values = self.inputs(history=False)
        output = self.model(**values, variant="chht")
        labels = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=output.scores.dtype
        )
        teacher = values["base_scores"].to(output.scores.dtype)
        listwise = multi_positive_listwise_loss(
            output.scores, labels, values["candidate_mask"]
        )
        preservation = preservation_loss(
            output.scores,
            teacher,
            values["candidate_mask"],
            values["repeat_mask"],
            temperature=1.0,
            margin=0.05,
        )
        corruption = corruption_loss(self.model, values, output.core_norm, margin=0.02)
        core_norm = output.core_norm.sum() * 0.0
        loss = listwise + preservation + 0.05 * corruption + 0.001 * core_norm

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(corruption.detach()), 0.0)
        self.assertTrue(corruption.requires_grad)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
