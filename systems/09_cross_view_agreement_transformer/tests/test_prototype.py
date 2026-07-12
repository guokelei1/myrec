from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))

from prototype import (  # noqa: E402
    AgreementContrastAttention,
    CrossViewAgreementTransformer,
    positive_margin_meet,
)


class AgreementOperatorTests(unittest.TestCase):
    def _identity_operator(self, d_model: int = 1) -> AgreementContrastAttention:
        operator = AgreementContrastAttention(d_model=d_model, eps=0.0).double()
        with torch.no_grad():
            operator.value_projection.weight.zero_()
            operator.output_projection.weight.zero_()
            operator.value_projection.weight.copy_(torch.eye(d_model, dtype=torch.float64))
            operator.output_projection.weight[0, 0] = 1.0
        return operator

    def test_parallel_sum_is_a_strict_conjunction(self) -> None:
        left = torch.tensor([2.0, 2.0, -2.0, -2.0])
        right = torch.tensor([1.0, -1.0, 1.0, -1.0])
        actual = positive_margin_meet(left, right, eps=0.0)
        torch.testing.assert_close(actual, torch.tensor([2.0 / 3.0, 0.0, 0.0, 0.0]))

    def test_hand_computed_three_candidate_update(self) -> None:
        # Agreed strengths: k01=2/3, k02=3/2, k12=2/3.
        # With hidden=(3,1,0), the two nonzero corrections are
        # (4/3+9/2)/(1+2/3+3/2)=35/19 and (2/3)/(1+2/3)=2/5.
        operator = self._identity_operator()
        base = torch.zeros(1, 3, dtype=torch.float64)
        hidden = torch.tensor([[[3.0], [1.0], [0.0]]], dtype=torch.float64)
        view_q = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
        view_c = torch.tensor([[1.0, 0.0, -2.0]], dtype=torch.float64)
        scores, diagnostics = operator(base, hidden, view_q, view_c, torch.tensor([True]))
        expected = torch.tensor([[35.0 / 19.0, 2.0 / 5.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(scores, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(int(torch.count_nonzero(diagnostics.agreement_strength)), 3)

    def test_disagreement_cannot_modify_scores(self) -> None:
        operator = self._identity_operator()
        base = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
        hidden = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
        # The average of these views prefers candidate 0, but their margins
        # disagree; ensemble averaging and C09 therefore have different output.
        view_q = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
        view_c = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        scores, diagnostics = operator(base, hidden, view_q, view_c, torch.tensor([True]))
        self.assertTrue(torch.equal(scores, base))
        self.assertEqual(int(torch.count_nonzero(diagnostics.agreement_strength)), 0)
        self.assertFalse(torch.equal(0.5 * (view_q + view_c), base))

    def test_no_history_is_bit_exact_base(self) -> None:
        operator = self._identity_operator()
        base = torch.tensor([[0.125, -0.25]], dtype=torch.float64)
        hidden = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
        view = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        scores, diagnostics = operator(base, hidden, view, view, torch.tensor([False]))
        self.assertTrue(torch.equal(scores, base))
        self.assertEqual(int(torch.count_nonzero(diagnostics.correction)), 0)

    def test_candidate_permutation_equivariance(self) -> None:
        torch.manual_seed(9)
        operator = AgreementContrastAttention(d_model=3).double()
        base = torch.randn(2, 4, dtype=torch.float64)
        hidden = torch.randn(2, 4, 3, dtype=torch.float64)
        view_q = base + torch.randn(2, 4, dtype=torch.float64)
        view_c = base + torch.randn(2, 4, dtype=torch.float64)
        available = torch.tensor([True, True])
        original, _ = operator(base, hidden, view_q, view_c, available)

        permutation = torch.tensor([2, 0, 3, 1])
        permuted, _ = operator(
            base[:, permutation],
            hidden[:, permutation],
            view_q[:, permutation],
            view_c[:, permutation],
            available,
        )
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(permuted[:, inverse], original, rtol=1e-12, atol=1e-12)

    def test_common_mode_view_shifts_cancel(self) -> None:
        operator = self._identity_operator()
        base = torch.zeros(1, 3, dtype=torch.float64)
        hidden = torch.tensor([[[3.0], [1.0], [0.0]]], dtype=torch.float64)
        view_q = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
        view_c = torch.tensor([[1.0, 0.0, -2.0]], dtype=torch.float64)
        original, _ = operator(base, hidden, view_q, view_c, torch.tensor([True]))
        shifted, _ = operator(
            base,
            hidden,
            view_q + 7.0,
            view_c - 4.0,
            torch.tensor([True]),
        )
        torch.testing.assert_close(shifted, original, rtol=0.0, atol=1e-12)

    def test_gradients_reach_both_agreeing_views_and_values(self) -> None:
        operator = self._identity_operator()
        base = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        hidden = torch.tensor(
            [[[3.0], [1.0], [0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        view_q = torch.tensor(
            [[2.0, 0.0, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        view_c = torch.tensor(
            [[1.0, 0.0, -2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        scores, _ = operator(base, hidden, view_q, view_c, torch.tensor([True]))
        weights = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
        (scores * weights).sum().backward()
        for tensor in (base, hidden, view_q, view_c):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_off_diagonal_witness_refutes_candidate_local_gate(self) -> None:
        operator = self._identity_operator()
        base = torch.zeros(1, 2, dtype=torch.float64)
        # Candidate 0's own hidden state is exactly zero.  A diagonal update
        # g_0 * z_0 must be zero, while agreed attention reads candidate 1 and
        # yields -2/3 at candidate 0.
        hidden = torch.tensor([[[0.0], [2.0]]], dtype=torch.float64)
        view = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        scores, _ = operator(base, hidden, view, view, torch.tensor([True]))
        torch.testing.assert_close(scores[0, 0], torch.tensor(-2.0 / 3.0, dtype=torch.float64))
        self.assertEqual(float(hidden[0, 0, 0]), 0.0)

    def test_single_candidate_degenerates_to_base(self) -> None:
        operator = self._identity_operator()
        base = torch.tensor([[0.7]], dtype=torch.float64)
        hidden = torch.tensor([[[2.0]]], dtype=torch.float64)
        view = torch.tensor([[9.0]], dtype=torch.float64)
        scores, _ = operator(base, hidden, view, view, torch.tensor([True]))
        self.assertTrue(torch.equal(scores, base))


class EndToEndTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260711)
        self.model = CrossViewAgreementTransformer(
            vocab_size=64,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            max_segment_tokens=3,
            max_history=4,
        )
        self.model.eval()
        self.query = torch.tensor([[1, 2, 0]])
        self.history = torch.tensor([[[3, 4, 0], [5, 6, 0]]])
        self.history_mask = torch.tensor([[True, True]])
        self.candidates = torch.tensor(
            [[[7, 8, 0], [9, 10, 0], [11, 12, 0]]]
        )

    def test_information_barriers_are_structural(self) -> None:
        original = self.model.encode_restricted_mediators(
            self.query,
            self.history,
            self.candidates,
            self.history_mask,
        )
        changed_candidates = self.candidates.clone()
        changed_candidates[:, :, :2] = torch.tensor([[[20, 21], [22, 23], [24, 25]]])
        candidate_changed = self.model.encode_restricted_mediators(
            self.query,
            self.history,
            changed_candidates,
            self.history_mask,
        )
        self.assertTrue(
            torch.equal(
                original["query_first_mediator"],
                candidate_changed["query_first_mediator"],
            )
        )

        changed_query = torch.tensor([[30, 31, 0]])
        query_changed = self.model.encode_restricted_mediators(
            changed_query,
            self.history,
            self.candidates,
            self.history_mask,
        )
        self.assertTrue(
            torch.equal(
                original["candidate_first_mediator"],
                query_changed["candidate_first_mediator"],
            )
        )

    def test_end_to_end_no_history_fallback(self) -> None:
        no_history_mask = torch.tensor([[False, False]])
        output = self.model(
            self.query,
            self.history,
            self.candidates,
            no_history_mask,
        )
        self.assertTrue(torch.equal(output["scores"], output["base_scores"]))
        self.assertFalse(bool(output["evidence_available"].item()))

    def test_query_mask_fallback(self) -> None:
        output = self.model(
            torch.zeros_like(self.query),
            self.history,
            self.candidates,
            self.history_mask,
            query_present=torch.tensor([False]),
        )
        self.assertTrue(torch.equal(output["scores"], output["base_scores"]))

    def test_full_model_candidate_permutation(self) -> None:
        original = self.model(
            self.query,
            self.history,
            self.candidates,
            self.history_mask,
        )["scores"]
        permutation = torch.tensor([2, 0, 1])
        permuted = self.model(
            self.query,
            self.history,
            self.candidates[:, permutation],
            self.history_mask,
        )["scores"]
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(permuted[:, inverse], original, rtol=1e-5, atol=1e-6)

    def test_end_to_end_backward_reaches_shared_transformer(self) -> None:
        self.model.train()
        output = self.model(
            self.query,
            self.history,
            self.candidates,
            self.history_mask,
        )
        # Auxiliary per-view train losses are required because exact blocking
        # intentionally gives no agreement gradient on a disagreeing pair.
        loss = (
            output["scores"].square().sum()
            + 0.1 * output["query_first_scores"].square().sum()
            + 0.1 * output["candidate_first_scores"].square().sum()
        )
        loss.backward()
        embedding_grad = self.model.segment_encoder.token_embedding.weight.grad
        mediator_grad = self.model.mediator_attention.in_proj_weight.grad
        rank_grad = self.model.rank_encoder.self_attn.in_proj_weight.grad
        for gradient in (embedding_grad, mediator_grad, rank_grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
