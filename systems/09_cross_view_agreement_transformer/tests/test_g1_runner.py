from __future__ import annotations

import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import g1_runner as g1  # noqa: E402


class G1ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)

    def test_generator_is_exactly_deterministic_and_stream_separated(self) -> None:
        first = g1.generate_split(20260711, 16)
        second = g1.generate_split(20260711, 16)
        validation = g1.generate_split(20260711 + 100000, 16)
        for name in ("query", "history", "candidates", "utility", "target"):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))
        self.assertFalse(torch.equal(first.query, validation.query))
        self.assertTrue(torch.isfinite(first.utility).all())

    def test_dual_causal_utility_matches_hand_equation(self) -> None:
        split = g1.generate_split(9, 1)
        alternating = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        history_mean = split.history.mean(dim=1)
        base = (
            split.query[:, None]
            * alternating
            * split.candidates
        ).sum(dim=-1) / (8.0 ** 0.5)
        q_factor = torch.tanh(
            (split.query * history_mean).sum(dim=-1) / (8.0 ** 0.5)
        )
        c_factor = torch.tanh(
            (split.candidates * history_mean[:, None]).sum(dim=-1) / (8.0 ** 0.5)
        )
        expected = 0.35 * base + 4.0 * q_factor[:, None] * c_factor
        torch.testing.assert_close(split.utility, expected, rtol=0.0, atol=0.0)

    def test_sattolo_is_deterministic_permutation_without_fixed_points(self) -> None:
        first = g1.sattolo_permutation(512, 20260711 + 300000)
        second = g1.sattolo_permutation(512, 20260711 + 300000)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch.sort(first).values, torch.arange(512)))
        self.assertEqual(int((first == torch.arange(512)).sum()), 0)

    def test_all_modes_have_identical_allocated_parameters(self) -> None:
        counts = []
        for mode in g1.METHODS:
            torch.manual_seed(1)
            model = g1.G1SharedTransformer(mode)
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_shared_initialization_is_identical_across_modes(self) -> None:
        states = []
        for mode in ("cma", "ordinary_attention", "diagonal", "base_matched"):
            torch.manual_seed(20260711 + 1000003)
            states.append(g1.G1SharedTransformer(mode).state_dict())
        reference = states[0]
        for state in states[1:]:
            self.assertEqual(reference.keys(), state.keys())
            for key in reference:
                self.assertTrue(torch.equal(reference[key], state[key]), key)

    def test_cma_factor_flips_and_disagreement_are_exact(self) -> None:
        base = torch.zeros(1, 3)
        q_scores = torch.tensor([[2.0, 0.0, -1.0]])
        c_scores = torch.tensor([[1.0, 0.0, -2.0]])
        hidden = torch.tensor(
            [[[1.0] * 8, [0.0] * 8, [-1.0] * 8]]
        )
        torch.manual_seed(2)
        model = g1.G1SharedTransformer("cma")
        available = torch.tensor([True])
        clean, clean_delta = model.fuse(
            base, q_scores, c_scores, hidden, available, "clean"
        )
        self.assertGreater(float(clean_delta.detach().abs().sum()), 0.0)
        for corruption in ("query_factor_flip", "candidate_factor_flip"):
            _, delta = model.fuse(
                base, q_scores, c_scores, hidden, available, corruption
            )
            self.assertEqual(int(torch.count_nonzero(delta)), 0)
        disagreement, delta = model.fuse(
            base,
            q_scores,
            c_scores,
            hidden,
            available,
            "all_pair_disagreement",
        )
        self.assertTrue(torch.equal(disagreement, base))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)

    def test_fallbacks_are_bit_exact(self) -> None:
        torch.manual_seed(3)
        model = g1.G1SharedTransformer("cma").eval()
        split = g1.generate_split(4, 4)
        false_mask = torch.zeros(4, dtype=torch.bool)
        no_history = model(
            split.query,
            split.history,
            split.candidates,
            history_present=false_mask,
        )
        query_masked = model(
            torch.zeros_like(split.query),
            split.history,
            split.candidates,
            query_present=false_mask,
        )
        self.assertTrue(torch.equal(no_history["scores"], no_history["base"]))
        self.assertTrue(torch.equal(query_masked["scores"], query_masked["base"]))

    def test_pairwise_metric_tie_rules(self) -> None:
        utility = torch.tensor([[3.0, 2.0, 1.0]])
        perfect = torch.tensor([[9.0, 5.0, -1.0]])
        all_tied = torch.zeros_like(perfect)
        reversed_scores = -perfect
        self.assertEqual(g1.pairwise_accuracy(perfect, utility), 1.0)
        self.assertEqual(g1.pairwise_accuracy(all_tied, utility), 0.5)
        self.assertEqual(g1.pairwise_accuracy(reversed_scores, utility), 0.0)

    def test_batch_schedule_is_exact_and_reproducible(self) -> None:
        first = g1.make_batch_schedule(20260711)
        second = g1.make_batch_schedule(20260711)
        self.assertEqual(len(first), 200)
        self.assertTrue(all(len(batch) == 64 for batch in first))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        first_epoch = torch.cat(first[:32])
        self.assertTrue(
            torch.equal(torch.sort(first_epoch).values, torch.arange(2048))
        )

    def test_execution_lock_and_transitive_prelock_verify(self) -> None:
        combined = g1.verify_execution_lock()
        self.assertEqual(len(combined), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in combined))

    def test_matched_modes_activate_all_frozen_parameter_groups(self) -> None:
        split = g1.generate_split(5, 8)
        for mode in g1.MATCHED_METHODS:
            torch.manual_seed(5 + 1000003)
            model = g1.G1SharedTransformer(mode)
            output = model(split.query, split.history, split.candidates)
            loss = g1.training_loss(output, split.target)
            loss.backward()
            groups = g1.gradient_groups(model)
            for name, value in groups.items():
                self.assertTrue(math_is_positive_finite(value), (mode, name, value))

    def test_terminal_decision_uses_frozen_thresholds(self) -> None:
        def seed_record(seed: int) -> dict[str, object]:
            methods: dict[str, object] = {}
            for mode in g1.METHODS:
                accuracy = 0.70
                if mode == "cma":
                    accuracy = 0.80
                elif mode == "ordinary_attention":
                    accuracy = 0.77
                elif mode == "constant_cma":
                    accuracy = 0.78
                methods[mode] = {
                    "training": {
                        "total_trainable_parameters": 100,
                        "first_backward_group_grad_norms": {"all": 1.0},
                        "all_losses_finite": True,
                    },
                    "evaluation": {
                        "clean": {"pairwise_accuracy": accuracy},
                    },
                }
            methods["cma"]["evaluation"]["corruptions"] = {
                "query_factor_flip": {"pairwise_accuracy": 0.71},
                "candidate_factor_flip": {"pairwise_accuracy": 0.71},
                "shuffled_history": {"pairwise_accuracy": 0.71},
                "all_pair_disagreement": {"bit_exact_base": True},
                "no_history": {"score_mismatch_count": 0},
                "query_masked": {"score_mismatch_count": 0},
            }
            return {"seed": seed, "methods": methods}

        passing = [seed_record(seed) for seed in g1.SEEDS]
        decision = g1.decide(passing)
        self.assertEqual(decision["terminal_decision"], "PASS_G1_REQUEST_D0_REVIEW")
        self.assertTrue(all(decision["criteria"].values()))

        failing = deepcopy(passing)
        for row in failing:
            row["methods"]["ordinary_attention"]["evaluation"]["clean"][
                "pairwise_accuracy"
            ] = 0.795
        failed_decision = g1.decide(failing)
        self.assertFalse(
            failed_decision["criteria"]["c2_cma_vs_ordinary_attention"]
        )
        self.assertEqual(failed_decision["terminal_decision"], "STOP_C09_G1_FAILED")


def math_is_positive_finite(value: float) -> bool:
    return value > 0.0 and value < float("inf")


if __name__ == "__main__":
    unittest.main()
