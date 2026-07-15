from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.llm_srec_adapter import (
    HISTORY_EMBED_TOKEN,
    ITEM_OUTPUT_TOKEN,
    USER_OUTPUT_TOKEN,
    LLMSRecRetrievalHead,
    build_item_prompt,
    build_user_prompt,
    uniformity_loss,
)
from myrec.baselines.representative_sequence_adapter import (
    TrainVocabulary,
    build_sequence_request,
)


def _record(history: bool = True) -> dict:
    return {
        "request_id": "r",
        "query": "red shoes",
        "ts": 30,
        "history": (
            [
                {"item_id": "h1", "title": "coat", "event": "click", "ts": 10},
                {"item_id": "h2", "title": "dress", "event": "click", "ts": 20},
            ]
            if history
            else []
        ),
        "candidates": [
            {"item_id": "c1", "title": "red running shoes"},
            {"item_id": "c2", "title": "blue hat"},
        ],
    }


class LLMSRecAdapterTest(unittest.TestCase):
    def test_prompts_have_exact_paper_mechanism_tokens_and_pps_query(self):
        train = _record()
        vocabulary = TrainVocabulary.fit([train])
        request = build_sequence_request(train, vocabulary, history_budget=8)
        user_prompt = build_user_prompt(request)
        self.assertIn("Current query: red shoes", user_prompt)
        self.assertEqual(user_prompt.count(HISTORY_EMBED_TOKEN), 2)
        self.assertEqual(user_prompt.count(USER_OUTPUT_TOKEN), 1)
        item_prompt = build_item_prompt(request.candidates[0])
        self.assertEqual(item_prompt.count(HISTORY_EMBED_TOKEN), 1)
        self.assertEqual(item_prompt.count(ITEM_OUTPUT_TOKEN), 1)

    def test_null_prompt_has_no_fabricated_collaborative_embedding(self):
        train = _record()
        vocabulary = TrainVocabulary.fit([train])
        null = build_sequence_request(_record(history=False), vocabulary, history_budget=8)
        prompt = build_user_prompt(null)
        self.assertIn("No prior interactions", prompt)
        self.assertNotIn(HISTORY_EMBED_TOKEN, prompt)
        self.assertIn(USER_OUTPUT_TOKEN, prompt)

    def test_uniformity_matches_hand_computed_orthogonal_pair(self):
        value = uniformity_loss(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(value.item(), torch.exp(torch.tensor(-4.0)).item())

    def test_retrieval_distillation_uniformity_loss_has_finite_gradients(self):
        torch.manual_seed(7)
        head = LLMSRecRetrievalHead(
            llm_dim=4, cf_dim=3, projection_dim=2, hidden_dim=5
        )
        llm_user = torch.randn(2, 4)
        llm_items = torch.randn(2, 3, 4)
        cf_user = torch.randn(2, 3, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        scores, losses = head.losses(
            llm_user=llm_user,
            llm_items=llm_items,
            cf_user=cf_user,
            positive_indices=torch.tensor([1, 2]),
            candidate_mask=mask,
        )
        losses.total.backward()
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            )
        )
        self.assertIsNone(cf_user.grad)

    def test_positive_candidate_cannot_be_padding(self):
        head = LLMSRecRetrievalHead(
            llm_dim=2, cf_dim=2, projection_dim=2, hidden_dim=2
        )
        with self.assertRaisesRegex(ValueError, "padded candidate"):
            head.losses(
                llm_user=torch.ones(1, 2),
                llm_items=torch.ones(1, 2, 2),
                cf_user=torch.ones(1, 2),
                positive_indices=torch.tensor([1]),
                candidate_mask=torch.tensor([[True, False]]),
            )


if __name__ == "__main__":
    unittest.main()
