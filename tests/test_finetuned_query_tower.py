import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.finetuned_query_tower import (
    FineTunedQueryTower,
    _zscore,
    compose_d2p_history_scores,
    iter_query_batches,
)
from myrec.analysis.supervised_diagnostics import (
    PackedRequestData,
    multi_positive_listwise_loss,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FineTunedQueryTowerTest(unittest.TestCase):
    def test_identity_adapter_and_finite_backward(self):
        model = FineTunedQueryTower(
            TinyEncoder(),
            torch.eye(4),
            logit_scale_initial=20.0,
            logit_scale_bounds=(1.0, 100.0),
        )
        torch.testing.assert_close(model.item_adapter.weight, torch.eye(4))
        self.assertNotIn("item_embeddings", model.state_dict())
        scores = model(
            input_ids=torch.tensor([[0, 1], [1, 0]]),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            candidate_indices=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            candidate_mask=torch.tensor([[True, True, False], [True, True, True]]),
        )
        loss = multi_positive_listwise_loss(
            scores,
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[True, True, False], [True, True, True]]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_dynamic_batches_preserve_all_requests_and_candidates(self):
        data = PackedRequestData(
            root=Path("."),
            split="train",
            request_ids=["r0", "r1", "r2"],
            query_indices=np.asarray([0, 1, 2], dtype=np.int32),
            timestamps=np.asarray([0, 1, 2], dtype=np.int64),
            candidate_offsets=np.asarray([0, 2, 7, 10], dtype=np.int64),
            candidate_embedding_indices=np.arange(10, dtype=np.int32),
            candidate_item_ids=np.arange(10, dtype=np.int64),
            candidate_labels=np.asarray([1, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
            candidate_purchase_labels=np.zeros(10, dtype=np.uint8),
            candidate_b0b_scores=np.zeros(10, dtype=np.float32),
            history_offsets=np.asarray([0, 0, 0, 0], dtype=np.int64),
            history_embedding_indices=np.asarray([], dtype=np.int32),
            history_event_weights=np.asarray([], dtype=np.float16),
        )
        batches = list(
            iter_query_batches(
                data,
                np.arange(3),
                max_requests=3,
                max_padded_candidates=8,
                seed=0,
                shuffle=False,
            )
        )
        observed_requests = sum(len(batch["request_indices"]) for batch in batches)
        observed_candidates = sum(int(batch["candidate_mask"].sum()) for batch in batches)
        self.assertEqual(observed_requests, 3)
        self.assertEqual(observed_candidates, 10)

    def test_zscore_is_finite_for_constant_values(self):
        values = _zscore(np.asarray([3.0, 3.0, 3.0]))
        np.testing.assert_array_equal(values, np.zeros(3, dtype=np.float32))

    def test_static_d2p_history_composition_falls_back_without_history(self):
        text = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
        popularity = np.asarray([2.0, 1.0, 4.0], dtype=np.float32)
        history = np.zeros(3, dtype=np.float32)
        beta = 0.4
        d2p = 0.6 * _zscore(text) + 0.4 * _zscore(popularity)
        actual = compose_d2p_history_scores(
            text, popularity, history, d2p_alpha=0.6, beta=beta
        )
        np.testing.assert_allclose(actual, beta * _zscore(d2p), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
