import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.supervised_diagnostics import (
    PackedRequestData,
    SupervisedDiagnosticRanker,
    _materialize_split,
    build_permuted_history_data,
    multi_positive_listwise_loss,
)


class SupervisedDiagnosticsTest(unittest.TestCase):
    def test_wrong_history_replaces_embedding_and_b0b_channels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records_dev.jsonl"
            assignments = root / "assignments.jsonl"
            donor_bank = root / "donor_bank.jsonl"
            rows = [
                {
                    "request_id": "r1",
                    "candidates": [
                        {"item_id": "10", "cat": ["a", "a1", "a2"]},
                        {"item_id": "20", "cat": ["b", "b1", "b2"]},
                    ],
                    "history": [
                        {
                            "item_id": "10",
                            "cat": ["a", "a1", "a2"],
                            "event": "click",
                        }
                    ],
                },
                {
                    "request_id": "r2",
                    "candidates": [
                        {"item_id": "10", "cat": ["a", "a1", "a2"]},
                        {"item_id": "20", "cat": ["b", "b1", "b2"]},
                    ],
                    "history": [
                        {
                            "item_id": "20",
                            "cat": ["b", "b1", "b2"],
                            "event": "purchase",
                        }
                    ],
                },
            ]
            with records.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            assignment_rows = [
                {
                    "request_id": "r1",
                    "donor_request_id": "r2",
                    "target_user_id": "u1",
                    "donor_user_id": "u2",
                    "match_tier": "length",
                    "target_history_length": 1,
                },
                {
                    "request_id": "r2",
                    "donor_request_id": "r1",
                    "target_user_id": "u2",
                    "donor_user_id": "u1",
                    "match_tier": "length",
                    "target_history_length": 1,
                },
            ]
            with assignments.open("w", encoding="utf-8") as handle:
                for row in assignment_rows:
                    handle.write(json.dumps(row) + "\n")
            with donor_bank.open("w", encoding="utf-8") as handle:
                for row, embedding_index, weight, user_id in [
                    (rows[0], 10, 1.0, "u1"),
                    (rows[1], 20, 1.5, "u2"),
                ]:
                    handle.write(
                        json.dumps(
                            {
                                "history": row["history"],
                                "history_embedding_indices": [embedding_index],
                                "history_event_weights": [weight],
                                "request_id": row["request_id"],
                                "user_id": user_id,
                            }
                        )
                        + "\n"
                    )
            data = PackedRequestData(
                root=root,
                split="dev",
                request_ids=["r1", "r2"],
                query_indices=np.asarray([0, 1], dtype=np.int32),
                timestamps=np.asarray([1, 2], dtype=np.int64),
                candidate_offsets=np.asarray([0, 2, 4], dtype=np.int64),
                candidate_embedding_indices=np.asarray(
                    [10, 20, 10, 20], dtype=np.int32
                ),
                candidate_item_ids=np.asarray([10, 20, 10, 20], dtype=np.int64),
                candidate_labels=np.zeros(4, dtype=np.uint8),
                candidate_purchase_labels=np.zeros(4, dtype=np.uint8),
                candidate_b0b_scores=np.asarray(
                    [3, 0, 0, 4.5], dtype=np.float32
                ),
                history_offsets=np.asarray([0, 1, 2], dtype=np.int64),
                history_embedding_indices=np.asarray([10, 20], dtype=np.int32),
                history_event_weights=np.asarray([1.0, 1.5], dtype=np.float16),
            )
            permuted, evidence = build_permuted_history_data(
                data, records, assignments, donor_bank
            )
            np.testing.assert_array_equal(
                permuted.history_embedding_indices, np.asarray([20, 10])
            )
            np.testing.assert_allclose(
                permuted.candidate_b0b_scores,
                np.asarray([0.0, 6.0, 4.0, 0.0], dtype=np.float32),
            )
            self.assertEqual(evidence["donor_requests"], 2)

    def test_multi_positive_listwise_loss(self):
        scores = torch.tensor([[2.0, 1.0, 0.0]])
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True]])
        expected = torch.logsumexp(scores[0], dim=0) - 2.0
        self.assertAlmostEqual(
            float(multi_positive_listwise_loss(scores, labels, mask)),
            float(expected),
            places=6,
        )

    def test_residual_is_exactly_zero_without_history(self):
        query_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        item_embeddings = torch.eye(4)
        popularity = torch.tensor([1.0, 0.0, 0.5, 0.0])
        base = SupervisedDiagnosticRanker(
            query_embeddings,
            item_embeddings,
            popularity,
            projection_dim=2,
            dropout=0.0,
            variant="d1q",
        )
        residual = SupervisedDiagnosticRanker(
            query_embeddings,
            item_embeddings,
            popularity,
            projection_dim=2,
            dropout=0.0,
            variant="d1a",
        )
        residual.load_state_dict(base.state_dict())
        batch = {
            "query_indices": torch.tensor([0]),
            "candidate_indices": torch.tensor([[0, 1, 2]]),
            "candidate_mask": torch.tensor([[True, True, True]]),
            "candidate_b0b": torch.zeros((1, 3)),
            "history_indices": torch.zeros((1, 1), dtype=torch.long),
            "history_event_weights": torch.zeros((1, 1)),
            "history_mask": torch.tensor([[False]]),
        }
        base.eval()
        residual.eval()
        self.assertTrue(torch.equal(base(batch), residual(batch)))

    def test_empty_history_backward_is_finite(self):
        query_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        item_embeddings = torch.eye(4)
        model = SupervisedDiagnosticRanker(
            query_embeddings,
            item_embeddings,
            torch.zeros(4),
            projection_dim=2,
            dropout=0.0,
            variant="d1a",
        )
        batch = {
            "query_indices": torch.tensor([0]),
            "candidate_indices": torch.tensor([[0, 1, 2]]),
            "candidate_mask": torch.tensor([[True, True, True]]),
            "candidate_b0b": torch.zeros((1, 3)),
            "history_indices": torch.zeros((1, 1), dtype=torch.long),
            "history_event_weights": torch.zeros((1, 1)),
            "history_mask": torch.tensor([[False]]),
        }
        loss = model(batch)[0, :3].sum()
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_materializer_preserves_candidates_and_history(self):
        record = {
            "request_id": "r1",
            "user_id": "u1",
            "ts": 10,
            "query": "query",
            "history": [
                {
                    "item_id": "3",
                    "cat": ["c"],
                    "event": "purchase",
                    "ts": 5,
                }
            ],
            "candidates": [
                {"item_id": "1", "cat": ["c"], "clicked": 1, "purchased": 0},
                {"item_id": "2", "cat": ["d"], "clicked": 0, "purchased": 0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_train.jsonl"
            records.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "train"
            manifest = _materialize_split(
                records_path=records,
                output_dir=output,
                split="train",
                query_index={"r1": 0},
                item_index={"1": 0, "2": 1, "3": 2},
                history_limit=50,
            )
            self.assertEqual(manifest["requests"], 1)
            self.assertEqual(manifest["candidate_rows"], 2)
            self.assertEqual(manifest["history_rows"], 1)
            np.testing.assert_array_equal(
                np.load(output / "candidate_item_ids.npy"), np.asarray([1, 2])
            )
            np.testing.assert_array_equal(
                np.load(output / "candidate_labels.npy"), np.asarray([1, 0])
            )
            self.assertAlmostEqual(
                float(np.load(output / "history_event_weights.npy")[0]), 1.5
            )


if __name__ == "__main__":
    unittest.main()
