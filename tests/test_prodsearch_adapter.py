import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.prodsearch_adapter import (
    convert_prodsearch_ranklist,
    deterministic_pad_candidates,
    materialize_prodsearch_format,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _item(item_id, clicked=0):
    return {
        "item_id": str(item_id),
        "title": f"item {item_id}",
        "brand": "brand",
        "cat": ["cat"],
        "clicked": clicked,
        "purchased": 0,
    }


class ProdSearchAdapterTest(unittest.TestCase):
    def test_official_batched_dot_keeps_singleton_batch_dimension(self):
        baseline_root = Path(__file__).resolve().parents[1] / "baselines" / "pps_prodsearch"
        sys.path.insert(0, str(baseline_root))
        try:
            from models.item_transformer import _batched_vector_dot

            scores = _batched_vector_dot(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
            self.assertEqual(tuple(scores.shape), (1,))
            self.assertEqual(scores.item(), 11.0)
        finally:
            sys.path.remove(str(baseline_root))

    def test_official_item_batch_keeps_empty_history_indices_integral(self):
        baseline_root = Path(__file__).resolve().parents[1] / "baselines" / "pps_prodsearch"
        sys.path.insert(0, str(baseline_root))
        try:
            from data.batch_data import ItemPVBatch

            batch = ItemPVBatch(
                query_word_idxs=[[1], [2]],
                target_prod_idxs=[3, 4],
                u_item_idxs=[[], []],
                candi_prod_idxs=[[3, 5], [4, 6]],
            )
            self.assertEqual(batch.u_item_idxs.dtype, torch.long)
            self.assertEqual(tuple(batch.u_item_idxs.shape), (2, 0))
        finally:
            sys.path.remove(str(baseline_root))

    def test_padding_is_deterministic_disjoint_and_prefix_preserving(self):
        first = deterministic_pad_candidates([2, 4], 5, 8, key="r1", seed=7)
        second = deterministic_pad_candidates([2, 4], 5, 8, key="r1", seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[:2], [2, 4])
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse(set(first[2:]) & {2, 4})

    def test_materializer_isolates_sibling_positives_and_restores_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            train = [
                {
                    "request_id": "train1",
                    "query": "query",
                    "ts": 10,
                    "history": [_item("h")],
                    "candidates": [_item("a", 1), _item("b", 1), _item("c"), _item("d"), _item("e")],
                }
            ]
            dev = [
                {
                    "request_id": "dev1",
                    "query": "dev query",
                    "ts": 20,
                    "history": [_item("h")],
                    "candidates": [_item("a"), _item("c"), _item("f")],
                }
            ]
            _write_jsonl(standardized / "records_train.jsonl", train)
            _write_jsonl(standardized / "records_dev.jsonl", dev)
            manifest = {
                "entries": [
                    {"split": "train", "request_id": "train1", "candidate_item_ids": ["a", "b", "c", "d", "e"]},
                    {"split": "dev", "request_id": "dev1", "candidate_item_ids": ["a", "c", "f"]},
                ]
            }
            (standardized / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "out"
            result = materialize_prodsearch_format(
                standardized,
                output,
                seed=7,
                valid_examples=1,
                valid_candidate_size=5,
                test_candidate_size=5,
            )
            self.assertEqual(result["multi_positive_guard"]["sibling_positives_added_by_adapter"], 0)
            self.assertTrue(result["multi_positive_guard"]["synthetic_histories_equal_input"])

            with gzip.open(output / "data" / "product.txt.gz", "rt", encoding="utf-8") as handle:
                products = [line.strip() for line in handle]
            h_review = products.index("h")
            with gzip.open(output / "data" / "u_r_seq.txt.gz", "rt", encoding="utf-8") as handle:
                sequences = [[int(value) for value in line.split()] for line in handle]
            train_sequences = sequences[1:3]
            self.assertEqual([sequence[:-1] for sequence in train_sequences], [[h_review], [h_review]])

            request_map = next(iter(_read_jsonl(output / "dev_request_map.jsonl")))
            ranklist = output / "official.ranklist"
            padded_ids = []
            with (output / "split" / "test.bias_product.ranklist").open() as handle:
                for rank, line in enumerate(handle, start=1):
                    fields = line.split()
                    padded_ids.append(fields[2])
                    fields[4] = str(float(rank))
                    with ranklist.open("a", encoding="utf-8") as out:
                        out.write(" ".join(fields) + "\n")
            self.assertEqual(padded_ids[:3], ["a", "c", "f"])
            scores = output / "scores.jsonl"
            converted = convert_prodsearch_ranklist(
                ranklist,
                output / "dev_request_map.jsonl",
                scores,
                method_id="b9_test",
                candidate_manifest_path=standardized / "candidate_manifest.json",
            )
            self.assertTrue(converted["candidate_sets_exact"])
            score_rows = list(_read_jsonl(scores))
            self.assertEqual([row["candidate_item_id"] for row in score_rows], ["a", "c", "f"])
            self.assertEqual(request_map["candidate_item_ids"], ["a", "c", "f"])


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


if __name__ == "__main__":
    unittest.main()
