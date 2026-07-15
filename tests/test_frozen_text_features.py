from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.frozen_text_features import (
    FrozenTextFeatureStore,
    collect_visible_content_texts,
)
from myrec.utils.hashing import sha256_text


class FrozenTextFeaturesTest(unittest.TestCase):
    def test_collection_ignores_labels_and_is_order_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            rows = [
                {
                    "request_id": "r",
                    "query": "shoe",
                    "history": [],
                    "candidates": [
                        {"item_id": "x", "title": "red shoe", "clicked": 1}
                    ],
                }
            ]
            a.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            rows[0]["request_id"] = "s"
            rows[0]["candidates"][0]["clicked"] = 0
            b.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            forward = collect_visible_content_texts([a, b])
            reverse = collect_visible_content_texts([b, a])
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward), {"query: shoe", "title: red shoe"})

    def test_read_only_feature_store_is_hash_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "title: red shoe"
            np.save(root / "vectors.npy", np.array([[1.0, 2.0]], dtype=np.float16))
            (root / "index.json").write_text(
                json.dumps({"schema_version": 1, "hash_to_row": {sha256_text(text): 0}}),
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps({"hidden_size": 2, "qrels_read": False}),
                encoding="utf-8",
            )
            store = FrozenTextFeatureStore(root)
            value = store(text)
            self.assertEqual(store.dimension, 2)
            np.testing.assert_allclose(value, [1.0, 2.0])
            with self.assertRaises(KeyError):
                store("missing")


if __name__ == "__main__":
    unittest.main()
