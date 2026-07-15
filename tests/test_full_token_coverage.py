from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.full_token_coverage import (
    audit_cross_encoder_preprocess_coverage,
    audit_full_token_coverage,
    fit_history_assignments_to_context_budget,
)
from myrec.utils.jsonl import iter_jsonl


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def num_special_tokens_to_add(self, pair=True):
        return 3 if pair else 2


class _FakeCrossEncoder:
    def _resolve_prompt(self, prompt, prompt_name):
        return "instruction "

    def preprocess(self, pairs, prompt=None):
        import torch

        lengths = [len((prompt + left + " " + right).split()) for left, right in pairs]
        width = max(lengths)
        mask = [([1] * length) + ([0] * (width - length)) for length in lengths]
        return {"attention_mask": torch.tensor(mask)}


class FullTokenCoverageTest(unittest.TestCase):
    def test_context_fit_drops_only_oldest_effective_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            records.write_text(
                json.dumps(
                    {
                        "request_id": "r1",
                        "query": "short query",
                        "history": [],
                        "candidates": [{"item_id": "a", "title": "doc"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            assignments = root / "wrong.jsonl"
            assignments.write_text(
                json.dumps(
                    {
                        "request_id": "r1",
                        "assignment": "wrong",
                        "history": [
                            {"title": "ignored older event"},
                            {"title": "one two three four"},
                            {"title": "five six seven eight"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = fit_history_assignments_to_context_budget(
                records,
                assignments,
                root / "fitted.jsonl",
                root / "report.json",
                model_name="fake",
                max_length=20,
                history_budget=2,
                tokenizer=_WhitespaceTokenizer(),
            )
            fitted = list(iter_jsonl(root / "fitted.jsonl"))[0]
            self.assertEqual(result["trimmed_requests"], 1)
            self.assertEqual(result["history_events_dropped_for_context"], 1)
            self.assertEqual(len(fitted["history"]), 1)
            self.assertEqual(fitted["history"][0]["title"], "five six seven eight")
            self.assertFalse(result["qrels_read"])

    def test_reports_overflow_without_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            row = {
                "request_id": "r1",
                "query": "red shoe",
                "history": [{"title": "very old coat", "event": "click"}],
                "candidates": [
                    {"item_id": "a", "title": "short"},
                    {"item_id": "b", "title": "many candidate words here"},
                ],
            }
            records.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = audit_full_token_coverage(
                records,
                root / "report.json",
                model_name="fake",
                max_length=10,
                history_budget=10,
                tokenizer=_WhitespaceTokenizer(),
            )
            self.assertEqual(result["requests"], 1)
            self.assertEqual(result["candidate_pairs"], 2)
            self.assertGreater(result["overflow_pairs"], 0)
            self.assertFalse(result["qrels_read"])

    def test_only_second_reports_context_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            records.write_text(json.dumps({
                "request_id": "r1", "query": "short query", "history": [],
                "candidates": [{"item_id": "a", "title": "one two three four five six"}],
            }) + "\n", encoding="utf-8")
            result = audit_full_token_coverage(
                records, root / "report.json", model_name="fake", max_length=12,
                history_budget=0, truncation_strategy="only_second",
                tokenizer=_WhitespaceTokenizer(),
            )
            self.assertGreater(result["overflow_pairs"], 0)
            self.assertTrue(result["context_preserved_under_configured_truncation"])
            self.assertGreaterEqual(
                result["candidate_capacity_if_context_preserved"]["min"], 1
            )

    def test_assignment_history_replaces_record_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            records.write_text(json.dumps({
                "request_id": "r1", "query": "q", "history": [],
                "candidates": [{"item_id": "a", "title": "doc"}],
            }) + "\n", encoding="utf-8")
            assignments = root / "wrong.jsonl"
            assignments.write_text(json.dumps({
                "request_id": "r1",
                "history": [{"title": "assigned history", "event": "purchase"}],
            }) + "\n", encoding="utf-8")
            result = audit_full_token_coverage(
                records, root / "report.json", model_name="fake",
                history_budget=1, history_assignments_path=assignments,
                tokenizer=_WhitespaceTokenizer(),
            )
            self.assertEqual(result["history_present_requests"], 1)
            self.assertEqual(result["history_assignments_path"], str(assignments))

    def test_real_preprocess_mode_includes_prompt_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            records.write_text(
                json.dumps(
                    {
                        "request_id": "r1",
                        "query": "red shoe",
                        "history": [{"title": "old coat", "event": "click"}],
                        "candidates": [{"item_id": "a", "title": "short doc"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = audit_cross_encoder_preprocess_coverage(
                records,
                root / "report.json",
                model_name="fake",
                max_length=5,
                history_budget=1,
                audit_max_length=100,
                predictor=_FakeCrossEncoder(),
            )
            self.assertEqual(result["candidate_pairs"], 1)
            self.assertGreater(result["overflow_pairs"], 0)
            self.assertEqual(result["preprocess_prompt"], "instruction ")


if __name__ == "__main__":
    unittest.main()
