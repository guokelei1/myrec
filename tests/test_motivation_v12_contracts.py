from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.motivation_v12_contracts import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    build_instructrec_selection_sections,
    build_prompt_sections,
    complete_candidate_chunks,
    encode_instructrec_selection_prompt,
    encode_prompt_sections,
    listwise_target_distribution,
    load_training_groups,
    pairwise_index_pairs,
    sanitize_record_for_model,
)


class _CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in value]


class MotivationV12ContractsTest(unittest.TestCase):
    def test_forbidden_labels_cannot_change_any_model_prompt(self):
        left = _record()
        right = json.loads(json.dumps(left))
        right.update(clicked=["a"], purchased=["b"], relevance={"b": 99}, label=7)
        right["history"][0].update(clicked=1, purchased=1, relevance=9, target="x")
        right["candidates"][0].update(clicked=1, purchased=1, relevance=9, label=1)

        clean_left = sanitize_record_for_model(left)
        clean_right = sanitize_record_for_model(right)
        self.assertEqual(clean_left, clean_right)
        for method_id in (
            "q0_qwen3_reranker_06b",
            "q2_recranker_generalqwen",
            "q3_tallrec_generalqwen",
        ):
            self.assertEqual(
                build_prompt_sections(
                    method_id, clean_left, clean_left.candidates[0], history_budget=6
                ),
                build_prompt_sections(
                    method_id, clean_right, clean_right.candidates[0], history_budget=6
                ),
            )
        self.assertEqual(
            build_instructrec_selection_sections(
                clean_left, clean_left.candidates, history_budget=6, template_index=0
            ),
            build_instructrec_selection_sections(
                clean_right, clean_right.candidates, history_budget=6, template_index=0
            ),
        )
        projected = [*clean_left.history, *clean_left.candidates]
        self.assertTrue(
            all(not (set(row) & FORBIDDEN_MODEL_INPUT_FIELDS) for row in projected)
        )

    def test_training_labels_are_loaded_separately_and_sampling_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            records.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            qrels.write_text(
                json.dumps(
                    {
                        "request_id": "r1",
                        "clicked": ["b"],
                        "purchased": ["b"],
                        "relevance": {"b": 2},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            first, stats = load_training_groups(
                records, qrels, seed=17, negatives_per_positive=2, max_group_size=3
            )
            second, _ = load_training_groups(
                records, qrels, seed=17, negatives_per_positive=2, max_group_size=3
            )
            self.assertEqual(first, second)
            self.assertEqual(stats["groups"], 1)
            self.assertEqual(sorted(first[0].gains), [0.0, 2.0])
            self.assertNotIn("relevance", first[0].candidates[0])

    def test_pairwise_and_listwise_conversions_have_hand_computed_targets(self):
        gains = [2.0, 1.0, 1.0, 0.0]
        self.assertEqual(
            pairwise_index_pairs(gains),
            [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
        )
        target = listwise_target_distribution(gains)
        self.assertEqual(target, [3.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 0.0])
        self.assertAlmostEqual(sum(target), 1.0)

    def test_candidate_chunking_preserves_exact_identity_and_order(self):
        candidates = [{"item_id": str(index)} for index in range(7)]
        chunks = complete_candidate_chunks(candidates, 3)
        self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 1])
        self.assertEqual(
            [row["item_id"] for chunk in chunks for row in chunk],
            [str(index) for index in range(7)],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            complete_candidate_chunks([{"item_id": "x"}, {"item_id": "x"}], 1)

    def test_prompt_boundary_is_deterministic_and_keeps_answer_suffix(self):
        record = sanitize_record_for_model(_record())
        sections = build_instructrec_selection_sections(
            record, record.candidates, history_budget=6, template_index=1
        )
        tokenizer = _CharacterTokenizer()
        first = encode_prompt_sections(tokenizer, sections, max_length=256)
        second = encode_prompt_sections(tokenizer, sections, max_length=256)
        suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 256)
        self.assertEqual(first[-len(suffix) :], suffix)

    def test_q1_candidate_exposure_is_identical_for_full_and_null_history(self):
        record = sanitize_record_for_model(_record())
        tokenizer = _CharacterTokenizer()
        full, full_targets, full_audit = encode_instructrec_selection_prompt(
            tokenizer,
            record,
            record.candidates,
            history_budget=6,
            template_index=0,
            max_length=1024,
            context_token_budget=256,
            max_target_length=128,
        )
        null, null_targets, null_audit = encode_instructrec_selection_prompt(
            tokenizer,
            record,
            record.candidates,
            history=[],
            history_budget=6,
            template_index=0,
            max_length=1024,
            context_token_budget=256,
            max_target_length=128,
        )
        self.assertNotEqual(full, null)
        self.assertEqual(full_targets, null_targets)
        self.assertEqual(full_audit["candidate_content_tokens"], null_audit["candidate_content_tokens"])
        self.assertTrue(full_audit["all_candidate_markers_preserved"])
        self.assertEqual(full_audit["candidate_response_collisions"], 0)
        self.assertTrue(full_audit["candidate_targets_include_answer_end"])


def _record():
    return {
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
        "ts": 10,
        "query": "red running shoes",
        "history": [
            {
                "item_id": "h",
                "title": "old shoe",
                "brand": "old",
                "cat": ["sport", "shoe"],
                "event": "click",
                "query": "shoes",
                "ts": 1,
            }
        ],
        "candidates": [
            {"item_id": "a", "title": "red sandal", "brand": "A", "cat": ["shoe"]},
            {"item_id": "b", "title": "red runner", "brand": "B", "cat": ["shoe"]},
        ],
        "masks": {"strict_nonrepeat": True},
    }


if __name__ == "__main__":
    unittest.main()
