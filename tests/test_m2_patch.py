from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.patch_evaluator import (
    IDENTITY_MAX_ABS_SCORE_DELTA_TOLERANCE,
    _audit_external_manifests,
    mediated_fraction_summary,
)
from myrec.mechanism.patch_scorer import (
    ReadoutActivationPatch,
    _cross_request_mapping,
    score_candidate_with_patch,
    score_candidates_with_patch,
    write_m2_patch_scores,
)
from myrec.utils.hashing import sha256_text


class PatchTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._special = {
            "yes": 3,
            "no": 4,
            "Yes": 5,
            "No": 6,
            "<|im_end|>": 7,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if text in self._special:
            return [self._special[text]]
        return [self._id(value) for value in text]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        return {
            "input_ids": [self._id(value) for value in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def _id(self, value: str) -> int:
        if value not in self._ids:
            self._ids[value] = 20 + len(self._ids)
        return self._ids[value]


class AddBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + self.value,)


class TinyQwen(torch.nn.Module):
    def __init__(self, vocab: int = 512, hidden: int = 4) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([AddBlock(0.01) for _ in range(28)])
        self.readout = torch.nn.Linear(hidden, vocab, bias=False)
        torch.manual_seed(7)
        torch.nn.init.uniform_(self.embed_tokens.weight, -0.2, 0.2)
        torch.nn.init.uniform_(self.readout.weight, -0.2, 0.2)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        logits = self.readout(hidden)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


def _record():
    return sanitize_record_for_model(
        {
            "request_id": "r1",
            "query": "电脑",
            "history": [
                {
                    "item_id": "h1",
                    "title": "旧电脑",
                    "brand": "甲",
                    "cat": ["数码", "电脑"],
                    "event": "click",
                    "query": "办公",
                    "ts": 1,
                }
            ],
            "candidates": [
                {"item_id": "a", "title": "短", "brand": "甲", "cat": ["电脑"]},
                {
                    "item_id": "b",
                    "title": "更长一些的商品",
                    "brand": "乙",
                    "cat": ["游戏本"],
                },
                {"item_id": "c", "title": "中等商品", "brand": "丙", "cat": ["电脑"]},
            ],
        }
    )


def test_patch_manifest_audit_hashes_raw_query_before_prompt_strip(
    tmp_path: Path,
) -> None:
    raw = {
        "request_id": "r-whitespace",
        "query": "matex6 ",
        "history": [],
        "candidates": [{"item_id": "a"}, {"item_id": "b"}],
    }
    record = sanitize_record_for_model(raw)
    assert record.query == "matex6"
    item_ids = ["a", "b"]
    candidate_path = tmp_path / "candidate_manifest.json"
    request_path = tmp_path / "request_manifest.json"
    candidate_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "dev",
                        "request_id": raw["request_id"],
                        "candidate_item_ids": item_ids,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    request_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "dev",
                        "request_id": raw["request_id"],
                        "candidate_item_ids_sha256": sha256_text(
                            json.dumps(item_ids, separators=(",", ":"))
                        ),
                        "query_sha256": sha256_text(str(raw["query"])),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _audit_external_manifests(candidate_path, request_path, [record], [raw])


@pytest.mark.parametrize(
    "method_id",
    ("q2_recranker_generalqwen", "q3_tallrec_generalqwen"),
)
def test_patch_batched_scores_equal_single_candidate_scores(method_id: str) -> None:
    model = TinyQwen().eval()
    tokenizer = PatchTokenizer()
    record = _record()
    config = {
        "method_id": method_id,
        "training": {"history_budget": 6, "max_length": 2048},
    }
    donors = [
        np.asarray([0.2 + index, -0.1, 0.3, 0.4], dtype=np.float32)
        for index in range(len(record.candidates))
    ]
    with torch.inference_mode(), ReadoutActivationPatch(model, 13) as patcher:
        batched = score_candidates_with_patch(
            model,
            tokenizer,
            patcher,
            record,
            record.candidates,
            [],
            donors,
            config,
            device="cpu",
        )
        singles = [
            score_candidate_with_patch(
                model,
                tokenizer,
                patcher,
                record,
                candidate,
                [],
                donor,
                config,
                device="cpu",
            )
            for candidate, donor in zip(record.candidates, donors)
        ]
    assert batched == pytest.approx(singles, rel=1.0e-5, abs=1.0e-5)


def test_cross_request_mapping_is_deterministic_and_has_no_identity() -> None:
    records = []
    for index in range(5):
        raw = {
            **{
                "request_id": f"r{index}",
                "query": "q",
                "history": [],
                "candidates": [{"item_id": "a"}, {"item_id": "b"}],
            }
        }
        records.append(sanitize_record_for_model(raw))
    first = _cross_request_mapping(records)
    second = _cross_request_mapping(list(reversed(records)))
    assert first == second
    assert all(key != value for key, value in first.items())
    assert set(first) == set(first.values())


def test_mediated_fraction_is_hand_computed_ratio_of_means() -> None:
    summary = mediated_fraction_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([2.0, 4.0]),
        np.asarray(["a", "b"]),
        samples=100,
        seed=3,
    )
    assert summary["mean_patch_minus_null_margin"] == pytest.approx(1.5)
    assert summary["mean_full_minus_null_margin"] == pytest.approx(3.0)
    assert summary["mediated_fraction"] == pytest.approx(0.5)
    assert summary["ci95"] == pytest.approx([0.5, 0.5])


def test_mediated_fraction_reports_undefined_zero_denominator() -> None:
    summary = mediated_fraction_summary(
        np.asarray([1.0, -1.0]),
        np.asarray([1.0, -1.0]),
        np.asarray(["same", "same"]),
        samples=10,
    )
    assert summary["mediated_fraction"] is None
    assert summary["ci95"] == [None, None]
    assert summary["bootstrap_valid_samples"] == 0


def test_patch_scorer_api_has_no_qrels_argument() -> None:
    import inspect

    assert "qrels" not in inspect.signature(write_m2_patch_scores).parameters


def test_identity_patch_numerical_tolerance_is_strict() -> None:
    assert IDENTITY_MAX_ABS_SCORE_DELTA_TOLERANCE == 1.0e-5
