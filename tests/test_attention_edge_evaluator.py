from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from myrec.mechanism.attention_edge_evaluator import (
    _audit_ineligible_frozen_conditions,
    _common_implementation_digest,
    _load_bundle_content_control_eligibility,
    _load_bundle_frozen_baseline,
    _ndcg_values,
)
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.utils.hashing import sha256_file


def test_attention_evaluator_uses_shared_nested_score_contract():
    request_ids = ["r"]
    candidates = {"r": ["a", "b", "c"]}
    gains = {"r": {"a": 2.0, "b": 1.0, "c": 0.0}}
    scores = {"r": {"a": 3.0, "b": 2.5, "c": -1.0}}
    margins = _target_margins(request_ids, candidates, gains, scores)
    np.testing.assert_allclose(margins, [0.5])
    ndcg = _ndcg_values(request_ids, candidates, gains, scores)
    np.testing.assert_allclose(ndcg, [1.0])


def test_attention_edge_evaluator_requires_one_implementation_digest():
    bundles = {
        "q2": {
            13: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
            20: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
        },
        "q3": {
            13: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
        },
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["q3"][13].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["q3"][13].metadata["run_contract"]["implementation_digest"] = "fixed"
    bundles["q3"][13].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)


def test_ineligible_conditions_must_equal_bound_frozen_baseline():
    conditions = ("baseline_full", "active_control")
    frozen = {("r", "a"): 0.1, ("r", "b"): -0.25}
    rows = [
        {
            "request_id": "r",
            "candidate_item_id": item_id,
            "conditions": {condition: score for condition in conditions},
        }
        for item_id, score in (("a", 0.1), ("b", -0.25))
    ]
    _audit_ineligible_frozen_conditions(
        "r", rows, conditions, frozen, label="test bundle"
    )
    rows[1]["conditions"]["active_control"] = -0.125
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        _audit_ineligible_frozen_conditions(
            "r", rows, conditions, frozen, label="test bundle"
        )


def test_bundle_frozen_baseline_identity_is_reloaded_and_byte_bound(tmp_path):
    baseline_root = tmp_path / "frozen"
    baseline_root.mkdir()
    metadata_path = baseline_root / "metadata.json"
    scores_path = baseline_root / "scores.jsonl"
    metadata_path.write_text(
        json.dumps({"method_id": "method", "checkpoint_id": "checkpoint"}),
        encoding="utf-8",
    )
    scores_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": "r",
                    "candidate_item_id": item_id,
                    "score": score,
                }
            )
            for item_id, score in (("a", 0.1), ("b", -0.25))
        )
        + "\n",
        encoding="utf-8",
    )
    identity = {
        "root": str(baseline_root),
        "metadata_sha256": sha256_file(metadata_path),
        "scores_sha256": sha256_file(scores_path),
        "score_rows": 2,
    }
    records = [
        SimpleNamespace(
            request_id="r",
            candidates=({"item_id": "a"}, {"item_id": "b"}),
        )
    ]
    bundle_metadata = {
        "method_id": "method",
        "checkpoint_id": "checkpoint",
        "frozen_baseline": identity,
    }
    assert _load_bundle_frozen_baseline(
        bundle_metadata, records, label="test bundle"
    ) == {("r", "a"): 0.1, ("r", "b"): -0.25}

    bundle_metadata["frozen_baseline"] = {
        **identity,
        "scores_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="byte identity drift"):
        _load_bundle_frozen_baseline(
            bundle_metadata, records, label="test bundle"
        )


def test_content_control_eligibility_is_request_bound_not_count_only(monkeypatch):
    identity = {
        "manifest_path": "controls/manifest.json",
        "manifest_sha256": "a" * 64,
        "rows_path": "controls/rows.jsonl",
        "rows_sha256": "b" * 64,
        "eligible_requests": 1,
        "ineligible_requests": 1,
    }
    controls = {
        "r1": {"eligible": True},
        "r2": {"eligible": False},
    }
    monkeypatch.setattr(
        "myrec.mechanism.attention_edge_evaluator._load_manifest",
        lambda _path: {"_sha256": "frozen-manifest"},
    )
    monkeypatch.setattr(
        "myrec.mechanism.attention_edge_evaluator._load_content_controls",
        lambda _manifest, _method, _records: (controls, identity),
    )
    metadata = {
        "method_id": "method",
        "deep_dive_manifest_sha256": "frozen-manifest",
        "content_control": identity,
    }
    records = [SimpleNamespace(request_id="r1"), SimpleNamespace(request_id="r2")]
    assert _load_bundle_content_control_eligibility(
        metadata, records, label="test bundle"
    ) == {"r1": True, "r2": False}

    metadata["content_control"] = {**identity, "eligible_requests": 2}
    with pytest.raises(ValueError, match="byte identity drift"):
        _load_bundle_content_control_eligibility(
            metadata, records, label="test bundle"
        )
