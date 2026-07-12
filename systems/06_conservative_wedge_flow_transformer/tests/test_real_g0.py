from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from train.materialize_real_g0 import (
    key_align_scores,
    verify_label_source_after_selection,
)
from train.real_data import sha256_file, write_json


def test_key_alignment_round_trip_is_bitwise_and_rank_exact() -> None:
    scores = np.asarray([0.25, -1.0, 2.0], dtype=np.float32)
    aligned, audit = key_align_scores(
        request_ids=["r1", "r2"],
        candidate_item_ids=[np.asarray([11, 12]), np.asarray([21])],
        canonical_scores=scores,
        offsets=np.asarray([0, 2, 3], dtype=np.int64),
        seed=20260708,
    )
    assert np.array_equal(aligned, scores)
    assert audit["bitwise_array_equal"] is True
    assert audit["rank_mismatches"] == 0


def test_key_alignment_rejects_duplicate_request_item_key() -> None:
    with pytest.raises(ValueError, match="duplicate alignment key"):
        key_align_scores(
            request_ids=["r1"],
            candidate_item_ids=[np.asarray([11, 11])],
            canonical_scores=np.asarray([0.0, 1.0], dtype=np.float32),
            offsets=np.asarray([0, 2], dtype=np.int64),
            seed=3,
        )


def test_label_source_is_verified_only_against_durable_selection(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    write_json(selection, {"roles": {"fit": {"indices": [1]}}})
    selection_hash = sha256_file(selection)
    labels = tmp_path / "candidate_labels.npy"
    np.save(labels, np.asarray([0.0, 1.0], dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "files": {
                "train/candidate_labels.npy": {
                    "path": str(labels),
                    "sha256": sha256_file(labels),
                }
            }
        },
    )
    config = {
        "paths": {
            "packed_manifest": str(manifest),
            "train_candidate_labels": str(labels),
        }
    }
    result = verify_label_source_after_selection(
        config, selection_path=selection, selection_hash=selection_hash
    )
    assert result["sha256"] == sha256_file(labels)
    assert result["verified_after_selection_sha256"] == selection_hash
    with pytest.raises(ValueError, match="selection changed"):
        verify_label_source_after_selection(
            config, selection_path=selection, selection_hash="0" * 64
        )
