from __future__ import annotations

import json

from myrec.mechanism.objective_conflict_synthesis import (
    _state_invariant_selection_sha256,
)


def test_state_invariant_selection_hash_excludes_only_state(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    changed = tmp_path / "changed"
    for path, state, request_id in (
        (left, "base_initialization", "r1"),
        (right, "frozen_final_checkpoint", "r1"),
        (changed, "frozen_final_checkpoint", "r2"),
    ):
        path.mkdir()
        (path / "selection_manifest.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "surfaces": {"strict_transfer": {"request_ids": [request_id]}},
                    "selection_seed": 20260715,
                }
            )
        )
    assert _state_invariant_selection_sha256(left) == _state_invariant_selection_sha256(right)
    assert _state_invariant_selection_sha256(left) != _state_invariant_selection_sha256(changed)
