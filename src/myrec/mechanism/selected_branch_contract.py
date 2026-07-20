"""Freeze the minimal qrels-derived contract for D2 selected-branch scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from myrec.mechanism.attention_edge_runtime import _write_json
from myrec.utils.hashing import sha256_file


def materialize_selected_branch_contract(
    selection_path: str | Path,
    confirmation_path: str | Path,
    output_path: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Expose only the immutable selected block and its evidence role."""

    selection_path = Path(selection_path)
    confirmation_path = Path(confirmation_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    selection = _read_json(selection_path)
    confirmation = _read_json(confirmation_path)
    if (
        selection.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_fold0_selection"
        or selection.get("status") != "completed"
        or selection.get("selection_frozen_before_fold1") is not True
        or selection.get("fold") != 0
    ):
        raise ValueError("invalid fold-0 selection for selected-branch contract")
    if (
        confirmation.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_fold1_confirmation"
        or confirmation.get("status") != "completed"
        or confirmation.get("selected_block_immutable") is not True
        or confirmation.get("fold") != 1
        or confirmation.get("method_id") != selection.get("method_id")
        or confirmation.get("checkpoint_id") != selection.get("checkpoint_id")
        or confirmation.get("selection")
        != {"path": str(selection_path), "sha256": sha256_file(selection_path)}
    ):
        raise ValueError("fold-1 confirmation does not bind the selection")
    implementation_digest = str(selection.get("implementation_digest") or "")
    if not implementation_digest or confirmation.get(
        "implementation_digest"
    ) != implementation_digest:
        raise ValueError("fold selection/confirmation implementation binding differs")
    selected = selection.get("selected_block")
    fixed = confirmation.get("fixed_transition_confirmation", {})
    if selected is None:
        eligible = False
        block = None
        role = "stopped_no_negative_fold0_transition"
        reproduced = False
    else:
        block = int(selected)
        if not 14 <= block <= 27:
            raise ValueError("selected branch block is outside registered range")
        if fixed.get("applicable") is not True or int(
            fixed.get("selected_block", -1)
        ) != block:
            raise ValueError("fold-1 confirmation selected block drift")
        eligible = True
        reproduced = fixed.get("confirmed_negative_transition") is True
        role = (
            "registered_confirmatory_branch_localization"
            if reproduced
            else "exploratory_unresolved_transition_branch_localization"
        )
    contract = {
        "schema_version": 1,
        "contract_type": "transformer_deep_dive_d2_selected_branch_contract",
        "status": "completed",
        "method_id": selection["method_id"],
        "checkpoint_id": selection["checkpoint_id"],
        "selected_block": block,
        "branch_scoring_eligible": eligible,
        "fold1_negative_transition_reproduced": reproduced,
        "evidence_role": role,
        "scoring_population": "normalized_query_fold_1",
        "selected_nodes": [
            "block_input_residual",
            "input_rmsnorm_output",
            "attention_o_projection",
            "post_attention_residual",
            "post_attention_rmsnorm_output",
            "mlp_down_projection",
            "block_output_residual",
        ],
        "postblock_implementation_digest": implementation_digest,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "confirmation": {
            "path": str(confirmation_path),
            "sha256": sha256_file(confirmation_path),
        },
        "qrels_values_exposed_to_scorer": False,
        "source_test_opened": False,
        "command": list(command or []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, contract)
    return contract


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
