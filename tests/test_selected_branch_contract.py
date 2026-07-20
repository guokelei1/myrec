from myrec.mechanism.selected_branch_contract import (
    materialize_selected_branch_contract,
)


def test_selected_branch_contract_exposes_no_effect_values(tmp_path):
    import hashlib
    import json

    selection_path = tmp_path / "selection.json"
    selection = {
        "analysis_type": "transformer_deep_dive_d2_postblock_fold0_selection",
        "status": "completed",
        "selection_frozen_before_fold1": True,
        "fold": 0,
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "implementation_digest": "postblock-runtime",
        "selected_block": 20,
        "effects": {"secret": 1.0},
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    confirmation_path = tmp_path / "confirmation.json"
    confirmation = {
        "analysis_type": "transformer_deep_dive_d2_postblock_fold1_confirmation",
        "status": "completed",
        "selected_block_immutable": True,
        "fold": 1,
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "implementation_digest": "postblock-runtime",
        "selection": {"path": str(selection_path), "sha256": selection_sha},
        "fixed_transition_confirmation": {
            "applicable": True,
            "selected_block": 20,
            "confirmed_negative_transition": False,
            "fold1_step": 123.0,
        },
    }
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    output = tmp_path / "contract.json"
    contract = materialize_selected_branch_contract(
        selection_path, confirmation_path, output
    )
    assert contract["selected_block"] == 20
    assert contract["postblock_implementation_digest"] == "postblock-runtime"
    assert contract["evidence_role"].startswith("exploratory")
    serialized = json.dumps(contract)
    assert "123.0" not in serialized
    assert "secret" not in serialized
