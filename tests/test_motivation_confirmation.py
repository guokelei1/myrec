from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.motivation_confirmation import resolve_confirmation_decision


def _all_pass() -> dict[str, bool]:
    return {
        "population_power": True,
        "task_adequacy": True,
        "target_repeat_positive_control": True,
        "recurrence_transfer_separation": True,
        "practical_nonrepeat_bound": True,
    }


def test_all_frozen_gates_are_required_for_confirmation() -> None:
    result = resolve_confirmation_decision(_all_pass())
    assert result["decision"] == "bounded_motivation_confirmed"
    assert result["all_five_gates_passed"] is True
    assert all(row["status"] == "passed" for row in result["gates"])


@pytest.mark.parametrize(
    ("gate", "decision"),
    [
        ("population_power", "underpowered_inconclusive"),
        ("task_adequacy", "task_inadequate_inconclusive"),
        ("target_repeat_positive_control", "bounded_motivation_rejected"),
        ("recurrence_transfer_separation", "bounded_motivation_rejected"),
        ("practical_nonrepeat_bound", "bounded_motivation_rejected"),
    ],
)
def test_first_failure_controls_hierarchical_decision(
    gate: str, decision: str
) -> None:
    gates = _all_pass()
    gates[gate] = False
    result = resolve_confirmation_decision(gates)
    assert result["decision"] == decision
    assert result["first_failure"] == gate
    failure_index = next(
        index for index, row in enumerate(result["gates"]) if row["name"] == gate
    )
    assert result["gates"][failure_index]["status"] == "failed"
    assert all(
        row["status"] == "reported_not_interpreted"
        for row in result["gates"][failure_index + 1 :]
    )


def test_rejects_missing_or_extra_gate() -> None:
    gates = _all_pass()
    del gates["population_power"]
    gates["rescue"] = True
    with pytest.raises(ValueError, match="gate mismatch"):
        resolve_confirmation_decision(gates)
