"""Frozen motivation-confirmation decision helpers."""

from __future__ import annotations

from typing import Any


CONFIRMATION_GATE_ORDER = (
    "population_power",
    "task_adequacy",
    "target_repeat_positive_control",
    "recurrence_transfer_separation",
    "practical_nonrepeat_bound",
)


def resolve_confirmation_decision(
    gate_passes: dict[str, bool],
) -> dict[str, Any]:
    """Apply the pre-registered hierarchy without allowing later rescue."""

    missing = set(CONFIRMATION_GATE_ORDER) - set(gate_passes)
    extra = set(gate_passes) - set(CONFIRMATION_GATE_ORDER)
    if missing or extra:
        raise ValueError(
            f"confirmation gate mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )

    first_failure: str | None = None
    gates = []
    for index, name in enumerate(CONFIRMATION_GATE_ORDER, start=1):
        passed = bool(gate_passes[name])
        interpreted = first_failure is None
        gates.append(
            {
                "gate": index,
                "name": name,
                "passed": passed,
                "status": (
                    "passed"
                    if interpreted and passed
                    else "failed"
                    if interpreted
                    else "reported_not_interpreted"
                ),
            }
        )
        if interpreted and not passed:
            first_failure = name

    if first_failure is None:
        decision = "bounded_motivation_confirmed"
    elif first_failure == "population_power":
        decision = "underpowered_inconclusive"
    elif first_failure == "task_adequacy":
        decision = "task_inadequate_inconclusive"
    else:
        decision = "bounded_motivation_rejected"
    return {
        "all_five_gates_passed": first_failure is None,
        "decision": decision,
        "first_failure": first_failure,
        "gates": gates,
    }
