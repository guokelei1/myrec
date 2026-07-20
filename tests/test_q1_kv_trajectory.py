from __future__ import annotations

import pytest

from myrec.mechanism.q1_kv_trajectory import q1_context_spans
from myrec.mechanism.q1_trajectory_evaluator import _implementation_digest
from myrec.mechanism.representation_probe import MechanicalPositionError


@pytest.mark.parametrize(
    ("template", "context"),
    [
        (
            0,
            "Task form: x\nCurrent user intention (query): shoes\n"
            "Implicit preference evidence (newest first):\nhistory\nInstruction: x",
        ),
        (
            1,
            "intro\nSearch query: shoes\nInteraction history:\nhistory\nInstruction: x",
        ),
    ],
)
def test_q1_context_spans_cover_both_frozen_templates(template, context):
    query, history = q1_context_spans(
        context,
        query="shoes",
        history_text="history",
        template_index=template,
    )
    assert context[slice(*query)] == "shoes"
    assert context[slice(*history)] == "history"


def test_q1_context_spans_reject_template_drift():
    with pytest.raises(MechanicalPositionError, match="uniqueness"):
        q1_context_spans(
            "Search query: q", query="q", history_text="h", template_index=1
        )


def test_q1_trajectory_evaluator_binds_implementation_to_run_contract():
    metadata = {
        "implementation_identity": {"digest": "fixed"},
        "run_contract": {"implementation_digest": "fixed"},
    }
    assert _implementation_digest(metadata) == "fixed"
    metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _implementation_digest(metadata)
