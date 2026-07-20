from __future__ import annotations

import pytest

from myrec.mechanism.q0_representation_prompt import q0_context_spans
from myrec.mechanism.representation_probe import MechanicalPositionError


def test_q0_context_spans_are_hand_computed():
    context = (
        "<Instruct>: rank\n<Query>: shoes\n<Prior user history>:\n"
        "1. item=a\n"
    )
    query_span, history_span = q0_context_spans(
        context, query="shoes", history_text="1. item=a\n"
    )
    assert context[slice(*query_span)] == "shoes"
    assert context[slice(*history_span)] == "1. item=a\n"
    assert query_span[1] < history_span[0]


def test_q0_context_spans_reject_template_or_bytes_drift():
    with pytest.raises(MechanicalPositionError, match="uniqueness"):
        q0_context_spans("<Query>: q", query="q", history_text="h")
    context = "<Query>: q\n<Prior user history>:\nh"
    with pytest.raises(MechanicalPositionError, match="query bytes"):
        q0_context_spans(context, query="x", history_text="h")
