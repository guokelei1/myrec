from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


SYSTEM = Path(__file__).resolve().parents[1]
REPO = SYSTEM.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SYSTEM))

from model.pitt import (  # noqa: E402
    CANDIDATE,
    HISTORY,
    PADDING,
    QUERY,
    admitted_tokens,
    attention_graphs,
    combine_candidate_scores,
    pack_candidate_graph,
)


class TinyData:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    query_token_ids = np.asarray([[10, 11, 0]])
    query_attention = np.asarray([[1, 1, 0]], dtype=bool)
    item_token_ids = np.asarray([[20, 21, 0], [30, 31, 0], [40, 41, 0]])
    item_attention = np.asarray([[1, 1, 0]] * 3, dtype=bool)

    @staticmethod
    def _tokens(ids, mask, limit):
        return [int(value) for value in ids[mask][:limit]]

    @staticmethod
    def history(request_index, scenario, max_history):
        del request_index
        if scenario == "null":
            return np.asarray([], dtype=np.int64)
        values = np.asarray([1, 2], dtype=np.int64)
        return values[::-1].copy() if scenario == "shuffle" else values


TOKEN = {
    "query_tokens": 3,
    "candidate_tokens": 3,
    "history_item_tokens": 3,
    "max_history": 2,
    "max_sequence_length": 24,
}


def test_event_set_positions_ignore_event_order() -> None:
    true = pack_candidate_graph(TinyData(), 0, 0, scenario="true", token_config=TOKEN)
    shuffled = pack_candidate_graph(TinyData(), 0, 0, scenario="shuffle", token_config=TOKEN)
    true_events = [
        true.set_position_ids[true.event_ids == event].tolist() for event in (0, 1)
    ]
    shuffled_events = [
        shuffled.set_position_ids[shuffled.event_ids == event].tolist() for event in (0, 1)
    ]
    assert true_events == shuffled_events
    assert true.input_ids[true.event_ids == 0].tolist() == [30, 31, 2]
    assert shuffled.input_ids[shuffled.event_ids == 0].tolist() == [40, 41, 2]


def test_triadic_admission_requires_shared_query_support() -> None:
    ids = torch.tensor([[10, 20, 30, 40]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    segments = torch.tensor([[QUERY, CANDIDATE, HISTORY, HISTORY]])
    events = torch.tensor([[PADDING, PADDING, 0, 0]])
    anchors = torch.zeros(64, 3)
    anchors[10] = torch.tensor([1.0, 0.0, 0.0])
    anchors[20] = torch.tensor([1.0, 1.0, 0.0]).div(2**0.5)
    anchors[30] = torch.tensor([1.0, 1.0, 0.0]).div(2**0.5)
    anchors[40] = torch.tensor([0.0, 0.0, 1.0])
    active = admitted_tokens(
        ids,
        valid,
        segments,
        events,
        anchors,
        mode="triadic_set",
        candidate_budget=1,
        history_budget_per_event=1,
        max_history=1,
        special_token_ids=(1, 2, 0),
    )
    assert active.tolist() == [[True, True, True, False]]


def test_history_cut_removes_only_cross_history_edges() -> None:
    active = torch.tensor([[True, True, True]])
    segments = torch.tensor([[QUERY, CANDIDATE, HISTORY]])
    full, cut = attention_graphs(active, segments)
    assert bool(full.all())
    assert bool(cut[0, 0, 1])
    assert not bool(cut[0, 0, 2])
    assert not bool(cut[0, 2, 1])
    assert bool(cut[0, 2, 2])


def test_candidate_correction_is_centered_per_request() -> None:
    base = torch.tensor([1.0, 2.0, 5.0, 6.0])
    raw = torch.tensor([1.0, 3.0, -2.0, 2.0])
    output = combine_candidate_scores(base, raw, [2, 2])
    assert torch.allclose(output, torch.tensor([0.0, 3.0, 3.0, 8.0]))
    assert torch.allclose((output - base).reshape(2, 2).mean(-1), torch.zeros(2))
