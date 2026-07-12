import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.token_edge_attribution import (  # noqa: E402
    CANDIDATE,
    HISTORY,
    PADDING,
    QUERY,
    additive_attention_mask,
    allowed_attention_edges,
)


def segments() -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    segment = torch.tensor([[QUERY, QUERY, CANDIDATE, HISTORY, PADDING]], dtype=torch.int8)
    return valid, segment


def test_history_isolation_blocks_every_cross_history_edge() -> None:
    valid, segment = segments()
    edges = allowed_attention_edges(valid, segment, "history_isolated")[0]
    assert not bool(edges[0, 3])
    assert not bool(edges[2, 3])
    assert not bool(edges[3, 0])
    assert not bool(edges[3, 2])
    assert bool(edges[0, 2])
    assert bool(edges[3, 3])
    assert not bool(edges[:, 4].any())
    assert not bool(edges[4, :].any())


def test_pair_and_directional_masks_have_registered_orientation() -> None:
    valid, segment = segments()
    no_qh = allowed_attention_edges(valid, segment, "no_query_history")[0]
    assert not bool(no_qh[0, 3]) and not bool(no_qh[3, 0])
    assert bool(no_qh[2, 3]) and bool(no_qh[3, 2])
    no_c_read = allowed_attention_edges(valid, segment, "no_candidate_reads_history")[0]
    assert not bool(no_c_read[2, 3])
    assert bool(no_c_read[3, 2])
    no_h_context = allowed_attention_edges(valid, segment, "no_history_reads_context")[0]
    assert not bool(no_h_context[3, 0]) and not bool(no_h_context[3, 2])
    assert bool(no_h_context[0, 3]) and bool(no_h_context[2, 3])


def test_additive_mask_is_four_dimensional_and_finite_on_allowed_edges() -> None:
    valid, segment = segments()
    mask = additive_attention_mask(valid, segment, "no_candidate_history")
    assert mask.shape == (1, 1, 5, 5)
    assert float(mask[0, 0, 0, 2]) == 0.0
    assert float(mask[0, 0, 2, 3]) == torch.finfo(torch.float32).min
