"""Frozen dominant-FLOP accounting for the C06 minimal mechanism gate."""

from __future__ import annotations


def dominant_probe_flops(
    *,
    variant: str,
    input_dim: int,
    evidence_dim: int,
    candidates: int,
    history: int,
    centered_compute_rounds: int = 4,
) -> int:
    """Count shared projections and load-bearing `C*H*r^2` contractions.

    Elementwise nonlinearities, masks, reductions and softmax are lower-order
    and intentionally omitted for both variants. A multiply-add counts as two
    FLOPs. This accounting is frozen before GPU timing.
    """

    if min(input_dim, evidence_dim, candidates) <= 0 or history < 0:
        raise ValueError("invalid probe dimensions")
    shared = 2 * input_dim * evidence_dim * (1 + candidates + history)
    pair_rows = candidates * history
    if variant == "local_hodge":
        # Two factor maps (4), three Gram contractions (6), and three
        # candidate-local quadratic forms (6): 16*r^2 per pair row.
        mechanism = 16 * pair_rows * evidence_dim * evidence_dim
    elif variant == "centered_cross_attention":
        if centered_compute_rounds <= 0:
            raise ValueError("centered_compute_rounds must be positive")
        # Two tied r-by-r maps per round, each a multiply-add contraction.
        mechanism = (
            4
            * centered_compute_rounds
            * pair_rows
            * evidence_dim
            * evidence_dim
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    return shared + mechanism
