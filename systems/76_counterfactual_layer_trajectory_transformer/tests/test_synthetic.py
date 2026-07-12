import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe.synthetic import make_surface  # noqa: E402


def test_surface_has_frozen_strata_and_corruptions() -> None:
    surface = make_surface(
        requests=128,
        candidates=8,
        history_events=6,
        attributes=8,
        values_per_attribute=8,
        seed=9,
        split="validation",
    )
    assert surface.tokens.shape == (128, 8, 46)
    assert torch.equal(surface.labels.sum(-1), torch.ones(128))
    assert set(surface.strata.tolist()) == {0, 1, 2}
    supported = surface.strata.eq(0)
    assert bool((surface.tokens[supported] != surface.wrong_tokens[supported]).any())
    assert bool((surface.tokens[supported] != surface.shuffled_tokens[supported]).any())
    assert bool((surface.tokens[supported] != surface.query_masked_tokens[supported]).any())
    assert bool(surface.history_present[surface.strata.eq(2)].logical_not().all())
    assert bool(surface.repeat_present[surface.strata.eq(1)].all())
