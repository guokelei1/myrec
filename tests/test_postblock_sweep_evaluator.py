from types import SimpleNamespace

import pytest

from myrec.mechanism.postblock_sweep_evaluator import (
    _common_implementation_digest,
    select_registered_block,
)


def test_selector_uses_most_negative_step_and_lower_block_tie():
    values = {block: float(block) for block in range(13, 28)}
    values[17] = values[16] - 3.0
    values[23] = values[22] - 3.0
    selected, adjacent = select_registered_block(values)
    assert selected == 17
    assert adjacent[17] == pytest.approx(-3.0)


def test_selector_stops_when_no_negative_step():
    selected, adjacent = select_registered_block(
        {block: float(block) for block in range(13, 28)}
    )
    assert selected is None
    assert all(value >= 0 for value in adjacent.values())


def test_selector_rejects_missing_registered_block():
    with pytest.raises(ValueError, match="blocks 13 through 27"):
        select_registered_block({block: 0.0 for block in range(13, 27)})


def test_postblock_evaluator_requires_one_implementation_digest():
    bundles = {
        block: SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}}
        )
        for block in range(13, 28)
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles[20].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
