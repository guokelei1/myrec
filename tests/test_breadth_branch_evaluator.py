from types import SimpleNamespace

import pytest

from myrec.mechanism.breadth_branch_evaluator import (
    BLOCKS,
    COMPARISONS,
    ENDPOINTS,
    FAMILY_SIZE,
    _per_method_implementation_digests,
)
from myrec.mechanism.q0_branch_scoring import Q0_BRANCH_NODES


def test_breadth_branch_family_size_is_exactly_96():
    assert 2 * len(BLOCKS) * len(Q0_BRANCH_NODES) * len(COMPARISONS) * len(ENDPOINTS) == FAMILY_SIZE
    assert COMPARISONS == ("same_minus_null", "same_minus_full")


def test_breadth_branch_digest_is_consistent_within_not_across_models():
    bundles = {
        "q0": {
            13: SimpleNamespace(metadata={"implementation_identity": {"digest": "q0"}, "run_contract": {"implementation_digest": "q0"}}),
            20: SimpleNamespace(metadata={"implementation_identity": {"digest": "q0"}, "run_contract": {"implementation_digest": "q0"}}),
        },
        "q1": {
            13: SimpleNamespace(metadata={"implementation_identity": {"digest": "q1"}, "run_contract": {"implementation_digest": "q1"}}),
            20: SimpleNamespace(metadata={"implementation_identity": {"digest": "q1"}, "run_contract": {"implementation_digest": "q1"}}),
        },
    }
    assert _per_method_implementation_digests(bundles) == {"q0": "q0", "q1": "q1"}
    bundles["q1"][20].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract: q1"):
        _per_method_implementation_digests(bundles)
    bundles["q1"][20].metadata["run_contract"]["implementation_digest"] = "q1"
    bundles["q1"][20].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests: q1"):
        _per_method_implementation_digests(bundles)
