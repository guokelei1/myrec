import numpy as np

from myrec.mechanism.q1_cache_phase_evaluator import _max_map_delta


def test_cache_phase_map_delta_is_candidate_aligned():
    left = {"r0": {"a": 1.0, "b": -2.0}, "r1": {"c": 0.5}}
    right = {"r0": {"a": 0.5, "b": -1.0}, "r1": {"c": 0.5}}
    assert _max_map_delta(left, right) == 1.0

