from myrec.mechanism.swiglu_formation_scoring import SWIGLU_CONDITIONS


def test_swiglu_condition_family_covers_all_operator_paths():
    assert len(SWIGLU_CONDITIONS) == 50
    assert SWIGLU_CONDITIONS[:2] == ("baseline_full", "baseline_null")
    for path in ("full", "null"):
        for operator in ("gate_pre", "up", "silu_gate", "product"):
            assert f"{path}_{operator}_identity" in SWIGLU_CONDITIONS
            assert f"{path}_{operator}_output_norm_matched_random" in SWIGLU_CONDITIONS

