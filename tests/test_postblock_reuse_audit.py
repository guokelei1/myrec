from myrec.mechanism.postblock_reuse_audit import CONDITION_MAP, TOLERANCE


def test_q2_postblock_reuse_gate_is_narrow_and_strict():
    assert CONDITION_MAP == {
        "full_to_full_identity": "full_to_full_identity",
        "same_full_to_null": "same_request_full_to_null",
        "cross_full_to_null": "cross_request_same_layer",
    }
    assert TOLERANCE == 1.0e-5
