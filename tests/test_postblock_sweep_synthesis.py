from myrec.mechanism.postblock_sweep_synthesis import _missing_row


def test_gate_stopped_cell_preserves_planned_family_with_p_one():
    row = _missing_row("q3_tallrec_generalqwen", block=13)
    assert row["two_sided_p"] == 1.0
    assert row["missing_cell"] is True
    assert row["mean"] is None
