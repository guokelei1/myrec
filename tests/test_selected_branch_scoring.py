from myrec.mechanism.selected_branch_scoring import (
    NODE_INTERVENTIONS,
    RANDOM_DIRECTION_SEED,
    SELECTED_NODES,
    _direction_controls,
    selected_branch_conditions,
)


def test_selected_branch_design_is_frozen_and_complete():
    assert SELECTED_NODES == (
        "block_input_residual",
        "input_rmsnorm_output",
        "attention_o_projection",
        "post_attention_residual",
        "post_attention_rmsnorm_output",
        "mlp_down_projection",
        "block_output_residual",
    )
    assert len(NODE_INTERVENTIONS) == 8
    conditions = selected_branch_conditions()
    assert len(conditions) == 58
    assert len(set(conditions)) == 58
    assert RANDOM_DIRECTION_SEED == 20260715


def test_selected_branch_direction_controls_match_registered_rms():
    import torch

    donor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    recipient = torch.tensor([[[4.0, 1.0, 2.0, 1.0]]])
    controls, geometry = _direction_controls(
        donor, recipient, identity_keys=[["fixed"]]
    )
    donor_rms = donor.square().mean(-1).sqrt()
    recipient_rms = recipient.square().mean(-1).sqrt()
    assert torch.allclose(
        controls["donor_direction_at_recipient_rms"].square().mean(-1).sqrt(),
        recipient_rms,
    )
    assert torch.allclose(
        controls["recipient_direction_at_donor_rms"].square().mean(-1).sqrt(),
        donor_rms,
    )
    assert geometry["d_at_r_rms_max_abs_error"] < 1.0e-6
