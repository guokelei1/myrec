from myrec.mechanism.q0_branch_scoring import (
    Q0_BRANCH_CONDITIONS,
    Q0_BRANCH_NODES,
)


def test_q0_branch_nodes_and_conditions_are_fixed():
    assert Q0_BRANCH_NODES == (
        "block_input_residual",
        "attention_o_projection",
        "mlp_down_projection",
        "block_output_residual",
    )
    assert len(Q0_BRANCH_CONDITIONS) == 10
    assert len(set(Q0_BRANCH_CONDITIONS)) == 10
