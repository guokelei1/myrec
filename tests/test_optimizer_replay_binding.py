from __future__ import annotations

import random

import pytest

from myrec.mechanism.optimizer_replay_binding import (
    audit_loaded_step500_state,
    restore_bound_rng_state,
)


def _fixture():
    torch = pytest.importorskip("torch")
    binding = {
        "checkpoint_id": "q3@abc",
        "optimizer_steps": 500,
        "optimizer_parameter_count": 2,
        "scheduler_last_epoch": 500,
        "scheduler_step_count": 501,
        "current_lr": 0.001,
        "parameter_order_digest": "digest",
    }
    progress = {"batch_cursor": 4, "epoch": 0, "micro_steps": 4, "optimizer_steps": 500}
    parameter_state = {
        index: {
            "step": torch.tensor(500.0),
            "exp_avg": torch.zeros(index + 1),
            "exp_avg_sq": torch.ones(index + 1),
        }
        for index in range(2)
    }
    state = {
        "checkpoint_id": "q3@abc",
        "config_sha256": "config",
        "optimizer": {
            "state": parameter_state,
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                }
            ],
        },
        "progress": progress,
        "rng": {"python": None, "torch_cpu": None, "torch_cuda": None},
        "scaler": {},
        "scheduler": {"last_epoch": 500, "_step_count": 501, "_last_lr": [0.001]},
        "training_contract": {},
    }
    return state, progress, binding


def test_step500_binding_audits_every_optimizer_object():
    state, progress, binding = _fixture()
    audit = audit_loaded_step500_state(
        state, progress, binding, checkpoint_id="q3@abc"
    )
    assert audit["optimizer_parameter_count"] == 2
    assert audit["optimizer_steps"] == 500
    assert audit["all_moments_finite"] is True


def test_step500_binding_rejects_parameter_or_scheduler_drift():
    state, progress, binding = _fixture()
    state["optimizer"]["param_groups"][0]["params"] = [1, 0]
    with pytest.raises(ValueError, match="index coverage"):
        audit_loaded_step500_state(state, progress, binding, checkpoint_id="q3@abc")
    state, progress, binding = _fixture()
    state["scheduler"]["_step_count"] = 500
    with pytest.raises(ValueError, match="step count"):
        audit_loaded_step500_state(state, progress, binding, checkpoint_id="q3@abc")


def test_single_gpu_replay_restores_original_training_device_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    cuda_states = [torch.arange(16, dtype=torch.uint8) + index for index in range(4)]
    observed = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def set_rng_state(state, device=None):
        observed["state"] = state.detach().clone()
        observed["device"] = str(device)

    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device=None: observed["state"].detach().clone(),
    )
    audit = restore_bound_rng_state(
        torch,
        {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_states,
        },
        method_id="q3_tallrec_generalqwen",
        device="cuda:0",
    )
    assert observed["device"] == "cuda:0"
    assert torch.equal(observed["state"], cuda_states[3])
    assert audit["saved_cuda_rng_state_count"] == 4
    assert audit["training_logical_cuda_index"] == 3
    assert audit["restore_identity_exact"] is True
