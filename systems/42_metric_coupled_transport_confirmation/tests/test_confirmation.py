from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))

from train.run_confirmation import ndcg_rows, uniform_correction  # noqa: E402


def test_uniform_control_contract() -> None:
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(16, generator=generator)
    history = torch.randn(5, 16, generator=generator)
    candidates = torch.randn(7, 16, generator=generator)
    config = {"model": {"profile_scale": 1.0, "correction_scale": 2.0}}
    assert torch.equal(uniform_correction(query, history[:0], candidates, config), torch.zeros(7))
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    expected = uniform_correction(query, history, candidates, config)[permutation]
    actual = uniform_correction(query, history, candidates[permutation], config)
    assert torch.allclose(actual, expected, atol=2e-7, rtol=0)


def test_shared_ndcg_top_positive() -> None:
    value = ndcg_rows(
        ["r"],
        [["a", "b", "c"]],
        [np.asarray([3.0, 2.0, 1.0])],
        [np.asarray([1.0, 0.0, 0.0])],
    )
    assert np.array_equal(value, np.asarray([1.0]))


def test_selection_is_unmaterialized_escrow() -> None:
    spec = importlib.util.spec_from_file_location(
        "c42_selection", ROOT / "train/materialize_selection.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    value = module.build(
        c38_selection_path=REPO
        / "artifacts/c38_cross_domain_global_tangent_transfer/train_gate_v1/selection.json",
        prior_feature_index_paths=[
            REPO / "artifacts/c38_cross_domain_global_tangent_transfer/train_gate_v1/features/feature_request_indices.npy",
            REPO / "artifacts/c39_halfspace_certified_value_transformer/train_gate_v1/features/feature_request_indices.npy",
            REPO / "artifacts/c41_semantic_carrier_routing_transformer/train_gate_v1/features/feature_request_indices.npy",
        ],
    )
    assert len(value["roles"]["internal_A"]["indices"]) == 1200
    assert value["outcome_isolation"]["internal_A_overlap_any_prior_feature_materialized"] == 0
    assert value["wrong_donor_audit"]["coverage_fraction"] == 1.0
    assert not value["label_access"]["records_train_labels_opened"]
