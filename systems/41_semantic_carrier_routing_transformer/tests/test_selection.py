from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "c41_selection", ROOT / "train" / "materialize_selection.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_real_selection_is_isolated() -> None:
    repo = ROOT.parents[1]
    value = module.build(
        c38_selection_path=repo
        / "artifacts/c38_cross_domain_global_tangent_transfer/train_gate_v1/selection.json",
        c39_selection_path=repo
        / "artifacts/c39_halfspace_certified_value_transformer/train_gate_v1/selection.json",
        c38_feature_indices_path=repo
        / "artifacts/c38_cross_domain_global_tangent_transfer/train_gate_v1/features/feature_request_indices.npy",
    )
    assert len(value["roles"]["fit"]["indices"]) == 6000
    assert len(value["roles"]["internal_A"]["indices"]) == 1200
    assert len(value["roles"]["delayed_B"]["indices"]) == 1200
    assert value["wrong_donor_audit"]["coverage_fraction"] == 1.0
    assert value["wrong_donor_audit"]["same_length_bin_fraction"] == 1.0
    assert not value["label_access"]["records_train_labels_opened"]
