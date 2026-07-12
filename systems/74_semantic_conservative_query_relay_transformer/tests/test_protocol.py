from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.semantic_relay import MODES  # noqa: E402
from probe.synthetic import make_dataset  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/design_gate.yaml").read_text())


def test_modes_authorization_and_fresh_seed() -> None:
    row = config()
    assert tuple(row["model"]["modes"]) == MODES
    assert row["data"]["generator_seed"] == 20265100
    assert row["data"]["generator_seed"] not in {20265000, 20265001}
    for name in ("repository_data", "repository_labels", "dev", "test", "qrels"):
        assert row["authorization"][name] is False


def test_external_generator_is_the_registered_file() -> None:
    row = config()
    path = REPO / row["paths"]["c73_generator"]
    assert path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_fresh_generator_is_deterministic() -> None:
    row = config()
    a = make_dataset(row, examples=64, seed=20265100, split="train")
    b = make_dataset(row, examples=64, seed=20265100, split="train")
    assert torch.equal(a.query_tokens, b.query_tokens)
    assert torch.equal(a.history_tokens, b.history_tokens)
    assert torch.equal(a.labels, b.labels)
