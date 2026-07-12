from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthetic import SyntheticSpec, construct_audit, corrupt_history, generate_batch


def test_generator_is_deterministic():
    spec = SyntheticSpec()
    first = generate_batch(spec, 128, torch.Generator().manual_seed(11))
    second = generate_batch(spec, 128, torch.Generator().manual_seed(11))
    for name in first.__dict__:
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_repeat_membership_contract_is_exact():
    spec = SyntheticSpec()
    batch = generate_batch(spec, 4096, torch.Generator().manual_seed(12))
    row = torch.arange(batch.targets.shape[0])
    positive = batch.candidate_tokens[row, batch.targets, 0]
    membership = positive[:, None].eq(batch.history_tokens[:, :, 0]).any(dim=1)
    assert torch.equal(membership, batch.exact_repeat)


def test_locked_scale_construct_audit_removes_variant_shortcut():
    spec = SyntheticSpec()
    batch = generate_batch(spec, 32768, torch.Generator().manual_seed(2026071191))
    audit = construct_audit(batch, spec)
    assert audit["target_position_max_deviation"] <= 0.01
    assert audit["variant_total_variation"] <= 0.02
    assert audit["attribute_variant_total_variation"] <= 0.04
    assert audit["exact_membership_ok"]
    assert audit["hard_negative_count_ok"]
    assert abs(audit["positive_variant_zero_rate"] - audit["negative_variant_zero_rate"]) <= 0.01


def test_corruptions_keep_shapes_and_query_mask_is_evidence_only():
    spec = SyntheticSpec()
    batch = generate_batch(spec, 32, torch.Generator().manual_seed(13))
    for kind in ("wrong_user", "shuffle_events", "query_mask"):
        history, query = corrupt_history(
            batch, kind, torch.Generator().manual_seed(14)
        )
        assert history.shape == batch.history_tokens.shape
        if kind == "query_mask":
            assert query is not None
            assert torch.equal(query[:, 0], batch.query_tokens[:, 0])
            assert torch.all(query[:, 1] == 2)
        else:
            assert query is None
