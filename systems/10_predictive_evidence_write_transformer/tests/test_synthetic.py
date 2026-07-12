from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from importlib import import_module

synthetic = import_module("10_predictive_evidence_write_transformer.synthetic")


def test_generator_is_seed_deterministic_and_target_is_valid():
    spec = synthetic.SyntheticSpec()
    first = synthetic.generate_batch(spec, 32, torch.Generator().manual_seed(17))
    second = synthetic.generate_batch(spec, 32, torch.Generator().manual_seed(17))
    for name in first.__dict__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    row = torch.arange(32)
    positive = first.candidate_tokens[row, first.targets]
    query_category = first.query_tokens[:, 1]
    assert torch.equal(positive[:, 1], query_category)


def test_exact_repeat_flag_matches_positive_item_membership():
    spec = synthetic.SyntheticSpec()
    batch = synthetic.generate_batch(spec, 256, torch.Generator().manual_seed(19))
    positive = batch.candidate_tokens[torch.arange(256), batch.targets, 0]
    membership = positive[:, None].eq(batch.history_tokens[:, :, 0]).any(dim=1)
    assert torch.equal(membership, batch.exact_repeat)


def test_corruptions_preserve_shapes_and_query_mask_only_changes_evidence_query():
    spec = synthetic.SyntheticSpec()
    batch = synthetic.generate_batch(spec, 16, torch.Generator().manual_seed(23))
    for kind in ("wrong_user", "shuffle_events", "query_mask"):
        history, query = synthetic.corrupt_history(batch, kind, torch.Generator().manual_seed(29))
        assert history.shape == batch.history_tokens.shape
        if kind == "query_mask":
            assert query is not None
            assert torch.equal(query[:, 0], batch.query_tokens[:, 0])
            assert torch.all(query[:, 1] == spec.masked_query_token)
        else:
            assert query is None
