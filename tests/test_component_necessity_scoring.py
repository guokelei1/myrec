from types import SimpleNamespace

import pytest

import myrec.mechanism.component_necessity_scoring as necessity


def test_component_necessity_design_is_fixed_and_separate_from_parent():
    assert necessity.NECESSITY_NODES == (
        "block_input_residual",
        "attention_o_projection",
        "mlp_down_projection",
        "block_output_residual",
    )
    assert necessity.NECESSITY_INTERVENTIONS == (
        "full_to_full_identity",
        "null_to_full_removal",
        "neutral_to_full_removal",
    )
    conditions = necessity.component_necessity_conditions()
    assert len(conditions) == 14
    assert len(set(conditions)) == 14
    assert conditions[:2] == ("baseline_full", "baseline_null")


def test_q2_reverse_removal_always_uses_full_recipient(monkeypatch):
    torch = pytest.importorskip("torch")
    full_batch = object()
    null_batch = object()
    record = SimpleNamespace(history=[{"item_id": "history"}])
    candidates = [{"item_id": "candidate"}]

    def fake_batch(_tokenizer, _record, _candidates, history, _config, *, device):
        assert device == "cpu"
        return full_batch if history else null_batch

    class FakeCapture:
        def __init__(self, _model, specs):
            self.specs = specs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_capture(_model, capture, batch):
        value = 10.0 if batch is full_batch else 2.0
        return {
            "score": torch.tensor([value]),
            "states": {
                spec.key: torch.tensor([[[value + index]]])
                for index, spec in enumerate(capture.specs)
            },
        }

    calls = []

    def fake_patch(_model, spec, recipient, donor):
        calls.append((spec.node_id, recipient, float(donor.item())))
        return donor[:, 0, 0].clone()

    monkeypatch.setattr(necessity, "build_q2_pointwise_batch", fake_batch)
    monkeypatch.setattr(necessity, "QwenNodeCapture", FakeCapture)
    monkeypatch.setattr(necessity, "_capture_q2", fake_capture)
    monkeypatch.setattr(necessity, "_patch_q2", fake_patch)

    result = necessity.score_component_necessity_chunk(
        object(),
        object(),
        record,
        candidates,
        {"method_id": "q2_recranker_generalqwen"},
        content_control={"eligible": False},
        block=20,
        device="cpu",
    )

    assert tuple(result["conditions"]) == necessity.component_necessity_conditions()
    assert len(calls) == 8
    assert all(recipient is full_batch for _node, recipient, _donor in calls)
    for index, node in enumerate(necessity.NECESSITY_NODES):
        identity_call, removal_call = calls[2 * index : 2 * index + 2]
        assert identity_call == (node, full_batch, 10.0 + index)
        assert removal_call == (node, full_batch, 2.0 + index)
        assert result["conditions"][f"{node}.full_to_full_identity"].item() == (
            10.0 + index
        )
        assert result["conditions"][f"{node}.null_to_full_removal"].item() == (
            2.0 + index
        )
        assert result["conditions"][f"{node}.neutral_to_full_removal"].item() == (
            10.0 + index
        )
    assert result["content_neutral_eligible"] is False
    assert result["neutral_path_identity_passed"] is False


def test_q2_position_preserving_neutral_removal_uses_full_recipient(monkeypatch):
    torch = pytest.importorskip("torch")
    full_batch = object()
    null_batch = object()
    neutral_ids = torch.tensor([[151643]])
    neutral_batch = (neutral_ids, torch.ones_like(neutral_ids), torch.zeros_like(neutral_ids))
    full_path = {"name": "prompt"}
    neutral_path = {
        "ids": neutral_batch[0],
        "mask": neutral_batch[1],
        "positions": neutral_batch[2],
    }
    record = SimpleNamespace(history=[{"item_id": "history"}])
    candidates = [{"item_id": "candidate"}]

    monkeypatch.setattr(
        necessity,
        "build_q2_pointwise_batch",
        lambda *args, **_kwargs: full_batch if args[3] else null_batch,
    )

    class FakeCapture:
        def __init__(self, _model, specs):
            self.specs = specs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_capture(_model, capture, batch):
        if batch is full_batch:
            value = 10.0
        elif batch is null_batch:
            value = 2.0
        else:
            assert batch[0] is neutral_ids
            value = 6.0
        return {
            "score": torch.tensor([value]),
            "states": {
                spec.key: torch.tensor([[[value + index]]])
                for index, spec in enumerate(capture.specs)
            },
        }

    calls = []

    def fake_patch(_model, spec, recipient, donor):
        calls.append((spec.node_id, recipient, float(donor.item())))
        return donor[:, 0, 0].clone()

    monkeypatch.setattr(necessity, "QwenNodeCapture", FakeCapture)
    monkeypatch.setattr(necessity, "_capture_q2", fake_capture)
    monkeypatch.setattr(necessity, "_patch_q2", fake_patch)
    monkeypatch.setattr(necessity, "_build_paths", lambda *_a, **_k: [full_path])
    monkeypatch.setattr(
        necessity, "_neutralize_paths", lambda _paths: [neutral_path]
    )
    monkeypatch.setattr(necessity, "_assert_q2_full_path_identity", lambda *_a: None)
    monkeypatch.setattr(necessity, "_assert_neutral_path_identity", lambda *_a: None)

    result = necessity.score_component_necessity_chunk(
        object(),
        object(),
        record,
        candidates,
        {"method_id": "q2_recranker_generalqwen"},
        content_control={"eligible": True},
        block=20,
        device="cpu",
    )
    assert len(calls) == 12
    assert all(recipient is full_batch for _node, recipient, _donor in calls)
    for index, node in enumerate(necessity.NECESSITY_NODES):
        identity, null_removal, neutral_removal = calls[3 * index : 3 * index + 3]
        assert identity == (node, full_batch, 10.0 + index)
        assert null_removal == (node, full_batch, 2.0 + index)
        assert neutral_removal == (node, full_batch, 6.0 + index)
    assert result["content_neutral_eligible"] is True
    assert result["neutral_path_identity_passed"] is True


def test_component_necessity_rejects_unregistered_method_or_bad_block():
    record = SimpleNamespace(history=[])
    candidates = [{"item_id": "candidate"}]
    with pytest.raises(ValueError, match=r"\[13,27\]"):
        necessity.score_component_necessity_chunk(
            object(), object(), record, candidates, {"method_id": "x"},
            content_control={"eligible": False}, block=12, device="cpu"
        )
    with pytest.raises(ValueError, match="only Q2/Q3"):
        necessity.score_component_necessity_chunk(
            object(), object(), record, candidates, {"method_id": "x"},
            content_control={"eligible": False}, block=20, device="cpu"
        )


def test_component_necessity_condition_validation_is_fail_closed():
    torch = pytest.importorskip("torch")
    conditions = {
        name: torch.zeros(1) for name in necessity.component_necessity_conditions()
    }
    necessity._validate_conditions(conditions, 1)
    reversed_conditions = dict(reversed(tuple(conditions.items())))
    with pytest.raises(ValueError, match="order or coverage"):
        necessity._validate_conditions(reversed_conditions, 1)
    conditions["baseline_full"] = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError, match="non-finite"):
        necessity._validate_conditions(conditions, 1)


def test_neutral_path_changes_only_registered_span_and_preserves_positions():
    torch = pytest.importorskip("torch")
    full = {
        "ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "mask": torch.ones(1, 5, dtype=torch.long),
        "positions": torch.tensor([[4]]),
        "starts": torch.tensor([1]),
        "ends": torch.tensor([4]),
    }
    neutral = {
        **full,
        "ids": torch.tensor([[1, 151643, 151643, 151643, 5]]),
    }
    necessity._assert_neutral_path_identity(full, neutral)
    bad = {**neutral, "ids": neutral["ids"].clone()}
    bad["ids"][0, 4] = 6
    with pytest.raises(RuntimeError, match="span exterior"):
        necessity._assert_neutral_path_identity(full, bad)
