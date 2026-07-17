from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from myrec.baselines.motivation_v12_ranker import _checkpoint_identity
from myrec.data.history_assignments import (
    current_motivation_v12_assignment_implementation_identity,
    motivation_v12_assignment_recipe,
)
from myrec.data.kuaisearch_holdout import (
    V12_EVALUATOR_IMPLEMENTATION_FILES,
    V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES,
    _current_analysis_selection_implementation_identity,
    _load_post_selection_lock,
)
from myrec.data.motivation_v12_release_lock import (
    LOCK_FILENAME,
    Q_METHOD_IDS,
    V12_METHOD_IDS,
    W0_METHOD_ID,
    _current_implementation_identity,
    build_motivation_v12_release_lock,
)
from myrec.utils.hashing import sha256_file, sha256_text


SELECTION_RULE = "final_finite_non_degenerate_completed_epoch"
SEED = 20260714


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _checkpoint_digest(entries: list[dict[str, object]]) -> str:
    return sha256_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))


def _fixture(tmp_path: Path) -> dict[str, object]:
    raw = tmp_path / "raw"
    _write_bytes(raw / "recall" / "train.jsonl", b"opaque mixed source bytes\n")
    _write_bytes(raw / "items" / "train.jsonl", b"opaque item source bytes\n")
    source_test = raw / "recall" / "test.jsonl"
    _write_bytes(source_test, b"source-test sentinel must remain untouched\n")

    development = tmp_path / "development"
    development_files = {
        "manifest": "manifest.json",
        "candidate_manifest": "candidate_manifest.json",
        "request_manifest": "request_manifest.json",
        "records_train": "records_train.jsonl",
        "records_dev": "records_dev.jsonl",
        "records_confirmation": "records_confirmation.jsonl",
        "qrels_train": "qrels_train.jsonl",
        "qrels_dev": "qrels_dev.jsonl",
    }
    for index, (name, filename) in enumerate(development_files.items()):
        # The qrels fixtures intentionally are not JSON.  A successful build
        # demonstrates that the builder treats them as opaque bytes.
        prefix = b"not-json-qrels" if name.startswith("qrels") else b"fixture"
        _write_bytes(development / filename, prefix + f"-{index}\n".encode())

    scout = tmp_path / "scout"
    for index, filename in enumerate(
        (
            "manifest.json",
            "candidate_manifest.json",
            "request_manifest.json",
            "records_train.jsonl",
            "records_dev.jsonl",
        )
    ):
        _write_bytes(scout / filename, f"scout-{index}\n".encode())

    population_fields = {
        "manifest_sha256": sha256_file(development / "manifest.json"),
        "candidate_manifest_sha256": sha256_file(
            development / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            development / "request_manifest.json"
        ),
        "records_train_sha256": sha256_file(development / "records_train.jsonl"),
        "records_dev_sha256": sha256_file(development / "records_dev.jsonl"),
        "records_legacy_compatibility_sha256": sha256_file(
            development / "records_confirmation.jsonl"
        ),
        "qrels_train_sha256": sha256_file(development / "qrels_train.jsonl"),
        "qrels_dev_sha256": sha256_file(development / "qrels_dev.jsonl"),
    }
    protocol = tmp_path / "protocol.yaml"
    protocol_payload = {
        "schema_version": 1,
        "protocol_id": "tiny_motivation_v12",
        "status": "pre_pilot_frozen",
        "seed_policy": {"pilot_seed": SEED},
        "common_training": {"checkpoint_selection": SELECTION_RULE},
        "data": {"development_population": population_fields},
    }
    protocol.write_text(
        yaml.safe_dump(protocol_payload, sort_keys=False), encoding="utf-8"
    )
    protocol_sha256 = sha256_file(protocol)

    configs: dict[str, Path] = {}
    for method_id in sorted(V12_METHOD_IDS):
        config = tmp_path / "configs" / f"{method_id}.yaml"
        payload = {
            "schema_version": 1,
            "method_id": method_id,
            "protocol": {
                "path": str(protocol.resolve()),
                "sha256": protocol_sha256,
            },
            "training": {
                "seed": SEED,
                "epochs": 2 if method_id == W0_METHOD_ID else 1,
            },
        }
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        configs[method_id] = config

    implementation_digest = _current_implementation_identity("q")[0]["digest"]
    q_roots: dict[str, Path] = {}
    for method_id in sorted(Q_METHOD_IDS):
        root = tmp_path / "checkpoints" / method_id
        model_dir = root / "checkpoint_latest" / "model"
        _write_bytes(model_dir / "config.json", f"config-{method_id}\n".encode())
        _write_bytes(
            model_dir / "model.safetensors", f"weights-{method_id}\n".encode()
        )
        _write_bytes(model_dir / "tokenizer.json", b"tokenizer\n")
        entries = [
            {
                "name": str(path.relative_to(model_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(model_dir.rglob("*"))
            if path.is_file()
        ]
        checkpoint_sha256 = _checkpoint_digest(entries)
        checkpoint_id = f"{method_id}@{checkpoint_sha256[:20]}"
        metadata = {
            "method_id": method_id,
            "status": "completed",
            "evidence_mode": "first_round_pilot",
            "seed": SEED,
            "protocol_sha256": protocol_sha256,
            "config_sha256": sha256_file(configs[method_id]),
            "implementation_identity": {"digest": implementation_digest},
            "checkpoint_id": checkpoint_id,
            "checkpoint_weight_files": entries,
            "resume_state_complete": True,
            "progress": {"epoch": 1, "batch_cursor": 0},
        }
        _write_json(root / "training_metadata.json", metadata)
        q_roots[method_id] = root

    w0_dir = tmp_path / "checkpoints" / W0_METHOD_ID
    _write_bytes(w0_dir / "model.pt", b"tiny W0 state dict bytes\n")
    w0_model_sha256 = sha256_file(w0_dir / "model.pt")
    w0_implementation = _current_implementation_identity("w0")[0]["digest"]
    _write_json(
        w0_dir / "metadata.json",
        {
            "method_id": W0_METHOD_ID,
            "status": "completed",
            "evidence_mode": "first_round_pilot",
            "seed": SEED,
            "protocol_sha256": protocol_sha256,
            "config_sha256": sha256_file(configs[W0_METHOD_ID]),
            "implementation_identity": {"digest": w0_implementation},
            "implementation_digest": w0_implementation,
            "checkpoint_id": f"{W0_METHOD_ID}@{w0_model_sha256[:20]}",
            "model_sha256": w0_model_sha256,
            "training": {"next_epoch": 2, "next_batch_cursor": 0},
        },
    )
    return {
        "raw": raw,
        "development": development,
        "scout": scout,
        "protocol": protocol,
        "configs": configs,
        "q_roots": q_roots,
        "w0_dir": w0_dir,
        "source_test": source_test,
        "output": tmp_path / "release_lock",
    }


def _build(fixture: dict[str, object]) -> dict[str, object]:
    return build_motivation_v12_release_lock(
        protocol_path=fixture["protocol"],
        config_paths=fixture["configs"],
        q_checkpoint_roots=fixture["q_roots"],
        w0_checkpoint_dir=fixture["w0_dir"],
        raw_dir=fixture["raw"],
        development_dir=fixture["development"],
        subsequent_scout_dir=fixture["scout"],
        output_lock_dir=fixture["output"],
        lock_id="tiny_post_selection_lock",
        command_argv=["tiny-release-lock-test"],
    )


def test_builds_schema_exact_atomic_release_without_opening_qrels(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_test_before = Path(fixture["source_test"]).read_bytes()
    result = _build(fixture)

    output = Path(fixture["output"])
    lock_path = output / LOCK_FILENAME
    assert result["passed"] is True
    assert result["published_atomically"] is True
    assert result["qrels_deserialized"] is False
    assert result["source_test_resolved_or_opened"] is False
    assert lock_path.is_file()
    assert len(list(output.glob("*.checkpoint_selection_identity.json"))) == 5
    assert Path(fixture["source_test"]).read_bytes() == source_test_before

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert set(payload["frozen_configs"]) == set(V12_METHOD_IDS)
    assert set(payload["frozen_checkpoints"]) == set(V12_METHOD_IDS)
    implementation = payload["analysis_selection_implementation"]
    assert implementation == _current_analysis_selection_implementation_identity()
    assert {
        entry["path"]
        for entry in implementation["holdout_selection"]["files"]
    } == set(V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES)
    assert {
        entry["path"] for entry in implementation["evaluator"]["files"]
    } == set(V12_EVALUATOR_IMPLEMENTATION_FILES)
    assert result["analysis_selection_implementation"] == {
        "schema_version": 1,
        "canonical_digest": implementation["canonical_digest"],
        "holdout_selection_digest": implementation["holdout_selection"][
            "digest"
        ],
        "evaluator_digest": implementation["evaluator"]["digest"],
    }
    assignment_release = payload["history_assignment_release"]
    assert assignment_release == {
        "schema_version": 1,
        "recipe": motivation_v12_assignment_recipe(),
        "implementation": (
            current_motivation_v12_assignment_implementation_identity()
        ),
    }
    assert result["history_assignment_release"] == {
        "schema_version": 1,
        "conditions": ["full", "null", "wrong"],
        "seed": 20260714,
        "global_donor_shortlist_size": 512,
        "implementation_digest": assignment_release["implementation"]["digest"],
    }
    assert set(payload["input_sha256"]) == {
        "source_recall_train",
        "source_items_train",
        "development_manifest",
        "development_candidate_manifest",
        "development_request_manifest",
        "development_records_train",
        "development_records_dev",
        "development_records_confirmation",
        "development_qrels_train",
        "development_qrels_dev",
        "subsequent_scout_manifest",
        "subsequent_scout_candidate_manifest",
        "subsequent_scout_request_manifest",
        "subsequent_scout_records_train",
        "subsequent_scout_records_dev",
    }
    loaded = _load_post_selection_lock(
        lock_path,
        protocol_sha256=sha256_file(fixture["protocol"]),
        checkpoint_selection_rule=SELECTION_RULE,
        pilot_seed=SEED,
    )
    assert set(loaded["checkpoint_identities"]) == set(V12_METHOD_IDS)
    for method_id in Q_METHOD_IDS:
        ranker_checkpoint_id, ranker_files = _checkpoint_identity(
            Path(fixture["q_roots"][method_id])
            / "checkpoint_latest"
            / "model",
            method_id,
        )
        frozen = loaded["checkpoint_identities"][method_id]
        assert frozen["checkpoint_id"] == ranker_checkpoint_id
        assert [
            {key: entry[key] for key in ("name", "sha256", "size_bytes")}
            for entry in frozen["checkpoint_files"]
        ] == ranker_files


def test_builder_implementation_digests_match_trainers() -> None:
    from myrec.baselines.copps_transfer_witness import (
        _implementation_identity as w0_implementation_identity,
    )
    from myrec.baselines.motivation_v12_ranker import (
        _implementation_identity as q_implementation_identity,
    )

    assert _current_implementation_identity("q")[0] == q_implementation_identity()
    assert _current_implementation_identity("w0")[0] == w0_implementation_identity()


def test_release_lock_rejects_assignment_recipe_or_generator_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture)
    lock_path = Path(fixture["output"]) / LOCK_FILENAME
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    original_release = json.loads(
        json.dumps(payload["history_assignment_release"])
    )
    payload["history_assignment_release"]["recipe"]["seed"] = 7
    _write_json(lock_path, payload)
    with pytest.raises(ValueError, match="assignment recipe"):
        _load_post_selection_lock(
            lock_path,
            protocol_sha256=sha256_file(fixture["protocol"]),
            checkpoint_selection_rule=SELECTION_RULE,
            pilot_seed=SEED,
        )
    payload["history_assignment_release"] = original_release
    payload["history_assignment_release"]["implementation"]["files"][0][
        "sha256"
    ] = "0" * 64
    _write_json(lock_path, payload)
    with pytest.raises(ValueError, match="assignment implementation differs"):
        _load_post_selection_lock(
            lock_path,
            protocol_sha256=sha256_file(fixture["protocol"]),
            checkpoint_selection_rule=SELECTION_RULE,
            pilot_seed=SEED,
        )


def test_tampered_checkpoint_is_rejected_by_final_lock(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture)
    method_id = "q1_instructrec_generalqwen"
    model = (
        Path(fixture["q_roots"][method_id])
        / "checkpoint_latest"
        / "model"
        / "model.safetensors"
    )
    model.write_bytes(model.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checkpoint file (size|SHA) mismatch"):
        _load_post_selection_lock(
            Path(fixture["output"]) / LOCK_FILENAME,
            protocol_sha256=sha256_file(fixture["protocol"]),
            checkpoint_selection_rule=SELECTION_RULE,
            pilot_seed=SEED,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "safe_exit_pending"),
        ("evidence_mode", "smoke_non_result"),
    ],
)
def test_rejects_nonfinal_training_state_without_publication(
    tmp_path: Path, field: str, value: str
) -> None:
    fixture = _fixture(tmp_path)
    method_id = "q0_qwen3_reranker_06b"
    metadata_path = Path(fixture["q_roots"][method_id]) / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match=f"{field}"):
        _build(fixture)
    assert not Path(fixture["output"]).exists()


def test_rejects_training_implementation_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    method_id = "q2_recranker_generalqwen"
    metadata_path = Path(fixture["q_roots"][method_id]) / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["implementation_identity"]["digest"] = "0" * 64
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="implementation differs"):
        _build(fixture)
    assert not Path(fixture["output"]).exists()


def test_rejects_w0_implementation_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    metadata_path = Path(fixture["w0_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["implementation_identity"]["digest"] = "0" * 64
    metadata["implementation_digest"] = "0" * 64
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="implementation differs"):
        _build(fixture)
    assert not Path(fixture["output"]).exists()


def test_release_directory_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _build(fixture)
    first_sha256 = sha256_file(first["lock_path"])

    with pytest.raises(FileExistsError, match="already exists"):
        _build(fixture)
    assert sha256_file(first["lock_path"]) == first_sha256
