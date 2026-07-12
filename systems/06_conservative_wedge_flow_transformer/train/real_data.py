"""Label-isolated C06 cohort selection, batching, and frozen G0 features.

The structural loader deliberately has no label field.  Cohort IDs are chosen
and written before a caller can ask this module to open the train label array.
The real gate then keeps fit labels in a compact G0 artifact; internal-A labels
are opened only by the runner after its label-free A0 audit has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml


FORBIDDEN_PATH_TOKENS = (
    "qrels",
    "records_dev",
    "records_test",
    "per_request_metrics",
)
REAL_GATE_COUNTS = {
    "fit": 12_000,
    "internal_A": 1_200,
    "internal_B": 600,
    "escrow": 515,
    "nohistory": 512,
}
FEATURE_ROLES = ("fit", "internal_A", "nohistory")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("candidate_id") != "c06":
        raise ValueError("configuration is not C06")
    if config.get("gate_id") != "c06_real_mechanism_gate_v1":
        raise ValueError("unexpected C06 real-gate ID")
    if int(config.get("seed", -1)) != 20260708:
        raise ValueError("unexpected C06 seed")
    actual_counts = {
        role: int(config["selection"][f"{role}_requests"])
        for role in REAL_GATE_COUNTS
    }
    if actual_counts != REAL_GATE_COUNTS:
        raise ValueError(f"C06 cohort counts changed: {actual_counts}")
    variants = tuple(config["training"]["variants"])
    required_variants = (
        "local_hodge",
        "untrusted",
        "direct_learned",
        "centered_cross_attention",
    )
    if variants != required_variants:
        raise ValueError("C06 minimal-gate variants or order changed")
    if int(config["training"]["epochs"]) != 2:
        raise ValueError("C06 real gate is fixed to two epochs")
    expected_gpus = {
        "local_hodge": 0,
        "untrusted": 1,
        "direct_learned": 2,
        "centered_cross_attention": 3,
    }
    if config.get("resources", {}).get("variant_physical_gpus") != expected_gpus:
        raise ValueError("C06 variant/GPU mapping changed")
    if int(config["resources"].get("g0_physical_gpu", -1)) != 0:
        raise ValueError("C06 G0 must remain on physical GPU 0")
    if int(config["resources"].get("audit_physical_gpu", -1)) != 0:
        raise ValueError("C06 A0/A1 audit must remain on physical GPU 0")
    repair = config.get("numeric_repair", {})
    if repair.get("repair_id") != "c06_cycle_energy_explicit_row_fallback_review1":
        raise ValueError("unexpected C06 numeric-repair ID")
    if tuple(repair.get("eligible_variants", ())) != required_variants[:3]:
        raise ValueError("C06 numeric-repair eligible variants changed")
    if int(repair.get("repair_attempts_per_failed_variant", -1)) != 1:
        raise ValueError("C06 numeric repair permits exactly one retry")
    if repair.get("thresholds_model_data_seed_training_changed") is not False:
        raise ValueError("C06 numeric repair changed a scientific setting")
    if repair.get("centered_cross_attention_rerun_forbidden") is not True:
        raise ValueError("C06 numeric repair must preserve centered v1")
    if repair.get("fallback_dtype") != "float64" or repair.get(
        "fallback_complexity_per_row"
    ) != "O(C*r)":
        raise ValueError("C06 numeric-repair arithmetic changed")
    if repair.get(
        "fallback_requires_primitive_absolute_forward_error_identity"
    ) is not True:
        raise ValueError("C06 numeric repair lacks its primitive identity guard")
    parent_snapshot = Path(repair["parent_config_snapshot_path"])
    if sha256_file(parent_snapshot) != repair["parent_config_sha256"]:
        raise ValueError("C06 byte-identical parent config snapshot changed")
    with parent_snapshot.open("r", encoding="utf-8") as handle:
        parent_config = yaml.safe_load(handle)
    for name in (
        "candidate_id",
        "gate_id",
        "seed",
        "environment",
        "program_device",
        "run_prefix",
        "g0_run_id",
        "audit_run_id",
        "variant_run_ids",
        "resources",
        "integrity",
        "selection",
        "base",
        "model",
        "controls",
        "training",
        "a0_gate",
        "a1_gate",
        "authorization",
    ):
        if config.get(name) != parent_config.get(name):
            raise ValueError(f"C06 numeric repair changed parent setting: {name}")
    current_paths = dict(config["paths"])
    parent_paths = dict(parent_config["paths"])
    current_paths.pop("real_gate_lock")
    parent_paths.pop("real_gate_lock")
    if current_paths != parent_paths:
        raise ValueError("C06 numeric repair changed a parent data/output path")
    assert_path_firewall(config)
    return config


def assert_path_firewall(config: Mapping[str, Any]) -> None:
    for name, raw in config.get("paths", {}).items():
        lowered = str(raw).lower()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise ValueError(f"forbidden C06 path {name}: {raw}")
        if lowered.endswith("/metrics.json"):
            raise ValueError(f"forbidden C06 path {name}: {raw}")
    checkpoint = str(config["paths"]["calibration_checkpoint"])
    if "_final_" in checkpoint:
        raise ValueError("C06 may not use the final D2 checkpoint")
    popularity = str(config["paths"]["internal_train_popularity"])
    if "full_train" in popularity:
        raise ValueError("C06 may not use full-train popularity")


def assert_candidate_manifest(config: Mapping[str, Any]) -> str:
    actual = sha256_file(config["paths"]["candidate_manifest"])
    expected = str(config["paths"]["candidate_manifest_sha256"])
    if actual != expected:
        raise ValueError(f"candidate manifest hash mismatch: {actual} != {expected}")
    return actual


def assert_numeric_repair_lock_semantics(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    """Validate review1 declarations and the exact permitted repair scope."""

    declarations = lock.get("declarations", {})
    required_false = (
        "internal_A_features_scored_before_review1",
        "internal_A_labels_opened_before_review1",
        "internal_B_or_escrow_opened_before_review1",
        "dev_or_test_observed_before_review1",
        "internal_A_or_later_ranking_outcome_observed_before_review1",
        "thresholds_model_data_seed_or_training_changed",
        "repair_choice_used_A_or_comparative_ranking_quality",
    )
    if any(declarations.get(name) is not False for name in required_false):
        raise ValueError("numeric-repair lock lacks its no-outcome declaration")
    if declarations.get("numeric_implementation_failure_observed") is not True:
        raise ValueError("numeric-repair lock does not declare the observed failure")
    if declarations.get("fit_training_telemetry_observed") is not True:
        raise ValueError("numeric-repair lock hides observed fit telemetry")

    repair = config["numeric_repair"]
    expected_scope = {
        "repair_id": repair["repair_id"],
        "eligible_variants": list(repair["eligible_variants"]),
        "repair_attempts_per_variant": int(
            repair["repair_attempts_per_failed_variant"]
        ),
        "centered_cross_attention_rerun": False,
        "fallback_complexity_per_row": repair["fallback_complexity_per_row"],
        "fallback_dtype": repair["fallback_dtype"],
        "fallback_requires_primitive_absolute_forward_error_identity": repair[
            "fallback_requires_primitive_absolute_forward_error_identity"
        ],
    }
    if lock.get("repair_scope") != expected_scope:
        raise ValueError("numeric-repair lock scope differs from the reviewed config")


def assert_real_gate_lock(config: Mapping[str, Any]) -> str:
    """Require the reviewed pre-A numeric-repair lock before execution."""

    path = Path(config["paths"]["real_gate_lock"])
    if not path.exists():
        raise PermissionError(
            "C06 real-gate lock is absent; implementation review must precede execution"
        )
    lock = read_json(path)
    if lock.get("lock_id") != "c06_real_gate_numeric_repair_review1":
        raise ValueError("unexpected C06 numeric-repair lock")
    if lock.get("status") != "locked_after_pre_A_numeric_failure_before_any_A_access":
        raise ValueError("C06 numeric repair is not review1 locked")
    assert_numeric_repair_lock_semantics(config, lock)

    repair = config["numeric_repair"]
    parent_path = Path(repair["parent_lock_path"])
    if sha256_file(parent_path) != repair["parent_lock_sha256"]:
        raise ValueError("C06 parent v1 lock changed")
    parent_lock = read_json(parent_path)
    if parent_lock.get("files", {}).get(
        "configs/c06_real_mechanism_gate.yaml"
    ) != repair["parent_config_sha256"]:
        raise ValueError("C06 parent v1 config hash changed")
    if sha256_file(repair["parent_config_snapshot_path"]) != repair[
        "parent_config_sha256"
    ]:
        raise ValueError("byte-identical C06 parent config snapshot changed")
    if lock.get("parent_lock_sha256") != repair["parent_lock_sha256"]:
        raise ValueError("review1 lock names a different parent v1 lock")
    if lock.get("parent_config_sha256") != repair["parent_config_sha256"]:
        raise ValueError("review1 lock names a different parent v1 config")
    g0_path = Path(repair["g0_report_path"])
    if sha256_file(g0_path) != repair["g0_report_sha256"]:
        raise ValueError("C06 G0 changed before numeric repair")
    if lock.get("g0_report_sha256") != repair["g0_report_sha256"]:
        raise ValueError("review1 lock names a different G0")
    for prefix in ("centered_report", "centered_checkpoint"):
        artifact_path = Path(repair[f"{prefix}_path"])
        if sha256_file(artifact_path) != repair[f"{prefix}_sha256"]:
            raise ValueError(f"preserved C06 {prefix} changed")
        if lock.get(f"{prefix}_sha256") != repair[f"{prefix}_sha256"]:
            raise ValueError(f"review1 lock names a different {prefix}")

    evidence = lock.get("failed_variant_evidence", {})
    for variant in repair["eligible_variants"]:
        row = evidence.get(variant)
        expected_ledger_path = (
            Path(config["paths"]["artifact_root"])
            / f"formal_attempt_{variant}.json"
        )
        if not isinstance(row, dict) or Path(row.get("ledger_path", "")) != expected_ledger_path:
            raise ValueError(f"review1 names a different failure ledger: {variant}")
        if not isinstance(row, dict) or sha256_file(row["ledger_path"]) != row[
            "ledger_sha256"
        ]:
            raise ValueError(f"review1 failure ledger changed: {variant}")
        ledger = read_json(row["ledger_path"])
        attempts = ledger.get("attempts", [])
        if len(attempts) != 1 or attempts[0].get("stage") != "started":
            raise ValueError(f"review1 does not bind one failed v1 attempt: {variant}")
        if attempts[0].get("internal_A_features_scored") is not False or attempts[
            0
        ].get("internal_A_labels_opened") is not False:
            raise ValueError(f"failed v1 attempt touched A: {variant}")
        if attempts[0].get("internal_B_or_escrow_opened") is not False:
            raise ValueError(f"failed v1 attempt touched delayed evidence: {variant}")
        if attempts[0].get("real_gate_lock_sha256") != repair[
            "parent_lock_sha256"
        ] or attempts[0].get("config_sha256") != repair["parent_config_sha256"]:
            raise ValueError(f"failed v1 attempt provenance changed: {variant}")
    source_root = Path(config["paths"]["candidate_source_root"])
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("real-gate lock has no candidate-local file manifest")
    required_files = {
        "configs/c06_real_mechanism_gate.yaml",
        "configs/c06_real_mechanism_gate_parent_v1.yaml",
        "notes/real_gate_protocol.md",
        "notes/numeric_repair_review1.md",
        "model/wedge_flow.py",
        "model/controls.py",
        "model/complexity.py",
        "model/information_barrier.py",
        "model/transformer_core.py",
        "train/losses.py",
        "train/real_data.py",
        "train/materialize_real_g0.py",
        "train/real_gate_metrics.py",
        "train/run_real_gate.py",
    }
    if not required_files.issubset(files):
        raise ValueError(
            f"real-gate lock misses required files: {sorted(required_files - set(files))}"
        )
    for relative, expected in files.items():
        actual = sha256_file(source_root / relative)
        if actual != expected:
            raise ValueError(f"locked C06 file changed: {relative}")
    for relative, expected in lock.get("repo_files", {}).items():
        if sha256_file(relative) != expected:
            raise ValueError(f"locked shared file changed: {relative}")
    if "src/myrec/eval/metrics.py" not in lock.get("repo_files", {}):
        raise ValueError("real-gate lock does not bind the shared metric source")
    return sha256_file(path)


def validate_execution_authority(
    config: Mapping[str, Any],
    *,
    stage: str,
    device: str,
    variant: str | None = None,
) -> None:
    authorization = config.get("authorization", {})
    if not bool(authorization.get(stage, False)):
        raise PermissionError(f"C06 stage is not authorized: {stage}")
    resources = config.get("resources", {})
    if stage == "cohort_materialization":
        physical_gpu = resources.get("g0_physical_gpu")
    elif stage == "gpu_smoke":
        physical_gpu = resources.get("variant_physical_gpus", {}).get(variant)
    elif stage == "train_variants":
        physical_gpu = resources.get("variant_physical_gpus", {}).get(variant)
    elif stage == "a0_a1_audit":
        physical_gpu = resources.get("audit_physical_gpu")
    else:
        raise ValueError(f"unknown C06 execution stage: {stage}")
    environment = config.get("environment")
    run_prefix = str(config.get("run_prefix", ""))
    if not isinstance(physical_gpu, int) or not environment or environment == "UNASSIGNED":
        raise PermissionError("C06 physical GPU/environment remains unassigned")
    if not run_prefix.startswith("20260711_kuaisearch_c06_"):
        raise PermissionError("C06 run prefix remains unassigned")
    if device != "cuda:0":
        raise ValueError("a bound C06 process must address its only GPU as cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C06 execution requires exactly one visible CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the registered C06 GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUBLAS_WORKSPACE_CONFIG is required")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class StructuralTrainData:
    """Packed train arrays that cannot expose any candidate label."""

    root: Path
    request_ids: list[str]
    query_indices: np.ndarray
    timestamps: np.ndarray
    candidate_offsets: np.ndarray
    candidate_embedding_indices: np.ndarray
    candidate_item_ids: np.ndarray
    history_offsets: np.ndarray
    history_embedding_indices: np.ndarray
    history_event_weights: np.ndarray

    @classmethod
    def load(cls, packed_root: str | Path) -> "StructuralTrainData":
        root = Path(packed_root) / "train"
        request_ids: list[str] = []
        with (root / "request_ids.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    request_ids.append(str(json.loads(line)["request_id"]))
        value = cls(
            root=root,
            request_ids=request_ids,
            query_indices=np.load(root / "query_indices.npy", mmap_mode="r"),
            timestamps=np.load(root / "timestamps.npy", mmap_mode="r"),
            candidate_offsets=np.load(root / "candidate_offsets.npy", mmap_mode="r"),
            candidate_embedding_indices=np.load(
                root / "candidate_embedding_indices.npy", mmap_mode="r"
            ),
            candidate_item_ids=np.load(
                root / "candidate_item_ids.npy", mmap_mode="r"
            ),
            history_offsets=np.load(root / "history_offsets.npy", mmap_mode="r"),
            history_embedding_indices=np.load(
                root / "history_embedding_indices.npy", mmap_mode="r"
            ),
            history_event_weights=np.load(
                root / "history_event_weights.npy", mmap_mode="r"
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        requests = len(self.request_ids)
        if len(set(self.request_ids)) != requests:
            raise ValueError("packed train request IDs are not unique")
        if len(self.query_indices) != requests or len(self.timestamps) != requests:
            raise ValueError("request-shaped structural array mismatch")
        if len(self.candidate_offsets) != requests + 1:
            raise ValueError("candidate offset mismatch")
        if len(self.history_offsets) != requests + 1:
            raise ValueError("history offset mismatch")
        if int(self.candidate_offsets[-1]) != len(self.candidate_embedding_indices):
            raise ValueError("candidate embedding row mismatch")
        if int(self.candidate_offsets[-1]) != len(self.candidate_item_ids):
            raise ValueError("candidate identity row mismatch")
        if int(self.history_offsets[-1]) != len(self.history_embedding_indices):
            raise ValueError("history embedding row mismatch")
        if int(self.history_offsets[-1]) != len(self.history_event_weights):
            raise ValueError("history event row mismatch")

    def __len__(self) -> int:
        return len(self.request_ids)


def _stable_key(seed: int, role: str, request_id: str) -> str:
    payload = (
        b"c06-relative-v1\0"
        + role.encode("utf-8")
        + b"\0"
        + request_id.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _is_nonrepeat(data: StructuralTrainData, index: int) -> bool:
    hs = int(data.history_offsets[index])
    he = int(data.history_offsets[index + 1])
    if hs == he:
        return False
    cs = int(data.candidate_offsets[index])
    ce = int(data.candidate_offsets[index + 1])
    return not bool(
        np.intersect1d(
            np.asarray(data.candidate_embedding_indices[cs:ce]),
            np.asarray(data.history_embedding_indices[hs:he]),
            assume_unique=False,
        ).size
    )


def build_selection(
    data: StructuralTrainData,
    *,
    c05_request_ids: set[str],
    seed: int,
    cut: int,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    """Select all roles from structure, excluding C05 and prior C06 roles."""

    if not 0 < cut < len(data):
        raise ValueError("invalid frozen D2 cut")
    missing = set(REAL_GATE_COUNTS) - set(counts)
    if missing:
        raise ValueError(f"selection count is missing roles: {sorted(missing)}")
    pre_nonrepeat = [
        index
        for index in range(0, cut)
        if data.request_ids[index] not in c05_request_ids and _is_nonrepeat(data, index)
    ]
    post_nonrepeat = [
        index
        for index in range(cut, len(data))
        if data.request_ids[index] not in c05_request_ids and _is_nonrepeat(data, index)
    ]
    post_nohistory = [
        index
        for index in range(cut, len(data))
        if data.request_ids[index] not in c05_request_ids
        and int(data.history_offsets[index]) == int(data.history_offsets[index + 1])
    ]
    used: set[int] = set()

    def take(pool: Sequence[int], role: str) -> list[int]:
        available = [index for index in pool if index not in used]
        ranked = sorted(
            available,
            key=lambda index: (
                _stable_key(seed, role, data.request_ids[index]),
                data.request_ids[index],
            ),
        )
        count = int(counts[role])
        if len(ranked) < count:
            raise ValueError(f"not enough structurally eligible requests for {role}")
        chosen = sorted(ranked[:count])
        used.update(chosen)
        return chosen

    roles = {
        "fit": take(pre_nonrepeat, "fit"),
        "internal_A": take(post_nonrepeat, "internal_A"),
        "internal_B": take(post_nonrepeat, "internal_B"),
        "escrow": take(post_nonrepeat, "escrow"),
        "nohistory": take(post_nohistory, "nohistory"),
    }
    selected_ids = {
        role: [data.request_ids[index] for index in indices]
        for role, indices in roles.items()
    }
    flat_ids = [request_id for values in selected_ids.values() for request_id in values]
    if len(flat_ids) != len(set(flat_ids)):
        raise AssertionError("C06 selection roles overlap")
    if set(flat_ids) & c05_request_ids:
        raise AssertionError("C06 selection overlaps C05")
    return {
        "candidate_id": "c06",
        "selection_id": "c06_relative_real_gate_v1",
        "labels_opened_before_selection": False,
        "rule": "ascending sha256(c06-relative-v1\\0role\\0request_id), then packed order",
        "seed": int(seed),
        "cut": int(cut),
        "pool_counts": {
            "precut_nonrepeat_after_c05_exclusion": len(pre_nonrepeat),
            "postcut_nonrepeat_after_c05_exclusion": len(post_nonrepeat),
            "postcut_nohistory_after_c05_exclusion": len(post_nohistory),
            "c05_blacklist_requests": len(c05_request_ids),
        },
        "roles": {
            role: {"indices": roles[role], "request_ids": selected_ids[role]}
            for role in roles
        },
        "pairwise_disjoint": True,
        "zero_c05_overlap": True,
    }


def freeze_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Write the exact request IDs before any label-shaped array is opened."""

    c05_path = Path(config["paths"]["c05_selection"])
    actual_c05_hash = sha256_file(c05_path)
    if actual_c05_hash != str(config["paths"]["c05_selection_sha256"]):
        raise ValueError("registered C05 selection changed")
    c05 = read_json(c05_path)
    c05_request_ids: set[str] = set()
    for value in c05.values():
        if isinstance(value, dict) and isinstance(value.get("request_ids"), list):
            c05_request_ids.update(str(item) for item in value["request_ids"])
    if not c05_request_ids:
        raise ValueError("C05 selection blacklist is empty")
    expected_c05 = int(config["integrity"]["c05_selected_requests"])
    if len(c05_request_ids) != expected_c05:
        raise ValueError(
            f"C05 blacklist request count changed: {len(c05_request_ids)}"
        )
    data = StructuralTrainData.load(config["paths"]["packed_train_root"])
    if len(data) != int(config["integrity"]["packed_train_requests"]):
        raise ValueError("packed request count changed")
    counts = {
        role: int(config["selection"][f"{role}_requests"])
        for role in REAL_GATE_COUNTS
    }
    result = build_selection(
        data,
        c05_request_ids=c05_request_ids,
        seed=int(config["seed"]),
        cut=int(config["integrity"]["packed_cut_request_index"]),
        counts=counts,
    )
    result["packed_request_ids_path"] = str(data.root / "request_ids.jsonl")
    result["packed_request_ids_sha256"] = sha256_file(
        data.root / "request_ids.jsonl"
    )
    result["c05_selection_path"] = str(c05_path)
    result["c05_selection_sha256"] = actual_c05_hash
    path = Path(config["paths"]["artifact_root"]) / "selection.json"
    if path.exists():
        raise FileExistsError(f"immutable C06 selection already exists: {path}")
    write_json(path, result)
    result["path"] = str(path)
    result["sha256"] = sha256_file(path)
    return result


def selected_candidate_key_sha256(
    data: StructuralTrainData, indices: Sequence[int]
) -> str:
    digest = hashlib.sha256()
    for raw_index in indices:
        index = int(raw_index)
        start = int(data.candidate_offsets[index])
        stop = int(data.candidate_offsets[index + 1])
        payload = json.dumps(
            [
                data.request_ids[index],
                [str(value) for value in data.candidate_item_ids[start:stop]],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def iter_request_batches(
    data: StructuralTrainData,
    indices: Sequence[int] | np.ndarray,
    *,
    history_limit: int,
    max_requests: int,
    max_padded_candidates: int,
    max_padded_history: int,
    seed: int,
    shuffle: bool,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    batch: list[int] = []
    max_candidates = 0
    max_history = 0
    for raw_index in order:
        index = int(raw_index)
        candidate_count = int(
            data.candidate_offsets[index + 1] - data.candidate_offsets[index]
        )
        history_count = min(
            int(data.history_offsets[index + 1] - data.history_offsets[index]),
            history_limit,
        )
        next_size = len(batch) + 1
        if batch and (
            next_size > max_requests
            or next_size * max(max_candidates, candidate_count) > max_padded_candidates
            or next_size * max(max_history, history_count, 1) > max_padded_history
        ):
            yield np.asarray(batch, dtype=np.int64)
            batch = []
            max_candidates = 0
            max_history = 0
        batch.append(index)
        max_candidates = max(max_candidates, candidate_count)
        max_history = max(max_history, history_count, 1)
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def collate_structural(
    data: StructuralTrainData,
    request_indices: Sequence[int] | np.ndarray,
    *,
    history_limit: int,
) -> dict[str, np.ndarray]:
    """Collate full candidates and the latest history, without labels."""

    indices = np.asarray(request_indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("cannot collate an empty request batch")
    candidate_counts = [
        int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
        for index in indices
    ]
    history_counts = [
        min(
            int(data.history_offsets[index + 1] - data.history_offsets[index]),
            history_limit,
        )
        for index in indices
    ]
    batch = len(indices)
    max_candidates = max(candidate_counts)
    max_history = max(1, max(history_counts))
    candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_item_ids = np.full((batch, max_candidates), -1, dtype=np.int64)
    candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
    history_indices = np.zeros((batch, max_history), dtype=np.int64)
    history_mask = np.zeros((batch, max_history), dtype=bool)
    history_prior = np.zeros((batch, max_history), dtype=np.float32)
    for row, raw_index in enumerate(indices):
        index = int(raw_index)
        cs = int(data.candidate_offsets[index])
        ce = int(data.candidate_offsets[index + 1])
        candidate_count = ce - cs
        candidate_indices[row, :candidate_count] = data.candidate_embedding_indices[cs:ce]
        candidate_item_ids[row, :candidate_count] = data.candidate_item_ids[cs:ce]
        candidate_mask[row, :candidate_count] = True
        hs = int(data.history_offsets[index])
        he = int(data.history_offsets[index + 1])
        start = max(hs, he - history_limit)
        history_count = he - start
        if history_count:
            history_indices[row, :history_count] = data.history_embedding_indices[start:he]
            history_mask[row, :history_count] = True
            # The first C06 probe is architecture-only: it uses no handcrafted
            # action, category, dataset, or recency multiplier.
            history_prior[row, :history_count] = 1.0
    return {
        "request_indices": indices,
        "candidate_indices": candidate_indices,
        "candidate_item_ids": candidate_item_ids,
        "candidate_mask": candidate_mask,
        "history_indices": history_indices,
        "history_mask": history_mask,
        "history_prior": history_prior,
    }


@dataclass(frozen=True)
class SelectedLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("selected-label offset mismatch")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("selected-label row mismatch")

    @property
    def positions(self) -> dict[int, int]:
        return {
            int(request_index): position
            for position, request_index in enumerate(self.request_indices)
        }

    def padded(
        self, batch: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        request_indices = np.asarray(batch["request_indices"], dtype=np.int64)
        positions = self.positions
        output = np.zeros(batch["candidate_mask"].shape, dtype=np.float32)
        for row, raw_index in enumerate(request_indices):
            index = int(raw_index)
            if index not in positions:
                raise PermissionError(f"labels were not opened for request {index}")
            position = positions[index]
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            count = int(batch["candidate_mask"][row].sum())
            if stop - start != count:
                raise ValueError("selected-label candidate count changed")
            output[row, :count] = self.values[start:stop]
        return output


def open_selected_labels(
    data: StructuralTrainData,
    request_indices: Sequence[int],
    *,
    label_path: str | Path,
    allowed_indices: set[int],
) -> SelectedLabels:
    """Read only explicitly permitted request slices from the train labels."""

    indices = np.asarray(request_indices, dtype=np.int64)
    if not set(int(value) for value in indices).issubset(allowed_indices):
        raise PermissionError("attempted to open labels outside the permitted role")
    source = np.load(label_path, mmap_mode="r")
    if len(source) != int(data.candidate_offsets[-1]):
        raise ValueError("train label row count changed")
    rows: list[np.ndarray] = []
    offsets = [0]
    for raw_index in indices:
        index = int(raw_index)
        start = int(data.candidate_offsets[index])
        stop = int(data.candidate_offsets[index + 1])
        row = np.asarray(source[start:stop], dtype=np.float32).copy()
        if not np.isfinite(row).all():
            raise ValueError(f"non-finite label at request {index}")
        rows.append(row)
        offsets.append(offsets[-1] + len(row))
    values = np.concatenate(rows) if rows else np.empty(0, dtype=np.float32)
    return SelectedLabels(indices, np.asarray(offsets, dtype=np.int64), values)


def assert_internal_a_opening_barrier(
    a0_report_path: str | Path, *, expected_candidate_key_sha256: str
) -> str:
    """Require a durable, passing label-free A0 report before A labels."""

    path = Path(a0_report_path)
    report = read_json(path)
    if report.get("gate") != "G2_A0_label_free" or report.get("status") != "passed":
        raise PermissionError("internal-A labels require a passing A0 report")
    if report.get("internal_A_labels_opened") is not False:
        raise PermissionError("A0 report is not label-free")
    if report.get("internal_B_or_escrow_opened") is not False:
        raise PermissionError("A0 report touched a delayed role")
    if report.get("candidate_key_sha256") != expected_candidate_key_sha256:
        raise PermissionError("A0 candidate identities differ from frozen internal A")
    if not all(bool(value) for value in report.get("checks", {}).values()):
        raise PermissionError("A0 report contains a failed check")
    return sha256_file(path)


class FrozenRealFeatures:
    """Read-only D2p G0 coordinate for fit, A, and no-history roles."""

    def __init__(self, config: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
        root = Path(config["paths"]["artifact_root"])
        expected = np.asarray(
            [
                index
                for role in FEATURE_ROLES
                for index in selection["roles"][role]["indices"]
            ],
            dtype=np.int64,
        )
        self.selected_indices = np.load(root / "feature_request_indices.npy")
        if not np.array_equal(self.selected_indices, expected):
            raise ValueError("G0 feature request order differs from selection")
        self.positions = {
            int(index): position for position, index in enumerate(self.selected_indices)
        }
        self.query = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.items = np.load(root / "item_embeddings.npy", mmap_mode="r")
        self.item_indices = np.load(
            root / "item_embedding_indices.npy", mmap_mode="r"
        )
        self.base_offsets = np.load(root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        fit_labels = SelectedLabels(
            np.load(root / "fit_request_indices.npy"),
            np.load(root / "fit_label_offsets.npy"),
            np.load(root / "fit_labels.npy", mmap_mode="r"),
        )
        expected_fit = np.asarray(selection["roles"]["fit"]["indices"], dtype=np.int64)
        if not np.array_equal(fit_labels.request_indices, expected_fit):
            raise ValueError("G0 fit labels differ from selection")
        self.fit_labels = fit_labels
        if len(self.query) != len(self.selected_indices):
            raise ValueError("G0 query-state row mismatch")
        if len(self.items) != len(self.item_indices):
            raise ValueError("G0 item-state index mismatch")
        if len(self.item_indices) and not bool(
            np.all(np.diff(self.item_indices) > 0)
        ):
            raise ValueError("G0 item-state indices are not sorted unique")
        if len(self.base_offsets) != len(self.selected_indices) + 1:
            raise ValueError("G0 base-score offset mismatch")
        if int(self.base_offsets[-1]) != len(self.base_scores):
            raise ValueError("G0 base-score row mismatch")

    def tensors(
        self,
        batch: Mapping[str, np.ndarray],
        device: str,
        *,
        labels: SelectedLabels | None = None,
    ) -> dict[str, torch.Tensor]:
        indices = np.asarray(batch["request_indices"], dtype=np.int64)
        positions = np.asarray([self.positions[int(index)] for index in indices])
        query = np.asarray(self.query[positions], dtype=np.float32).copy()
        candidates = self._item_states(batch["candidate_indices"])
        history = self._item_states(batch["history_indices"])
        base = np.zeros(batch["candidate_mask"].shape, dtype=np.float32)
        for row, position in enumerate(positions):
            start = int(self.base_offsets[position])
            stop = int(self.base_offsets[position + 1])
            count = int(batch["candidate_mask"][row].sum())
            if stop - start != count:
                raise ValueError("G0 candidate set changed")
            base[row, :count] = self.base_scores[start:stop]
        result = {
            "query": torch.from_numpy(query).to(device),
            "candidates": torch.from_numpy(candidates).to(device),
            "history": torch.from_numpy(history).to(device),
            "candidate_mask": torch.from_numpy(batch["candidate_mask"]).to(device),
            "history_mask": torch.from_numpy(batch["history_mask"]).to(device),
            "history_prior": torch.from_numpy(batch["history_prior"]).to(device),
            "base_scores": torch.from_numpy(base).to(device),
        }
        if labels is not None:
            result["labels"] = torch.from_numpy(labels.padded(batch)).to(device)
        return result

    def _item_states(self, raw_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(raw_indices, dtype=np.int64)
        positions = np.searchsorted(self.item_indices, indices)
        if bool((positions >= len(self.item_indices)).any()):
            raise ValueError("batch references an unmaterialized G0 item state")
        if not np.array_equal(
            np.asarray(self.item_indices[positions], dtype=np.int64), indices
        ):
            raise ValueError("batch references an unmaterialized G0 item state")
        return np.asarray(self.items[positions], dtype=np.float32).copy()
