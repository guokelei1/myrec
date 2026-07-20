#!/usr/bin/env python3
"""Audit frozen embedding/readout update geometry without labels or scores.

This descriptive D0/D6 bridge covers every vocabulary row and the exact token
roles used by the frozen 512-row candidate sample.  It does not select tokens,
read qrels/model scores, or perform an output-row intervention.  Q2 uses tied
input/output rows, so row changes are not interpreted causally; Q3's adapter is
audited to contain no embedding/readout parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from safetensors import safe_open

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import load_v12_ranker_config
from myrec.mechanism.attention_observation_runtime import _build_observation_paths
from myrec.utils.jsonl import iter_jsonl


DEEP_DIVE_PLAN = Path("experiments/motivation/transformer_deep_dive_plan.md")
DEEP_DIVE_PLAN_SHA256 = (
    "07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584"
)
DEEP_DIVE_MANIFEST = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
DEEP_DIVE_MANIFEST_SHA256 = (
    "76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758"
)
SAMPLE_MANIFEST = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/manifest.json"
)
SAMPLE_MANIFEST_SHA256 = (
    "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
)
BASE_WEIGHTS = Path("models/huggingface/Qwen3-0.6B/model.safetensors")
BASE_WEIGHTS_SHA256 = (
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
)
Q2_MODEL_DIR = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/checkpoint_latest/model"
)
Q2_WEIGHTS = Q2_MODEL_DIR / "model.safetensors"
Q2_WEIGHTS_SHA256 = (
    "83e3467dc26a02e65a0a49efabf08273ddb6dc7bcea7b06fe5bb0aaf2825f7c9"
)
TOKENIZER_JSON_SHA256 = (
    "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
)
TOKENIZER_CONFIG_SHA256 = (
    "84d5f3d5b23156305adcc3b446a09ee865766d87228af44bb3303042c68c9d6a"
)
Q3_ADAPTER = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q3_tallrec_generalqwen_seed20260714/checkpoint_latest/model/"
    "adapter_model.safetensors"
)
Q3_ADAPTER_SHA256 = (
    "fd51a9c6b9ee3a6651597c263a8120db52cb79d62e7c80e544666e46bc5e1cef"
)
CONFIGS = {
    "q2": Path(
        "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
    ),
    "q3": Path(
        "configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
    ),
}
REGISTERED_TOKEN_IDS = {
    "q2_yes": 9693,
    "q2_no": 2152,
    "q3_yes": 9454,
    "q3_no": 2753,
    "endoftext_content_neutral": 151643,
    "im_end": 151645,
}
ROLE_NAMES = ("query", "history", "candidate", "native_target", "structural")
EMBEDDING_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d0_embedding_readout_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected, label in (
        (DEEP_DIVE_PLAN, DEEP_DIVE_PLAN_SHA256, "deep-dive plan"),
        (DEEP_DIVE_MANIFEST, DEEP_DIVE_MANIFEST_SHA256, "deep-dive manifest"),
        (SAMPLE_MANIFEST, SAMPLE_MANIFEST_SHA256, "fixed sample manifest"),
        (BASE_WEIGHTS, BASE_WEIGHTS_SHA256, "base weights"),
        (Q2_WEIGHTS, Q2_WEIGHTS_SHA256, "Q2 final weights"),
        (Q2_MODEL_DIR / "tokenizer.json", TOKENIZER_JSON_SHA256, "tokenizer"),
        (
            Q2_MODEL_DIR / "tokenizer_config.json",
            TOKENIZER_CONFIG_SHA256,
            "tokenizer config",
        ),
        (Q3_ADAPTER, Q3_ADAPTER_SHA256, "Q3 final adapter"),
    ):
        absolute = root / path
        if _sha256_file(absolute) != expected:
            raise ValueError(f"{label} hash drift")

    manifest = yaml.safe_load((root / DEEP_DIVE_MANIFEST).read_text(encoding="utf-8"))
    sample_manifest = _read_json(root / SAMPLE_MANIFEST)
    sample_path = root / str(sample_manifest["path"])
    if (
        sample_manifest.get("status") != "frozen_qrels_blind_sample"
        or sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or sample_manifest.get("source_test_opened") is not False
        or _sha256_file(sample_path) != sample_manifest.get("sha256")
    ):
        raise ValueError("embedding fixed sample boundary differs")
    records_path = root / str(sample_manifest["target_records_path"])
    if _sha256_file(records_path) != sample_manifest["target_records_sha256"]:
        raise ValueError("embedding records hash drift")
    records = {
        str(row["request_id"]): sanitize_record_for_model(row)
        for row in iter_jsonl(records_path)
    }
    if len(records) != 8000:
        raise ValueError("embedding audit requires all frozen dev request identities")
    samples = list(iter_jsonl(sample_path))
    if len(samples) != 512:
        raise ValueError("embedding sample row count drift")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        root / Q2_MODEL_DIR, local_files_only=True
    )
    role_counts: dict[str, dict[str, Counter[int]]] = {}
    role_audits = {}
    for model_key in ("q2", "q3"):
        config = load_v12_ranker_config(root / CONFIGS[model_key])
        method_id = str(config["method_id"])
        frozen_model = manifest["frozen_inputs"]["models"][method_id]
        if config["_config_sha256"] != frozen_model["config_sha256"]:
            raise ValueError(f"{model_key} config differs from deep-dive manifest")
        counts, audit = _collect_role_counts(
            tokenizer, records, samples, config
        )
        role_counts[model_key] = counts
        role_audits[model_key] = {
            **audit,
            "method_id": method_id,
            "config_path": CONFIGS[model_key].as_posix(),
            "config_sha256": config["_config_sha256"],
        }

    base_embedding, base_tied = _load_base_embedding(root / BASE_WEIGHTS)
    q2_embedding = _load_embedding(root / Q2_WEIGHTS)
    if base_embedding.shape != q2_embedding.shape:
        raise ValueError("Q2/base embedding shapes differ")
    if (
        len(tokenizer) > int(base_embedding.shape[0])
        or max(REGISTERED_TOKEN_IDS.values()) >= int(base_embedding.shape[0])
    ):
        raise ValueError("tokenizer/token IDs exceed padded embedding vocabulary")
    metrics = _row_update_metrics(base_embedding, q2_embedding)
    vocab_summary = _vocabulary_summary(metrics)
    role_geometry = {
        role: _role_summary(counts, metrics)
        for role, counts in role_counts["q2"].items()
    }
    registered_rows = {
        name: _registered_row_summary(token_id, metrics)
        for name, token_id in REGISTERED_TOKEN_IDS.items()
    }
    q2_readout = _direction_metrics(
        base_embedding,
        q2_embedding,
        yes_token_id=REGISTERED_TOKEN_IDS["q2_yes"],
        no_token_id=REGISTERED_TOKEN_IDS["q2_no"],
        update_rms_distribution=metrics["update_rms"],
    )
    q3_adapter = _audit_q3_adapter(root / Q3_ADAPTER)

    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d0_embedding_readout_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_token_or_row_selector": False,
        "qrels_read": False,
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": "scripts/analyze_deep_dive_embedding_readout.py",
        "deep_dive_plan_sha256": DEEP_DIVE_PLAN_SHA256,
        "deep_dive_manifest_sha256": DEEP_DIVE_MANIFEST_SHA256,
        "sample": {
            "manifest_path": SAMPLE_MANIFEST.as_posix(),
            "manifest_sha256": SAMPLE_MANIFEST_SHA256,
            "rows_path": str(sample_manifest["path"]),
            "rows_sha256": str(sample_manifest["sha256"]),
            "selected_candidate_rows": 512,
            "records_path": str(sample_manifest["target_records_path"]),
            "records_sha256": str(sample_manifest["target_records_sha256"]),
        },
        "weights": {
            "base_path": BASE_WEIGHTS.as_posix(),
            "base_sha256": BASE_WEIGHTS_SHA256,
            "q2_final_path": Q2_WEIGHTS.as_posix(),
            "q2_final_sha256": Q2_WEIGHTS_SHA256,
            "q3_adapter_path": Q3_ADAPTER.as_posix(),
            "q3_adapter_sha256": Q3_ADAPTER_SHA256,
            "embedding_key": EMBEDDING_KEY,
            "base_lm_head_key": LM_HEAD_KEY,
            "base_embedding_lm_head_exactly_equal": base_tied,
            "q2_checkpoint_uses_tied_embedding_without_separate_lm_head": True,
        },
        "role_token_usage": {
            model_key: {
                role: {
                    "occurrences": int(sum(counts.values())),
                    "unique_token_ids": len(counts),
                }
                for role, counts in model_counts.items()
            }
            for model_key, model_counts in role_counts.items()
        },
        "role_audits": role_audits,
        "q2_vocabulary_update": vocab_summary,
        "q2_role_update_geometry": role_geometry,
        "registered_token_rows_in_q2_checkpoint": registered_rows,
        "q2_yes_no_readout_direction": q2_readout,
        "q3_embedding_readout_update": q3_adapter,
        "interpretation_boundary": (
            "Q2 input embeddings and output rows are tied, so row geometry cannot "
            "separate prompt-input from output-readout causality. Role token sets are "
            "fixed and reported in aggregate; no token is selected for intervention. "
            "Q3 uses the frozen base embedding/readout and trains only q/v LoRA A/B."
        ),
    }
    result["q2_vocabulary_update"].update(
        {
            "tokenizer_assigned_rows": len(tokenizer),
            "padded_unassigned_rows": int(base_embedding.shape[0]) - len(tokenizer),
        }
    )
    del base_embedding, q2_embedding
    output_path = output_dir / "embedding_readout.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "sha256": _sha256_file(output_path),
                "vocabulary_rows": vocab_summary["vocabulary_rows"],
                "sample_rows": 512,
            },
            sort_keys=True,
        )
    )


def _collect_role_counts(
    tokenizer: Any,
    records: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Counter[int]], dict[str, Any]]:
    counts = {role: Counter() for role in ROLE_NAMES}
    request_ids = []
    for sample in samples:
        request_id = str(sample["request_id"])
        record = records.get(request_id)
        if record is None:
            raise ValueError("embedding sample request is absent from frozen records")
        ordinal = int(sample["candidate_ordinal"])
        if (
            not 0 <= ordinal < len(record.candidates)
            or str(record.candidates[ordinal]["item_id"])
            != str(sample["candidate_item_id"])
        ):
            raise ValueError("embedding sample candidate identity drift")
        paths = _build_observation_paths(
            tokenizer, record, ordinal, config, device="cpu"
        )
        path = paths[0]["full"]
        selected = int(paths[0]["selected_batch_row"])
        ids = [int(value) for value in path["ids"][selected].tolist()]
        mask = [int(value) for value in path["mask"][selected].tolist()]
        occupied: set[int] = set()
        for role in ("query", "history", "candidate"):
            starts, ends = path["spans"][role]
            start, end = int(starts[selected]), int(ends[selected])
            if not 0 <= start < end <= len(ids):
                raise ValueError(f"embedding semantic span is invalid: {role}")
            counts[role].update(ids[start:end])
            occupied.update(range(start, end))
        first_target = [int(value) for value in path["target"]]
        if first_target:
            # With left padding, the target is the final non-padding suffix.
            target_start = (
                max(index for index, value in enumerate(mask) if value)
                + 1
                - len(first_target)
            )
            if ids[target_start : target_start + len(first_target)] != first_target:
                raise ValueError("embedding native target token suffix drift")
            occupied.update(range(target_start, target_start + len(first_target)))
        structural = [
            token_id
            for position, (token_id, active) in enumerate(zip(ids, mask))
            if active and position not in occupied
        ]
        counts["structural"].update(structural)
        for pair in paths:
            target_path = pair["full"]
            target_selected = int(pair["selected_batch_row"])
            target = [int(value) for value in target_path["target"]]
            if not target:
                continue
            target_ids = [
                int(value) for value in target_path["ids"][target_selected].tolist()
            ]
            target_mask = [
                int(value) for value in target_path["mask"][target_selected].tolist()
            ]
            target_start = (
                max(index for index, value in enumerate(target_mask) if value)
                + 1
                - len(target)
            )
            if target_ids[target_start : target_start + len(target)] != target:
                raise ValueError("embedding native target path suffix drift")
            counts["native_target"].update(target)
        request_ids.append(request_id)
    return counts, {
        "sample_rows": len(samples),
        "unique_requests": len(set(request_ids)),
        "request_sequence_sha256": _canonical_sha256(request_ids),
        "roles_are_occurrence_disjoint": True,
        "role_token_ids_may_overlap_across_roles": True,
    }


def _load_base_embedding(path: Path) -> tuple[Any, bool]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if EMBEDDING_KEY not in keys or LM_HEAD_KEY not in keys:
            raise ValueError("base checkpoint lacks embedding/tied readout rows")
        embedding = handle.get_tensor(EMBEDDING_KEY)
        lm_head = handle.get_tensor(LM_HEAD_KEY)
    tied = bool(embedding.shape == lm_head.shape and (embedding == lm_head).all().item())
    if not tied:
        raise ValueError("base Qwen embedding and lm-head are not exactly tied")
    return embedding, tied


def _load_embedding(path: Path) -> Any:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if EMBEDDING_KEY not in keys or LM_HEAD_KEY in keys:
            raise ValueError("Q2 checkpoint tied embedding key coverage differs")
        return handle.get_tensor(EMBEDDING_KEY)


def _row_update_metrics(base: Any, final: Any, chunk_size: int = 4096) -> dict[str, Any]:
    if base.ndim != 2 or final.shape != base.shape:
        raise ValueError("embedding row matrices differ")
    rows, hidden = map(int, base.shape)
    base_l2 = np.empty(rows, dtype=np.float64)
    final_l2 = np.empty(rows, dtype=np.float64)
    update_l2 = np.empty(rows, dtype=np.float64)
    cosine = np.empty(rows, dtype=np.float64)
    exact = np.empty(rows, dtype=np.bool_)
    for start in range(0, rows, int(chunk_size)):
        end = min(rows, start + int(chunk_size))
        left = base[start:end].double()
        right = final[start:end].double()
        delta = right - left
        left_sq = left.square().sum(dim=1).numpy()
        right_sq = right.square().sum(dim=1).numpy()
        delta_sq = delta.square().sum(dim=1).numpy()
        dot = (left * right).sum(dim=1).numpy()
        base_l2[start:end] = np.sqrt(left_sq)
        final_l2[start:end] = np.sqrt(right_sq)
        update_l2[start:end] = np.sqrt(delta_sq)
        cosine[start:end] = np.divide(
            dot,
            np.sqrt(left_sq * right_sq),
            out=np.zeros_like(dot),
            where=(left_sq > 0) & (right_sq > 0),
        )
        exact[start:end] = (base[start:end] == final[start:end]).all(dim=1).numpy()
    if not all(np.isfinite(value).all() for value in (base_l2, final_l2, update_l2, cosine)):
        raise FloatingPointError("embedding row geometry is non-finite")
    return {
        "vocabulary_rows": rows,
        "hidden_size": hidden,
        "base_l2": base_l2,
        "final_l2": final_l2,
        "update_l2": update_l2,
        "update_rms": update_l2 / math.sqrt(hidden),
        "relative_update_l2": np.divide(
            update_l2,
            base_l2,
            out=np.zeros_like(update_l2),
            where=base_l2 > 0,
        ),
        "base_final_cosine": cosine,
        "exactly_unchanged": exact,
    }


def _vocabulary_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    update_rms = np.asarray(metrics["update_rms"], dtype=np.float64)
    update_energy = np.square(np.asarray(metrics["update_l2"], dtype=np.float64))
    total = float(update_energy.sum())
    participation = (
        float(total * total / (len(update_energy) * np.square(update_energy).sum()))
        if total > 0
        else 0.0
    )
    top = max(1, math.ceil(0.1 * len(update_energy)))
    return {
        "vocabulary_rows": int(metrics["vocabulary_rows"]),
        "hidden_size": int(metrics["hidden_size"]),
        "exactly_unchanged_rows": int(np.asarray(metrics["exactly_unchanged"]).sum()),
        "update_row_rms": _array_summary(update_rms),
        "relative_update_l2": _array_summary(metrics["relative_update_l2"]),
        "base_final_row_cosine": _array_summary(metrics["base_final_cosine"]),
        "normalized_row_participation_ratio": participation,
        "top_10pct_row_update_energy_share": (
            float(np.partition(update_energy, -top)[-top:].sum() / total)
            if total > 0
            else 0.0
        ),
    }


def _role_summary(counts: Mapping[int, int], metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not counts:
        return {
            "occurrences": 0,
            "unique_token_ids": 0,
            "occurrence_weighted_update_rms": None,
            "occurrence_weighted_mean_row_update_rms": None,
            "unique_token_update_rms": None,
            "occurrence_weighted_relative_update_l2": None,
            "occurrence_weighted_base_final_cosine": None,
        }
    ids = np.asarray(sorted(int(value) for value in counts), dtype=np.int64)
    if ids.min() < 0 or ids.max() >= int(metrics["vocabulary_rows"]):
        raise ValueError("role token ID lies outside embedding vocabulary")
    weights = np.asarray([counts[int(value)] for value in ids], dtype=np.float64)
    update_rms = np.asarray(metrics["update_rms"])[ids]
    update_l2 = np.asarray(metrics["update_l2"])[ids]
    total_occurrences = float(weights.sum())
    hidden = int(metrics["hidden_size"])
    return {
        "occurrences": int(total_occurrences),
        "unique_token_ids": len(ids),
        "occurrence_weighted_update_rms": float(
            math.sqrt(np.dot(weights, np.square(update_l2)) / (total_occurrences * hidden))
        ),
        "occurrence_weighted_mean_row_update_rms": float(
            np.dot(weights, update_rms) / total_occurrences
        ),
        "unique_token_update_rms": _array_summary(update_rms),
        "occurrence_weighted_relative_update_l2": float(
            np.dot(weights, np.asarray(metrics["relative_update_l2"])[ids])
            / total_occurrences
        ),
        "occurrence_weighted_base_final_cosine": float(
            np.dot(weights, np.asarray(metrics["base_final_cosine"])[ids])
            / total_occurrences
        ),
        "exactly_unchanged_occurrence_fraction": float(
            np.dot(weights, np.asarray(metrics["exactly_unchanged"])[ids])
            / total_occurrences
        ),
    }


def _registered_row_summary(token_id: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    token_id = int(token_id)
    update_rms = float(metrics["update_rms"][token_id])
    distribution = np.asarray(metrics["update_rms"], dtype=np.float64)
    return {
        "token_id": token_id,
        "base_l2": float(metrics["base_l2"][token_id]),
        "final_l2": float(metrics["final_l2"][token_id]),
        "update_l2": float(metrics["update_l2"][token_id]),
        "update_rms": update_rms,
        "update_rms_empirical_cdf": float(np.mean(distribution <= update_rms)),
        "relative_update_l2": float(metrics["relative_update_l2"][token_id]),
        "base_final_cosine": float(metrics["base_final_cosine"][token_id]),
        "exactly_unchanged": bool(metrics["exactly_unchanged"][token_id]),
    }


def _direction_metrics(
    base: Any,
    final: Any,
    *,
    yes_token_id: int,
    no_token_id: int,
    update_rms_distribution: np.ndarray,
) -> dict[str, Any]:
    base_yes = base[int(yes_token_id)].double().numpy()
    base_no = base[int(no_token_id)].double().numpy()
    final_yes = final[int(yes_token_id)].double().numpy()
    final_no = final[int(no_token_id)].double().numpy()
    base_direction = base_yes - base_no
    final_direction = final_yes - final_no
    direction_update = final_direction - base_direction
    base_common = 0.5 * (base_yes + base_no)
    final_common = 0.5 * (final_yes + final_no)
    common_update = final_common - base_common
    return {
        "yes_token_id": int(yes_token_id),
        "no_token_id": int(no_token_id),
        "base_direction_l2": float(np.linalg.norm(base_direction)),
        "final_direction_l2": float(np.linalg.norm(final_direction)),
        "base_final_direction_cosine": _cosine(base_direction, final_direction),
        "direction_update_l2": float(np.linalg.norm(direction_update)),
        "direction_update_relative_to_base": float(
            np.linalg.norm(direction_update) / np.linalg.norm(base_direction)
        ),
        "base_common_l2": float(np.linalg.norm(base_common)),
        "final_common_l2": float(np.linalg.norm(final_common)),
        "base_final_common_cosine": _cosine(base_common, final_common),
        "common_update_l2": float(np.linalg.norm(common_update)),
        "direction_common_update_cosine": _cosine(direction_update, common_update),
        "yes_update_rms_empirical_cdf": float(
            np.mean(update_rms_distribution <= np.sqrt(np.mean(np.square(final_yes - base_yes))))
        ),
        "no_update_rms_empirical_cdf": float(
            np.mean(update_rms_distribution <= np.sqrt(np.mean(np.square(final_no - base_no))))
        ),
    }


def _audit_q3_adapter(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    invalid = [
        key
        for key in keys
        if ".lora_A." not in key and ".lora_B." not in key
    ]
    embedding = [
        key for key in keys if "embed_tokens" in key or "lm_head" in key
    ]
    if len(keys) != 112 or invalid or embedding:
        raise ValueError("Q3 adapter parameter coverage differs")
    return {
        "adapter_parameter_objects": len(keys),
        "embedding_or_lm_head_parameter_objects": len(embedding),
        "base_embedding_readout_frozen": True,
        "trained_parameter_scope": "28 blocks x q/v projections x LoRA A/B",
    }


def _array_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("embedding summary array differs")
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(quantiles[7]),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
