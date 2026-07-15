"""Ordinary full-token cross-encoder counterfactual scorer for exploration."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_full_token_cross_encoder_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    history_condition: str,
    history_assignments_path: str | Path,
    *,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    device: str = "cuda:0",
    dtype: str = "float16",
    batch_size: int = 32,
    max_length: int = 512,
    history_budget: int = 10,
    pair_chunk_size: int = 2048,
    serialization_version: str = "query_history_event_text_v1",
    checkpoint_id: str | None = None,
    local_files_only: bool = False,
    request_aligned_batches: bool = False,
    method_id: str = "e_full_zero_shot",
    tuning_class: str = "zero_shot_instrumentation",
    truncation_strategy: str = "longest_first",
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Score one true/null/wrong assignment with one unchanged checkpoint."""

    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    if history_budget < 0:
        raise ValueError("history_budget must be non-negative")
    if truncation_strategy not in {"longest_first", "only_second"}:
        raise ValueError(f"unsupported truncation_strategy={truncation_strategy}")
    standardized_dir = Path(standardized_dir)
    history_assignments_path = Path(history_assignments_path)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    assignments = _load_assignments(
        history_assignments_path,
        expected_condition=history_condition,
    )
    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)

    package_versions: dict[str, str] = {}
    if predictor is None:
        import sentence_transformers
        import torch
        from sentence_transformers import CrossEncoder

        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype={dtype}")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        predictor = CrossEncoder(
            model_name,
            cache_folder=str(cache_folder),
            device=device,
            max_length=max_length,
            trust_remote_code=True,
            local_files_only=local_files_only,
            model_kwargs={"dtype": torch_dtype},
        )
        package_versions = {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        }
        activation_fn = torch.nn.Identity()
    else:
        activation_fn = None

    resolved_checkpoint = checkpoint_id or _resolved_checkpoint_id(
        predictor, model_name=model_name
    )
    scoring_signature = {
        "serialization_version": serialization_version,
        "max_length": max_length,
        "history_budget": history_budget,
        "candidate_scoring_head": "sequence_classification_raw_logit",
        "dtype": dtype,
        "model_name": model_name,
        "request_aligned_batches": request_aligned_batches,
        "truncation_strategy": truncation_strategy,
    }
    scores_path = run_dir / "scores.jsonl"
    pair_buffer: list[tuple[str, str]] = []
    key_buffer: list[tuple[str, str]] = []
    rows = 0
    requests = 0
    seen_request_ids: set[str] = set()
    started = time.perf_counter()

    def flush(handle: Any) -> None:
        nonlocal rows
        if not pair_buffer:
            return
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if activation_fn is not None:
            kwargs["activation_fn"] = activation_fn
        kwargs["processing_kwargs"] = {
            "text": {"truncation": truncation_strategy}
        }
        predicted = predictor.predict(pair_buffer, **kwargs)
        flattened = _flatten_predictions(predicted)
        if len(flattened) != len(key_buffer):
            raise ValueError(
                f"prediction count mismatch: {len(flattened)} != {len(key_buffer)}"
            )
        for (request_id, item_id), score in zip(key_buffer, flattened):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score for {request_id} {item_id}: {value}")
            handle.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "candidate_item_id": item_id,
                        "score": value,
                        "method_id": method_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            rows += 1
        pair_buffer.clear()
        key_buffer.clear()

    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            request_id = str(record["request_id"])
            if request_id not in assignments:
                raise ValueError(f"assignment missing request_id={request_id}")
            seen_request_ids.add(request_id)
            requests += 1
            history = assignments[request_id]["history"]
            query_context = serialize_query_history(
                str(record["query"]),
                history,
                history_budget=history_budget,
                serialization_version=serialization_version,
            )
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                pair_buffer.append((query_context, document_text(candidate)))
                key_buffer.append((request_id, item_id))
                if len(pair_buffer) >= pair_chunk_size:
                    flush(handle)
            if request_aligned_batches:
                # Dynamic padding can otherwise make identical no-history inputs
                # differ across counterfactual runs because neighboring requests
                # have different sequence lengths. Keep each request's slate as
                # the invariant inference unit.
                flush(handle)
        flush(handle)
    extra_assignments = set(assignments) - seen_request_ids
    if extra_assignments:
        raise ValueError(f"assignments contain unknown request_ids: {sorted(extra_assignments)[:5]}")

    elapsed = time.perf_counter() - started
    metadata = {
        "batch_size": batch_size,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "checkpoint_id": resolved_checkpoint,
        "config_path": str(config_path) if config_path else None,
        "dataset_id": str(dataset_manifest["dataset_id"]),
        "dataset_version": str(dataset_manifest["dataset_version"]),
        "device": device,
        "elapsed_seconds": elapsed,
        "history_assignment_sha256": sha256_file(history_assignments_path),
        "history_assignments_path": str(history_assignments_path),
        "history_condition": history_condition,
        "local_files_only": local_files_only,
        "request_aligned_batches": request_aligned_batches,
        "input_fields_used": _input_fields_used(serialization_version),
        "latency": {
            "candidate_pairs_per_second": rows / elapsed if elapsed else None,
            "seconds_total": elapsed,
        },
        "method_id": method_id,
        "model_name": model_name,
        "package_versions": package_versions,
        "qrels_read": False,
        "request_count": requests,
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "run_id": run_id,
        "score_definition": (
            "ordinary cross-encoder raw sequence-classification logit over "
            "(serialized query+history, candidate text)"
        ),
        "score_rows": rows,
        "scoring_signature": scoring_signature,
        "split": split,
        "standardized_dir": str(standardized_dir),
        "tuning": {
            "class": tuning_class,
            "adequate_model_family": False,
            "dev_labels_used_during_scoring": False,
        },
    }
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _input_fields_used(serialization_version: str) -> list[str]:
    fields = ["query"]
    if serialization_version == "query_history_event_text_v1":
        fields.extend(
            [
                "assigned_history.query",
                "assigned_history.title",
                "assigned_history.brand",
                "assigned_history.cat",
                "assigned_history.event",
            ]
        )
    elif serialization_version != "query_only_text_v1":
        raise ValueError(
            f"unsupported serialization_version={serialization_version}"
        )
    fields.extend(
        [
            "candidates.title",
            "candidates.brand",
            "candidates.cat",
        ]
    )
    return fields


def serialize_query_history(
    query: str,
    history: list[dict[str, Any]],
    *,
    history_budget: int,
    serialization_version: str,
) -> str:
    if serialization_version == "query_only_text_v1":
        if history:
            raise ValueError("query_only_text_v1 cannot serialize non-empty history")
        return query
    if serialization_version != "query_history_event_text_v1":
        raise ValueError(f"unsupported serialization_version={serialization_version}")
    selected = history[-history_budget:] if history_budget else []
    if selected:
        history_rows = []
        for index, event in enumerate(selected, start=1):
            categories = "/".join(str(value) for value in event.get("cat", []) if value)
            fields = [
                str(event.get("event") or "interaction"),
                (
                    f"prior query: {event['query']}"
                    if str(event.get("query") or "").strip()
                    else ""
                ),
                str(event.get("title") or ""),
                str(event.get("brand") or ""),
                categories,
            ]
            history_rows.append(
                f"{index}. " + " | ".join(value for value in fields if value)
            )
        history_text = "\n".join(history_rows)
    else:
        history_text = "(empty)"
    return f"[QUERY]\n{query}\n[PRIOR USER HISTORY]\n{history_text}"


def _load_assignments(
    path: Path,
    *,
    expected_condition: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate assignment request_id={request_id}")
        if str(row.get("assignment")) != expected_condition:
            raise ValueError(
                f"assignment condition mismatch for request_id={request_id}"
            )
        result[request_id] = row
    if not result:
        raise ValueError(f"empty history assignment file: {path}")
    return result


def _resolved_checkpoint_id(predictor: Any, *, model_name: str) -> str:
    model = getattr(predictor, "model", None)
    config = getattr(model, "config", None)
    commit = getattr(config, "_commit_hash", None)
    return f"{model_name}@{commit or 'unresolved_zero_shot_pilot'}"


def _flatten_predictions(predicted: Any) -> list[float]:
    if hasattr(predicted, "detach"):
        predicted = predicted.detach().cpu().numpy()
    if hasattr(predicted, "reshape"):
        return [float(value) for value in predicted.reshape(-1)]
    result = []
    for value in predicted:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("multi-logit cross-encoder output is unsupported")
            value = value[0]
        result.append(float(value))
    return result
