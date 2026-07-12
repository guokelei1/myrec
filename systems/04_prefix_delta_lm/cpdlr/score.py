"""Label-free C04 dev scoring and paired-prefix structural diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .io import (
    assert_candidate_manifest,
    assert_label_free_record,
    finite_float,
    iter_jsonl,
    sha256_file,
    write_json,
)
from .model import PrefixDeltaRanker
from .tokenization import PrefixTokenizer


DIAGNOSTIC_CORRUPTIONS = ("wrong", "shuffled", "query_masked", "coarse")


def _load_train_history_donors(path: str | Path) -> list[dict[str, Any]]:
    donors = [
        {
            "history": list(row["history"]),
            "request_id": str(row["request_id"]),
            "user_id": str(row["user_id"]),
        }
        for row in iter_jsonl(path)
        if row.get("history")
    ]
    if not donors:
        raise ValueError("C04 wrong-history diagnostic has no train-only donors")
    return donors


def _attach_wrong_history(
    record: dict[str, Any], donors: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    from .io import stable_hash

    copied = dict(record)
    start = int(stable_hash("c04_dev_wrong", seed, record["request_id"])[:12], 16) % len(
        donors
    )
    donor = None
    for offset in range(len(donors)):
        candidate = donors[(start + offset) % len(donors)]
        if str(candidate["user_id"]) != str(record.get("user_id", "")):
            donor = candidate
            break
    if donor is None:
        raise ValueError(f"no different-user train donor for {record['request_id']}")
    copied["wrong_history"] = donor["history"]
    return copied


def _tokenizer(config: dict[str, Any]) -> PrefixTokenizer:
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["backbone"],
        local_files_only=bool(model_cfg["local_files_only"]),
        use_fast=False,
    )
    return PrefixTokenizer(
        tokenizer,
        max_length=int(model_cfg["max_length"]),
        query_tokens=int(model_cfg["query_tokens"]),
        candidate_tokens=int(model_cfg["candidate_tokens"]),
        max_history_events=int(model_cfg["max_history_events"]),
        event_tokens=int(model_cfg["event_tokens"]),
    )


def _move(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def _branch_logits(
    model: PrefixDeltaRanker,
    tokenizer: PrefixTokenizer,
    record: dict[str, Any],
    prefix: str,
    config: dict[str, Any],
    device: str,
    structured: bool,
) -> torch.Tensor:
    values = []
    candidates = list(record["candidates"])
    batch_size = int(config["scoring"]["candidate_batch_size"])
    for start in range(0, len(candidates), batch_size):
        pairs = [(record, candidate) for candidate in candidates[start : start + batch_size]]
        encoded = tokenizer.batch_encode(
            pairs,
            prefix,
            int(config["seed"]),
            structured=structured,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ):
            logits = model.score(_move(encoded, device))
        values.append(logits.float())
    return torch.cat(values)


def _paired_branch_logits(
    model: PrefixDeltaRanker,
    tokenizer: PrefixTokenizer,
    record: dict[str, Any],
    config: dict[str, Any],
    device: str,
    structured: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    factual_values = []
    null_values = []
    candidates = list(record["candidates"])
    batch_size = int(config["scoring"]["candidate_batch_size"])
    for start in range(0, len(candidates), batch_size):
        pairs = [(record, candidate) for candidate in candidates[start : start + batch_size]]
        factual = tokenizer.batch_encode(
            pairs, "factual", int(config["seed"]), structured=structured
        )
        null = tokenizer.batch_encode(
            pairs, "null", int(config["seed"]), structured=structured
        )
        joint = {
            key: torch.cat([factual[key], null[key]], dim=0) for key in factual
        }
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ):
            logits = model.score(_move(joint, device)).float()
        width = len(pairs)
        factual_values.append(logits[:width])
        null_values.append(logits[width:])
    return torch.cat(factual_values), torch.cat(null_values)


def _score_record(
    model: PrefixDeltaRanker,
    tokenizer: PrefixTokenizer,
    record: dict[str, Any],
    config: dict[str, Any],
    device: str,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    structured = model.mode != "concat_head"
    factual, null = _paired_branch_logits(
        model, tokenizer, record, config, device, structured
    )
    mask = torch.ones((1, len(record["candidates"])), dtype=torch.bool, device=device)
    present = torch.tensor([bool(record.get("history"))], dtype=torch.bool, device=device)
    history_ids = {str(event.get("item_id")) for event in record.get("history", [])}
    exact = torch.tensor(
        [
            [str(candidate["item_id"]) in history_ids for candidate in record["candidates"]]
        ],
        dtype=torch.bool,
        device=device,
    )
    outputs = model.combine(
        factual.unsqueeze(0),
        null.unsqueeze(0),
        mask,
        present,
        exact_repeat=exact,
    )
    rows = {
        "mean_abs_delta_factual": float(outputs["raw_delta"].abs().mean()),
        "mean_abs_tangent_factual": float(outputs["tangent_delta"].abs().mean()),
    }
    if diagnostics and bool(record.get("history")):
        for corruption in DIAGNOSTIC_CORRUPTIONS:
            corrupt = _branch_logits(
                model, tokenizer, record, corruption, config, device, structured
            )
            if corruption == "query_masked":
                corrupt_null = _branch_logits(
                    model,
                    tokenizer,
                    record,
                    "query_masked_null",
                    config,
                    device,
                    structured,
                )
            else:
                corrupt_null = null
            _, raw_delta = model.tangent_delta(
                corrupt.unsqueeze(0),
                corrupt_null.unsqueeze(0),
                mask,
                present,
            )
            rows[f"mean_abs_delta_{corruption}"] = float(raw_delta.abs().mean())
    return outputs["final"].squeeze(0), rows


def score_dev(
    config: dict[str, Any],
    config_path: str | Path,
    checkpoint_path: str | Path,
    run_id: str,
    device: str,
    output_dir: str | Path | None = None,
    limit_requests: int | None = None,
    diagnostics: bool = True,
) -> dict[str, Any]:
    if not run_id.startswith("20260710_kuaisearch_c04_"):
        raise ValueError(f"invalid C04 run prefix: {run_id}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise ValueError("C04 GPU commands must set CUDA_VISIBLE_DEVICES=3")
    candidate_hash = assert_candidate_manifest(
        config["paths"]["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    records_path = Path(config["paths"]["records_dev"])
    if records_path.name != "records_dev.jsonl":
        raise ValueError(f"C04 screening scores only the frozen blind dev file: {records_path}")
    target = Path(output_dir) if output_dir else Path("runs") / run_id
    target.mkdir(parents=True, exist_ok=False)
    lock_path = target / "run.lock"
    lock_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    model = PrefixDeltaRanker(config, mode="paired_delta").to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "paired_delta":
        raise ValueError(f"screening checkpoint is not paired_delta: {checkpoint.get('mode')}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer = _tokenizer(config)
    wrong_donors = _load_train_history_donors(config["paths"]["probe_train"])
    diagnostic_limit = int(config["scoring"]["diagnostic_history_requests"])
    diagnostic_totals: dict[str, float] = {}
    diagnostic_count = 0
    request_count = 0
    score_rows = 0
    scores_path = target / "scores.jsonl"
    try:
        with scores_path.open("w", encoding="utf-8") as score_handle, torch.inference_mode():
            for record in iter_jsonl(records_path):
                if limit_requests is not None and request_count >= int(limit_requests):
                    break
                assert_label_free_record(record)
                record = _attach_wrong_history(record, wrong_donors, int(config["seed"]))
                do_diagnostics = (
                    diagnostics
                    and bool(record.get("history"))
                    and diagnostic_count < diagnostic_limit
                )
                scores, diagnostic_row = _score_record(
                    model,
                    tokenizer,
                    record,
                    config,
                    device,
                    diagnostics=do_diagnostics,
                )
                scores_cpu = scores.float().cpu().tolist()
                if len(scores_cpu) != len(record["candidates"]):
                    raise ValueError("candidate score count mismatch")
                for candidate, score in zip(record["candidates"], scores_cpu):
                    score_handle.write(
                        json.dumps(
                            {
                                "candidate_item_id": str(candidate["item_id"]),
                                "method_id": "c04_prefix_delta_lm",
                                "request_id": str(record["request_id"]),
                                "score": finite_float(score),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    score_rows += 1
                if do_diagnostics:
                    for key, value in diagnostic_row.items():
                        diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + value
                    diagnostic_count += 1
                request_count += 1
    finally:
        if lock_path.exists():
            lock_path.unlink()
    elapsed = time.time() - started
    diagnostic_means = {
        key: value / diagnostic_count for key, value in diagnostic_totals.items()
    }
    factual = diagnostic_means.get("mean_abs_delta_factual", 0.0)
    diagnostic_means["corruption_ratios_to_factual"] = {
        corruption: (
            diagnostic_means.get(f"mean_abs_delta_{corruption}", 0.0) / factual
            if factual > 0
            else 0.0
        )
        for corruption in DIAGNOSTIC_CORRUPTIONS
    }
    write_json(
        target / "delta_diagnostics.json",
        {
            "history_present_requests": diagnostic_count,
            "means": diagnostic_means,
            "qrels_read": False,
            "test_read": False,
        },
    )
    if output_dir is None:
        shutil.copy2(config_path, target / "config_snapshot.yaml")
    metadata = {
        "candidate_id": config["candidate_id"],
        "candidate_manifest_path": config["paths"]["candidate_manifest"],
        "candidate_manifest_sha256": candidate_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": "3",
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "elapsed_seconds": elapsed,
        "env_group": "system-04",
        "env_name": "myrec-c04",
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hostname": socket.gethostname(),
        "input_fields_used": [
            "query",
            "strictly-prior history item/text/event/order",
            "fixed candidate item/text",
        ],
        "limit_requests": limit_requests,
        "mean_latency_ms_per_request_including_tokenization": (
            1000.0 * elapsed / request_count if request_count else None
        ),
        "method_id": "c04_prefix_delta_lm",
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(0) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "qrels_read": False,
        "request_count": request_count,
        "run_id": run_id,
        "score_rows": score_rows,
        "scores_sha256": sha256_file(scores_path),
        "seed": int(config["seed"]),
        "split": "dev_label_free",
        "test_read": False,
    }
    write_json(target / "metadata.json", metadata)
    return metadata
