"""Adapters for the B4o RecBole SASRec baseline."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_recbole_atomic_interactions(
    interactions_path: str | Path,
    output_root: str | Path,
    dataset_name: str = "kuaisearch_b4o",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert the shared Batch 2b interactions artifact to RecBole atomic format."""

    interactions_path = Path(interactions_path)
    output_root = Path(output_root)
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    inter_path = dataset_dir / f"{dataset_name}.inter"
    tmp_path = inter_path.with_name(f"{inter_path.name}.{os.getpid()}.tmp")

    users: set[str] = set()
    items: set[str] = set()
    event_counts = {"click": 0, "purchase": 0}
    rows = 0
    min_time: int | None = None
    max_time: int | None = None
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for row in iter_jsonl(interactions_path):
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            event_time = int(row["event_time"])
            event_type = str(row["event_type"])
            handle.write(f"{user_id}\t{item_id}\t{event_time}\n")
            users.add(user_id)
            items.add(item_id)
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            rows += 1
            min_time = event_time if min_time is None else min(min_time, event_time)
            max_time = event_time if max_time is None else max(max_time, event_time)
    tmp_path.replace(inter_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "event_counts": event_counts,
        "input_interactions_path": str(interactions_path),
        "input_interactions_sha256": sha256_file(interactions_path),
        "output_inter_path": str(inter_path),
        "output_inter_sha256": sha256_file(inter_path),
        "rows": rows,
        "schema": "user_id:token item_id:token timestamp:float",
        "time_range": [min_time, max_time],
        "unique_items": len(items),
        "unique_users": len(users),
    }
    if report_path is not None:
        write_json(report_path, manifest)
    return manifest


def train_and_score_b4o_sasrec(
    config: dict[str, Any],
    run_id: str,
    split: str = "dev",
    seed: int = 20260708,
    runs_dir: str | Path = "runs",
    epochs: int | None = None,
    recbole_overrides: dict[str, Any] | None = None,
    saved: bool = True,
    max_score_requests: int | None = None,
) -> dict[str, Any]:
    """Train RecBole SASRec and export project-standard fixed-candidate scores."""

    # Import RecBole only inside the execution path used by the pps-recbole env.
    import recbole
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.interaction import Interaction
    from recbole.quick_start.quick_start import construct_transform
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    standardized_dir = Path(config["standardized_dir"]).resolve()
    runs_dir = Path(runs_dir)
    run_dir = (runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    atomic_root = Path(config["recbole"]["data_dir"]).resolve()
    dataset_name = config["recbole"]["dataset_name"]
    atomic_manifest = write_recbole_atomic_interactions(
        interactions_path=config["train_interactions"]["path"],
        output_root=atomic_root,
        dataset_name=dataset_name,
        report_path=run_dir / "atomic_manifest.json",
    )

    recbole_config = _build_recbole_config(
        config=config,
        atomic_root=atomic_root,
        dataset_name=dataset_name,
        run_dir=run_dir,
        seed=seed,
        epochs=epochs,
        overrides=recbole_overrides or {},
    )

    started = time.perf_counter()
    previous_cwd = Path.cwd()
    previous_argv = sys.argv[:]
    os.chdir(run_dir)
    try:
        sys.argv = [previous_argv[0]]
        rb_config = Config(
            model=recbole_config["model"],
            dataset=dataset_name,
            config_dict=recbole_config["config_dict"],
        )
        init_seed(rb_config["seed"], rb_config["reproducibility"])
        init_logger(rb_config)
        dataset = create_dataset(rb_config)
        train_data, valid_data, test_data = data_preparation(rb_config, dataset)
        init_seed(rb_config["seed"] + rb_config["local_rank"], rb_config["reproducibility"])
        model = get_model(rb_config["model"])(rb_config, train_data._dataset).to(rb_config["device"])
        transform = construct_transform(rb_config)
        trainer = get_trainer(rb_config["MODEL_TYPE"], rb_config["model"])(rb_config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=saved,
            show_progress=rb_config["show_progress"],
        )
        test_result = trainer.evaluate(
            test_data,
            load_best_model=saved,
            show_progress=rb_config["show_progress"],
        )
        if saved:
            checkpoint = torch.load(trainer.saved_model_file, map_location=rb_config["device"])
            model.load_state_dict(checkpoint["state_dict"])
            model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
        score_stats = _score_split(
            model=model,
            dataset=dataset,
            rb_config=rb_config,
            standardized_dir=standardized_dir,
            split=split,
            run_dir=run_dir,
            method_id=config["method_id"],
            max_score_requests=max_score_requests,
        )
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    elapsed = time.perf_counter() - started

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    _copy_if_exists(config.get("_config_path"), config_snapshot_path)

    metadata = {
        "atomic_manifest": atomic_manifest,
        "best_valid_result": _jsonable(best_valid_result),
        "best_valid_score": _jsonable(best_valid_score),
        "candidate_manifest_path": str(standardized_dir / "candidate_manifest.json"),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "config_path": config.get("_config_path"),
        "config_sha256": sha256_file(config["_config_path"]) if config.get("_config_path") else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "env_group": config["environment_group"],
        "env_name": config["environment_name"],
        "git_dirty": None,
        "hostname": platform.node(),
        "implementation_type": config["implementation_type"],
        "input_fields_used": config["input_fields_used"],
        "method_id": config["method_id"],
        "package_versions": {
            "recbole": recbole.__version__,
            "torch": torch.__version__,
        },
        "python": platform.python_version(),
        "qrels_read": False,
        "recbole_config": recbole_config,
        "run_id": run_id,
        "score_stats": score_stats,
        "seed": seed,
        "split": split,
        "test_result_internal_train_split": _jsonable(test_result),
        "timing": {"train_and_score_seconds": elapsed},
        "train_interactions": config["train_interactions"],
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _build_recbole_config(
    config: dict[str, Any],
    atomic_root: Path,
    dataset_name: str,
    run_dir: Path,
    seed: int,
    epochs: int | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    rb = config["recbole"]
    config_dict: dict[str, Any] = {
        "ITEM_ID_FIELD": rb["item_id_field"],
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "LIST_SUFFIX": "_list",
        "MAX_ITEM_LIST_LENGTH": int(rb["max_item_list_length"]),
        "TIME_FIELD": rb["time_field"],
        "USER_ID_FIELD": rb["user_id_field"],
        "checkpoint_dir": str((run_dir / "recbole_checkpoints").resolve()),
        "data_path": str(atomic_root),
        "eval_args": {
            "group_by": "user",
            "mode": {"test": "full", "valid": "full"},
            "order": "TO",
            "split": {"LS": "valid_and_test"},
        },
        "eval_batch_size": 4096,
        "gpu_id": int(rb.get("gpu_id", 0)),
        "load_col": {"inter": [rb["user_id_field"], rb["item_id_field"], rb["time_field"]]},
        "log_wandb": False,
        "loss_type": "CE",
        "metrics": ["Recall", "NDCG"],
        "reproducibility": True,
        "seed": seed,
        "show_progress": bool(rb.get("show_progress", False)),
        "topk": [10],
        "train_neg_sample_args": None,
        "use_gpu": bool(rb.get("use_gpu", True)),
        "valid_metric": "NDCG@10",
    }
    if epochs is not None:
        config_dict["epochs"] = int(epochs)
    config_dict.update(overrides)
    return {"config_dict": config_dict, "dataset": dataset_name, "model": rb["model"]}


def _score_split(
    model: Any,
    dataset: Any,
    rb_config: Any,
    standardized_dir: Path,
    split: str,
    run_dir: Path,
    method_id: str,
    max_score_requests: int | None,
) -> dict[str, Any]:
    import torch
    from recbole.data.interaction import Interaction

    item_field = rb_config["ITEM_ID_FIELD"]
    item_list_field = item_field + rb_config["LIST_SUFFIX"]
    item_length_field = rb_config["ITEM_LIST_LENGTH_FIELD"]
    max_len = int(rb_config["MAX_ITEM_LIST_LENGTH"])
    item_token_map = dataset.field2token_id[item_field]
    device = rb_config["device"]
    margin = 1.0

    score_rows = 0
    request_count = 0
    candidate_rows = 0
    cold_candidate_rows = 0
    scored_candidate_rows = 0
    zero_history_requests = 0
    zero_in_vocab_candidate_requests = 0
    output_path = run_dir / "scores.jsonl"
    records_path = standardized_dir / f"records_{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            if max_score_requests is not None and request_count >= max_score_requests:
                break
            request_count += 1
            request_id = str(record["request_id"])
            candidates = record.get("candidates") or []
            history_tokens = [
                item_token_map[str(event["item_id"])]
                for event in record.get("history") or []
                if str(event.get("item_id")) in item_token_map
            ][-max_len:]
            history_len = len(history_tokens)
            candidate_token_ids = [
                item_token_map.get(str(candidate["item_id"])) for candidate in candidates
            ]
            in_vocab_candidate_count = sum(token is not None for token in candidate_token_ids)
            cold_count = len(candidates) - in_vocab_candidate_count
            candidate_rows += len(candidates)
            cold_candidate_rows += cold_count
            if history_len == 0:
                zero_history_requests += 1
            if in_vocab_candidate_count == 0:
                zero_in_vocab_candidate_requests += 1

            raw_scores: dict[int, float] = {}
            if history_len > 0 and in_vocab_candidate_count > 0:
                item_seq = torch.zeros((1, max_len), dtype=torch.long, device=device)
                item_seq[0, :history_len] = torch.tensor(history_tokens, dtype=torch.long, device=device)
                item_length = torch.tensor([history_len], dtype=torch.long, device=device)
                interaction = Interaction(
                    {
                        item_list_field: item_seq,
                        item_length_field: item_length,
                    }
                )
                with torch.no_grad():
                    full_scores = model.full_sort_predict(interaction).detach().cpu().view(-1)
                raw_scores = {
                    int(token): float(full_scores[int(token)].item())
                    for token in candidate_token_ids
                    if token is not None
                }
            cold_score = min(raw_scores.values()) - margin if raw_scores else 0.0
            scored_candidate_rows += len(raw_scores)

            for candidate, token in zip(candidates, candidate_token_ids):
                in_vocab = token is not None and token in raw_scores
                score = raw_scores[int(token)] if in_vocab else cold_score
                row = {
                    "candidate_in_vocab_count": in_vocab_candidate_count,
                    "candidate_item_id": str(candidate["item_id"]),
                    "cold_candidate_count": cold_count,
                    "history_in_vocab_len": history_len,
                    "in_vocab": bool(in_vocab),
                    "method_id": method_id,
                    "request_id": request_id,
                    "score": score,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                score_rows += 1

    return {
        "candidate_rows": candidate_rows,
        "cold_candidate_rate": cold_candidate_rows / candidate_rows if candidate_rows else 0.0,
        "cold_candidate_rows": cold_candidate_rows,
        "max_score_requests": max_score_requests,
        "request_count": request_count,
        "score_rows": score_rows,
        "scored_candidate_rows": scored_candidate_rows,
        "scores_path": str(output_path),
        "scores_sha256": sha256_file(output_path),
        "zero_history_request_rate": zero_history_requests / request_count if request_count else 0.0,
        "zero_history_requests": zero_history_requests,
        "zero_in_vocab_candidate_request_rate": (
            zero_in_vocab_candidate_requests / request_count if request_count else 0.0
        ),
        "zero_in_vocab_candidate_requests": zero_in_vocab_candidate_requests,
    }


def _copy_if_exists(source: str | Path | None, target: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if source_path.exists():
        shutil.copyfile(source_path, target)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)
