from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.selection import (  # noqa: E402
    candidate_key_sha256,
    iter_structural_records,
    load_json_map,
)
from model import episode_value, memory_value, normalize_rows, score_memory  # noqa: E402


SCORE_NAMES = (
    "base",
    "primary_true",
    "primary_wrong",
    "positive_only",
    "uniform_slate",
    "semantic_history",
    "primary_correction",
    "wrong_correction",
)


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(row)
    return offsets, np.concatenate(rows).astype(np.float32, copy=False)


def rankings(item_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        item_id
        for item_id, _ in sorted(
            zip((str(value) for value in item_ids), (float(value) for value in scores)),
            key=lambda row: (-row[1], row[0]),
        )
    ]


def load_required_records(
    records_path: Path, selection: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    required = {
        *[row["request_id"] for row in selection["targets"]],
        *[row["request_id"] for row in selection["selected_donors"]],
        *selection["source_request_ids"],
    }
    output = {}
    for record in iter_structural_records(records_path):
        if record["request_id"] in required:
            output[record["request_id"]] = record
    missing = required - set(output)
    if missing:
        raise RuntimeError(f"C71 required structural records missing: {len(missing)}")
    for expected in [*selection["targets"], *selection["selected_donors"]]:
        actual = output[expected["request_id"]]
        for key in ("request_id", "user_id", "ts", "query", "candidate_ids", "history"):
            if actual[key] != expected[key]:
                raise RuntimeError(f"C71 selected structural row changed: {expected['request_id']}/{key}")
    return output


def embedding_tensor(
    item_ids: Sequence[str], item_map: Mapping[str, int], item_embeddings: np.ndarray, device: torch.device
) -> torch.Tensor:
    indices = np.asarray([item_map[str(value)] for value in item_ids], dtype=np.int64)
    values = np.asarray(item_embeddings[indices], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(values)).to(device)


class EpisodeCache:
    def __init__(
        self,
        *,
        records: Mapping[str, Mapping[str, Any]],
        item_map: Mapping[str, int],
        item_embeddings: np.ndarray,
        query_map: Mapping[str, int],
        query_embeddings: np.ndarray,
        device: torch.device,
        temperature: float,
        epsilon: float,
    ) -> None:
        self.records = records
        self.item_map = item_map
        self.item_embeddings = item_embeddings
        self.query_map = query_map
        self.query_embeddings = query_embeddings
        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon
        self.cache: dict[tuple[str, str], tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}

    def get(self, source_request_id: str, selected_item_id: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        key = (str(source_request_id), str(selected_item_id))
        if key in self.cache:
            return self.cache[key]
        record = self.records[key[0]]
        ordered_ids = sorted((str(value) for value in record["candidate_ids"]))
        matches = [index for index, value in enumerate(ordered_ids) if value == key[1]]
        if len(matches) != 1:
            raise RuntimeError(f"C71 selected item/source mismatch: {key}")
        slate = embedding_tensor(ordered_ids, self.item_map, self.item_embeddings, self.device)
        query_np = np.asarray(
            self.query_embeddings[int(self.query_map[key[0]])], dtype=np.float32
        )
        query = torch.from_numpy(np.ascontiguousarray(query_np)).to(self.device)
        values = {
            mode: episode_value(
                query,
                slate,
                matches[0],
                mode=mode,
                temperature=self.temperature,
                epsilon=self.epsilon,
            )
            for mode in ("choice_gradient", "positive_only", "uniform_slate")
        }
        self.cache[key] = (normalize_rows(query, self.epsilon), values)
        return self.cache[key]


def episode_memory(
    current_query: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    cache: EpisodeCache,
    *,
    mode: str,
    temperature: float,
    epsilon: float,
) -> tuple[torch.Tensor, int, int]:
    queries, values = [], []
    nonzero = 0
    for episode in episodes:
        query, mode_values = cache.get(episode["source_request_id"], episode["selected_item_id"])
        value = mode_values[mode]
        queries.append(query)
        values.append(value)
        nonzero += int(bool(value.ne(0).any()))
    query_tensor = torch.stack(queries) if queries else torch.empty(0, len(current_query), device=current_query.device)
    value_tensor = torch.stack(values) if values else torch.empty_like(query_tensor)
    return (
        memory_value(
            current_query,
            query_tensor,
            value_tensor,
            temperature=temperature,
            epsilon=epsilon,
        ),
        nonzero,
        len(values),
    )


def score_one(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    item_map: Mapping[str, int],
    item_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    cache: EpisodeCache,
    config: Mapping[str, Any],
    device: torch.device,
    candidate_ids: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    operator = config["operator"]
    epsilon = float(operator["normalization_epsilon"])
    current_query = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(query_embeddings[int(target["query_embedding_index"])], dtype=np.float32)
        )
    ).to(device)
    ids = list(candidate_ids if candidate_ids is not None else target["candidate_ids"])
    candidates = embedding_tensor(ids, item_map, item_embeddings, device)
    primary_memory, nonzero, total = episode_memory(
        current_query,
        target["episodes"],
        cache,
        mode="choice_gradient",
        temperature=float(operator["current_query_episode_temperature"]),
        epsilon=epsilon,
    )
    wrong_memory, _, _ = episode_memory(
        current_query,
        donor["episodes"],
        cache,
        mode="choice_gradient",
        temperature=float(operator["current_query_episode_temperature"]),
        epsilon=epsilon,
    )
    positive_memory, _, _ = episode_memory(
        current_query,
        target["episodes"],
        cache,
        mode="positive_only",
        temperature=float(operator["current_query_episode_temperature"]),
        epsilon=epsilon,
    )
    uniform_memory, _, _ = episode_memory(
        current_query,
        target["episodes"],
        cache,
        mode="uniform_slate",
        temperature=float(operator["current_query_episode_temperature"]),
        epsilon=epsilon,
    )
    history = embedding_tensor(
        [row["item_id"] for row in target["history"]], item_map, item_embeddings, device
    )
    semantic_memory = memory_value(
        current_query,
        history,
        normalize_rows(history, epsilon),
        temperature=float(operator["current_query_episode_temperature"]),
        epsilon=epsilon,
    )
    scale = float(operator["correction_scale"])
    primary, primary_correction = score_memory(
        current_query, candidates, primary_memory, correction_scale=scale, epsilon=epsilon
    )
    wrong, wrong_correction = score_memory(
        current_query, candidates, wrong_memory, correction_scale=scale, epsilon=epsilon
    )
    positive, _ = score_memory(
        current_query, candidates, positive_memory, correction_scale=scale, epsilon=epsilon
    )
    uniform, _ = score_memory(
        current_query, candidates, uniform_memory, correction_scale=scale, epsilon=epsilon
    )
    semantic, _ = score_memory(
        current_query, candidates, semantic_memory, correction_scale=scale, epsilon=epsilon
    )
    base, _ = score_memory(
        current_query,
        candidates,
        torch.zeros_like(current_query),
        correction_scale=scale,
        epsilon=epsilon,
    )
    rows = {
        "base": base,
        "primary_true": primary,
        "primary_wrong": wrong,
        "positive_only": positive,
        "uniform_slate": uniform,
        "semantic_history": semantic,
        "primary_correction": primary_correction,
        "wrong_correction": wrong_correction,
    }
    return {name: value.detach().cpu().numpy().astype(np.float32) for name, value in rows.items()}, (nonzero, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    lock, lock_hash = verify_execution_lock(config)
    expected_gpu = str(config["resources"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"C71 scoring requires physical GPU {expected_gpu}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C71 expects exactly one visible GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")

    paths = config["paths"]
    selection_path = REPO_ROOT / paths["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "passed" or selection["proposal_lock_sha256"] != lock["proposal_lock_sha256"]:
        raise RuntimeError("C71 selection identity differs")
    records = load_required_records(REPO_ROOT / paths["records_train"], selection)
    target_rows = [records[row["request_id"]] for row in selection["targets"]]
    if candidate_key_sha256(target_rows) != selection["candidate_key_sha256"]:
        raise RuntimeError("C71 candidate key changed before scoring")
    item_map = load_json_map(REPO_ROOT / paths["item_id_map"])
    query_map = load_json_map(REPO_ROOT / paths["request_query_map"])
    item_embeddings = np.load(REPO_ROOT / paths["item_embeddings"], mmap_mode="r")
    query_embeddings = np.load(REPO_ROOT / paths["query_embeddings"], mmap_mode="r")
    operator = config["operator"]
    cache = EpisodeCache(
        records=records,
        item_map=item_map,
        item_embeddings=item_embeddings,
        query_map=query_map,
        query_embeddings=query_embeddings,
        device=device,
        temperature=float(operator["historical_query_slate_temperature"]),
        epsilon=float(operator["normalization_epsilon"]),
    )
    donors = {row["request_id"]: row for row in selection["selected_donors"]}
    donor_by_target = {
        row["target_request_id"]: donors[row["donor_request_id"]]
        for row in selection["wrong_donors"]
    }
    scores: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    deterministic = 0.0
    permutation = 0.0
    nohistory = 0.0
    nonzero_events = 0
    total_events = 0
    order_changes = 0
    top10_changes = 0
    started = time.monotonic()
    for position, target in enumerate(selection["targets"]):
        donor = donor_by_target[target["request_id"]]
        output, activity = score_one(
            target,
            donor,
            item_map=item_map,
            item_embeddings=item_embeddings,
            query_embeddings=query_embeddings,
            cache=cache,
            config=config,
            device=device,
        )
        for name in SCORE_NAMES:
            scores[name].append(output[name])
        nonzero_events += activity[0]
        total_events += activity[1]
        base_rank = rankings(target["candidate_ids"], output["base"])
        primary_rank = rankings(target["candidate_ids"], output["primary_true"])
        order_changes += int(base_rank != primary_rank)
        top10_changes += int(base_rank[:10] != primary_rank[:10])
        if position < 64:
            repeat, _ = score_one(
                target, donor, item_map=item_map, item_embeddings=item_embeddings,
                query_embeddings=query_embeddings, cache=cache, config=config, device=device,
            )
            reverse_ids = list(reversed(target["candidate_ids"]))
            reverse, _ = score_one(
                target, donor, item_map=item_map, item_embeddings=item_embeddings,
                query_embeddings=query_embeddings, cache=cache, config=config, device=device,
                candidate_ids=reverse_ids,
            )
            deterministic = max(
                deterministic,
                max(float(np.max(np.abs(output[name] - repeat[name]))) for name in SCORE_NAMES),
            )
            permutation = max(
                permutation,
                max(float(np.max(np.abs(output[name] - reverse[name][::-1]))) for name in SCORE_NAMES),
            )
            empty = torch.zeros(item_embeddings.shape[1], device=device)
            current_query = torch.from_numpy(
                np.asarray(query_embeddings[int(target["query_embedding_index"])], dtype=np.float32).copy()
            ).to(device)
            candidates = embedding_tensor(target["candidate_ids"], item_map, item_embeddings, device)
            empty_scores, empty_correction = score_memory(
                current_query, candidates, empty,
                correction_scale=float(operator["correction_scale"]),
                epsilon=float(operator["normalization_epsilon"]),
            )
            base = torch.from_numpy(output["base"]).to(device)
            nohistory = max(
                nohistory,
                float((empty_scores - base).abs().max().cpu()),
                float(empty_correction.abs().max().cpu()),
            )

    root = REPO_ROOT / paths["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "scores.npz"
    report_path = root / "a0_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else report_path)
    offsets, _ = flatten(scores["base"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(rows)[1] for name, rows in scores.items()})
    primary_flat = flatten(scores["primary_correction"])[1].astype(np.float64)
    wrong_flat = flatten(scores["wrong_correction"])[1].astype(np.float64)
    primary_rms = float(np.sqrt(np.mean(primary_flat**2)))
    true_wrong_rms = float(np.sqrt(np.mean((primary_flat - wrong_flat) ** 2)))
    mechanics = config["mechanical_gate"]
    request_count = len(selection["targets"])
    diagnostics = {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation,
        "nohistory_max_abs": nohistory,
        "nonzero_episode_gradient_fraction": nonzero_events / max(1, total_events),
        "primary_correction_rms": primary_rms,
        "true_wrong_correction_rms": true_wrong_rms,
        "order_change_fraction_vs_base": order_changes / request_count,
        "top10_change_fraction_vs_base": top10_changes / request_count,
        "requests": request_count,
        "episode_values_cached": len(cache.cache),
    }
    checks = {
        "selection_passed": selection["status"] == "passed",
        "candidate_key_matches": candidate_key_sha256(target_rows) == selection["candidate_key_sha256"],
        "request_count_exact": request_count == int(config["selection"]["target_requests"]),
        "scores_finite": all(np.isfinite(row).all() for values in scores.values() for row in values),
        "deterministic": deterministic <= float(mechanics["deterministic_tolerance"]),
        "candidate_permutation": permutation <= float(mechanics["candidate_permutation_tolerance"]),
        "nohistory_exact": nohistory <= float(mechanics["nohistory_tolerance"]),
        "episode_gradients_active": diagnostics["nonzero_episode_gradient_fraction"] >= float(mechanics["minimum_nonzero_episode_gradient_fraction"]),
        "primary_correction_active": primary_rms >= float(mechanics["minimum_primary_correction_rms"]),
        "true_wrong_active": true_wrong_rms >= float(mechanics["minimum_true_wrong_correction_rms"]),
        "order_active": diagnostics["order_change_fraction_vs_base"] >= float(mechanics["minimum_order_change_fraction_vs_base"]),
        "top10_active": diagnostics["top10_change_fraction_vs_base"] >= float(mechanics["minimum_top10_change_fraction_vs_base"]),
        "target_labels_closed": True,
        "source_episode_labels_closed": True,
        "dev_test_qrels_closed": True,
    }
    report = {
        "schema": "myrec.c71.a0.v1",
        "candidate_id": "c71",
        "created_at": timestamp(),
        "execution_lock_sha256": lock_hash,
        "proposal_lock_sha256": lock["proposal_lock_sha256"],
        "selection_sha256": sha256_file(selection_path),
        "candidate_key_sha256": selection["candidate_key_sha256"],
        "checks": checks,
        "diagnostics": diagnostics,
        "passed_A0": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "score_artifact": {
            "path": str(score_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(score_path),
        },
        "elapsed_seconds": time.monotonic() - started,
        "isolation": {
            "target_labels_opened": False,
            "source_episode_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed_A0": report["passed_A0"], "failed_checks": report["failed_checks"], "diagnostics": diagnostics}, sort_keys=True))


if __name__ == "__main__":
    main()
