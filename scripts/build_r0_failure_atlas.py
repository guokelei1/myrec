#!/usr/bin/env python
"""Build the R0 failure atlas from frozen run families and label-free surfaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_history_signal_observability import cluster_bootstrap, derived_seed  # noqa: E402


FAMILY_COMPARISONS = {
    "full_token_minus_item_only": ("full_token", "item_only"),
    "full_token_minus_D2s": ("full_token", "D2s"),
    "full_token_minus_D2p": ("full_token", "D2p"),
    "item_only_minus_D2p": ("item_only", "D2p"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-token-run-ids", nargs=3, required=True)
    parser.add_argument("--item-only-run-ids", nargs=3, required=True)
    parser.add_argument("--d2s-run-ids", nargs=3, required=True)
    parser.add_argument("--d2p-run-ids", nargs=3, required=True)
    parser.add_argument("--records-dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20267200)
    return parser.parse_args()


def normalized_characters(value: str) -> set[str]:
    return set(re.findall(r"[0-9a-z\u3400-\u9fff]", value.lower()))


def query_recall(query: str, titles: list[str]) -> float | None:
    query_chars = normalized_characters(query)
    if not query_chars or not titles:
        return None
    title_chars = normalized_characters(" ".join(titles))
    return len(query_chars & title_chars) / len(query_chars)


def user_fold(user_id: str) -> int:
    return int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16) % 2


def cohort_tags(record: dict[str, Any]) -> set[str]:
    history = record["history"]
    candidates = record["candidates"]
    history_ids = {str(row["item_id"]) for row in history}
    repeated = [row for row in candidates if str(row["item_id"]) in history_ids]
    tags = {"all"}
    if not history:
        tags.add("no_history")
        return tags
    tags.add("history_present")
    length = len(history)
    if length == 1:
        tags.add("history_len_1")
    elif length <= 3:
        tags.add("history_len_2_3")
    elif length <= 6:
        tags.add("history_len_4_6")
    else:
        tags.add("history_len_7_plus")
    if not repeated:
        tags.add("strict_nonrepeat")
        recall = query_recall(str(record["query"]), [str(row.get("title", "")) for row in history])
        if recall is not None:
            tags.add("nonrepeat_history_query_aligned" if recall >= 0.8 else "nonrepeat_history_query_weak")
    else:
        tags.add("repeat_present")
        tags.add("repeat_candidates_1" if len(repeated) == 1 else "repeat_candidates_2_plus")
        recall = query_recall(str(record["query"]), [str(row.get("title", "")) for row in repeated])
        if recall is not None:
            if recall >= 0.8:
                tags.add("repeat_query_aligned")
            elif recall <= 0.4:
                tags.add("repeat_query_conflict")
            else:
                tags.add("repeat_query_intermediate")
    return tags


def load_records(path: Path) -> list[dict[str, Any]]:
    if "qrels" in path.name.lower() or "test" in path.name.lower():
        raise PermissionError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(any("clicked" in candidate or "purchased" in candidate for candidate in row["candidates"]) for row in rows):
        raise PermissionError("failure atlas requires label-free dev records")
    return rows


def load_run(run_id: str) -> dict[str, float]:
    path = ROOT / "runs" / run_id / "per_request_metrics.jsonl"
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values[str(row["request_id"])] = float(row["ndcg@10"])
    return values


def load_family(run_ids: list[str]) -> dict[str, float]:
    runs = [load_run(run_id) for run_id in run_ids]
    request_sets = {frozenset(run) for run in runs}
    if len(request_sets) != 1:
        raise RuntimeError(f"run family request sets differ: {run_ids}")
    request_ids = sorted(runs[0])
    return {request_id: float(np.mean([run[request_id] for run in runs])) for request_id in request_ids}


def main() -> int:
    args = parse_args()
    run_families = {
        "full_token": list(args.full_token_run_ids),
        "item_only": list(args.item_only_run_ids),
        "D2s": list(args.d2s_run_ids),
        "D2p": list(args.d2p_run_ids),
    }
    families = {name: load_family(run_ids) for name, run_ids in run_families.items()}
    request_sets = {frozenset(values) for values in families.values()}
    if len(request_sets) != 1:
        raise RuntimeError("failure-atlas families differ in request coverage")
    records = load_records(ROOT / args.records_dev)
    record_by_id = {str(row["request_id"]): row for row in records}
    request_ids = sorted(next(iter(request_sets)))
    if set(record_by_id) != set(request_ids):
        raise RuntimeError("failure-atlas records differ from evaluator requests")
    users = np.asarray([str(record_by_id[request_id]["user_id"]) for request_id in request_ids])
    user_folds = np.asarray([user_fold(value) for value in users], dtype=np.int8)
    tags = {request_id: cohort_tags(record_by_id[request_id]) for request_id in request_ids}
    cohort_names = sorted(set().union(*tags.values()))
    arrays = {
        name: np.asarray([values[request_id] for request_id in request_ids], dtype=np.float64)
        for name, values in families.items()
    }
    cohorts = {}
    for cohort in cohort_names:
        mask = np.asarray([cohort in tags[request_id] for request_id in request_ids], dtype=bool)
        if not bool(mask.any()):
            continue
        row: dict[str, Any] = {
            "requests": int(mask.sum()),
            "users": int(len(np.unique(users[mask]))),
            "family_mean_ndcg_at_10": {name: float(values[mask].mean()) for name, values in arrays.items()},
            "comparisons": {},
            "user_disjoint_folds": {},
        }
        for comparison, (left, right) in FAMILY_COMPARISONS.items():
            difference = arrays[left][mask] - arrays[right][mask]
            seed = derived_seed(args.bootstrap_seed, f"{cohort}:{comparison}")
            row["comparisons"][comparison] = cluster_bootstrap(
                difference, users[mask], samples=args.bootstrap_samples, seed=seed
            )
        for fold in (0, 1):
            fold_mask = mask & (user_folds == fold)
            fold_row: dict[str, Any] = {
                "requests": int(fold_mask.sum()),
                "users": int(len(np.unique(users[fold_mask]))),
                "family_mean_ndcg_at_10": {
                    name: float(values[fold_mask].mean()) for name, values in arrays.items()
                },
                "comparisons": {},
            }
            for comparison, (left, right) in FAMILY_COMPARISONS.items():
                difference = arrays[left][fold_mask] - arrays[right][fold_mask]
                seed = derived_seed(args.bootstrap_seed, f"{cohort}:{comparison}:fold{fold}")
                fold_row["comparisons"][comparison] = cluster_bootstrap(
                    difference, users[fold_mask], samples=args.bootstrap_samples, seed=seed
                )
            row["user_disjoint_folds"][str(fold)] = fold_row
        cohorts[cohort] = row
    report = {
        "report_id": "pps_r0_failure_atlas",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_phase": "R0-D",
        "surface_definitions": {
            "repeat_query_aligned": "repeat candidate title covers >=0.8 of normalized query character set",
            "repeat_query_conflict": "repeat candidate title covers <=0.4 of normalized query character set",
            "nonrepeat_history_query_aligned": "history titles cover >=0.8 of normalized query character set",
            "nonrepeat_history_query_weak": "history-title query coverage <0.8",
            "threshold_role": "exploratory localization only; not a paper gate or architecture threshold",
            "user_fold": "sha256_utf8_user_id_integer_mod_2; fold0 discovery and fold1 replication",
        },
        "run_families": run_families,
        "cohorts": cohorts,
        "label_boundary": {
            "records_are_label_free": True,
            "evaluator_outputs_used": True,
            "direct_qrels_read": False,
            "test_opened": False,
            "c80_fresh_labels_opened": False,
        },
    }
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "cohorts": {name: row["requests"] for name, row in cohorts.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
