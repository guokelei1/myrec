"""Run C47's frozen label-free A0 and one-shot two-domain S0 gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(SYSTEM_ROOT), str(C38_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from probe.data import PackedTrain  # noqa: E402
from probe.freeze_signal_lock import load_config, verify_signal_lock  # noqa: E402
from probe.locking import sha256_file  # noqa: E402
from probe.signal_features import collect_amazon, encode_amazon, finalize_amazon  # noqa: E402
from probe.signal_scoring import (  # noqa: E402
    FixedScores,
    fixed_scores,
    flatten,
    max_row_difference,
    unflatten,
)
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.store import FrozenTransferStore, open_role_labels  # noqa: E402


SCORE_NAMES = (
    "base",
    "posterior_supported",
    "plain_ridge",
    "softmax_attention",
    "wrong_posterior",
    "correction",
    "wrong_correction",
    "support",
    "wrong_support",
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class KuaiStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        paths = config["paths"]
        root = REPO_ROOT / paths["kuai_packed_root"]
        self.data = PackedTrain(root)
        self.query_indices = np.load(root / "query_indices.npy", mmap_mode="r")
        self.query_embeddings = np.load(REPO_ROOT / paths["kuai_query_embeddings"], mmap_mode="r")
        self.item_embeddings = np.load(REPO_ROOT / paths["kuai_item_embeddings"], mmap_mode="r")
        if len(self.query_indices) != len(self.data.request_ids):
            raise ValueError("C47 Kuai query index count differs")

    def query(self, index: int) -> np.ndarray:
        return np.asarray(self.query_embeddings[int(self.query_indices[index])], dtype=np.float32)

    def candidates(self, index: int) -> np.ndarray:
        return np.asarray(self.item_embeddings[self.data.candidates(index)], dtype=np.float32)

    def history(self, index: int) -> np.ndarray:
        return np.asarray(self.item_embeddings[self.data.history(index)], dtype=np.float32)

    def request_id(self, index: int) -> str:
        return str(self.data.request_ids[index])

    def candidate_ids(self, index: int) -> list[str]:
        return [str(value) for value in self.data.candidate_ids(index).tolist()]


class AmazonStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.store = FrozenTransferStore(
            {
                "paths": {
                    "selection": str(REPO_ROOT / config["paths"]["amazon_adapter_selection"]),
                    "feature_root": str(REPO_ROOT / config["paths"]["amazon_feature_root"]),
                },
                "model": {"embedding_dim": int(config["encoding"]["embedding_dim"])},
            }
        )

    def query(self, index: int) -> np.ndarray:
        return self.store.query(index)

    def candidates(self, index: int) -> np.ndarray:
        return self.store.items(self.store.candidate_positions(index))

    def history(self, index: int, source: str = "true") -> np.ndarray:
        return self.store.items(self.store.history_positions(index, source))

    def request_id(self, index: int) -> str:
        return self.store.request_id(index)

    def candidate_ids(self, index: int) -> list[str]:
        return self.store.candidate_ids(index)


def candidate_key_sha256(store: Any, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        row = [store.request_id(index), *store.candidate_ids(index)]
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _settings(config: Mapping[str, Any]) -> dict[str, float]:
    row = config["operator"]
    return {
        "ridge": float(row["ridge"]),
        "softmax_temperature": float(row["softmax_temperature"]),
        "epsilon": float(row["normalization_epsilon"]),
    }


def _finite(output: FixedScores) -> bool:
    return all(
        np.isfinite(value).all()
        for value in (
            output.base,
            output.posterior_supported,
            output.plain_ridge,
            output.softmax_attention,
            output.correction,
            output.plain_correction,
            output.support,
            output.query_write,
        )
    )


def score_domain(config: Mapping[str, Any], domain: str) -> dict[str, Any]:
    _, lock_hash = verify_signal_lock(config)
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    if domain == "kuai":
        store: Any = KuaiStore(config)
        role = "kuai_internal_A"
        expected_hash = config["integrity"]["kuai_candidate_key_sha256"]
    elif domain == "amazon":
        store = AmazonStore(config)
        role = "amazon_internal_A"
        expected_hash = config["integrity"]["amazon_candidate_key_sha256"]
    else:
        raise ValueError(f"unknown C47 domain: {domain}")
    indices = [int(value) for value in selection["roles"][role]["indices"]]
    donors = [int(value) for value in selection["wrong_history_donors"][role]["indices"]]
    if candidate_key_sha256(store, indices) != expected_hash:
        raise RuntimeError(f"C47 {domain} candidate key changed before scoring")
    settings = _settings(config)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    history_permutation_max = 0.0
    support_min, support_max = 1.0, 0.0
    finite = True
    contraction = True
    nohistory_exact = True
    for index, donor in zip(indices, donors):
        query = store.query(index)
        candidates = store.candidates(index)
        true_history = store.history(index) if domain == "kuai" else store.history(index, "true")
        wrong_history = store.history(donor) if domain == "kuai" else store.history(index, "wrong")
        primary = fixed_scores(query, true_history, candidates, **settings)
        repeated = fixed_scores(query, true_history, candidates, **settings)
        wrong = fixed_scores(query, wrong_history, candidates, **settings)
        reversed_candidates = fixed_scores(query, true_history, candidates[::-1], **settings)
        reversed_history = fixed_scores(query, true_history[::-1], candidates, **settings)
        empty = fixed_scores(
            query,
            np.empty((0, candidates.shape[1]), dtype=np.float32),
            candidates,
            **settings,
        )
        deterministic_max = max(
            deterministic_max,
            float(np.max(np.abs(primary.posterior_supported - repeated.posterior_supported))),
        )
        candidate_permutation_max = max(
            candidate_permutation_max,
            float(np.max(np.abs(primary.posterior_supported - reversed_candidates.posterior_supported[::-1]))),
            float(np.max(np.abs(primary.support - reversed_candidates.support[::-1]))),
        )
        history_permutation_max = max(
            history_permutation_max,
            float(np.max(np.abs(primary.posterior_supported - reversed_history.posterior_supported))),
            float(np.max(np.abs(primary.support - reversed_history.support))),
        )
        support_min = min(support_min, float(primary.support.min()), float(wrong.support.min()))
        support_max = max(support_max, float(primary.support.max()), float(wrong.support.max()))
        finite = finite and _finite(primary) and _finite(wrong)
        contraction = contraction and bool(
            np.all(np.abs(primary.correction) <= np.abs(primary.plain_correction) + 1e-6)
            and np.all(np.abs(wrong.correction) <= np.abs(wrong.plain_correction) + 1e-6)
        )
        nohistory_exact = nohistory_exact and bool(
            np.array_equal(empty.posterior_supported, empty.base)
            and np.array_equal(empty.plain_ridge, empty.base)
            and np.array_equal(empty.softmax_attention, empty.base)
            and np.count_nonzero(empty.correction) == 0
            and np.count_nonzero(empty.support) == 0
        )
        rows["base"].append(primary.base)
        rows["posterior_supported"].append(primary.posterior_supported)
        rows["plain_ridge"].append(primary.plain_ridge)
        rows["softmax_attention"].append(primary.softmax_attention)
        rows["wrong_posterior"].append(wrong.posterior_supported)
        rows["correction"].append(primary.correction)
        rows["wrong_correction"].append(wrong.correction)
        rows["support"].append(primary.support)
        rows["wrong_support"].append(wrong.support)
    offsets, _ = flatten(rows["base"])
    output_root = REPO_ROOT / config["paths"]["artifact_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    score_path = output_root / f"{domain}_fixed_scores.npz"
    report_path = output_root / f"{domain}_fixed_score_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else report_path)
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(value)[1] for name, value in rows.items()})
    checks = {
        "candidate_key_matches": candidate_key_sha256(store, indices) == expected_hash,
        "request_count_exact": len(indices) == (600 if domain == "kuai" else 300),
        "deterministic": deterministic_max <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": candidate_permutation_max <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "history_permutation": history_permutation_max <= float(config["evaluation"]["history_permutation_tolerance"]),
        "finite": finite,
        "support_bounds": support_min >= -1e-7 and support_max <= 1.0 + 1e-7,
        "posterior_contracts_plain": contraction,
        "nohistory_exact_base": nohistory_exact,
        "A_labels_closed": True,
        "dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c47",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "label_free_fixed_scoring",
        "domain": domain,
        "signal_execution_lock_sha256": lock_hash,
        "requests": len(indices),
        "candidate_key_sha256": expected_hash,
        "checks": checks,
        "diagnostics": {
            "deterministic_max_abs": deterministic_max,
            "candidate_permutation_max_abs": candidate_permutation_max,
            "history_permutation_max_abs": history_permutation_max,
            "support_min": support_min,
            "support_max": support_max,
        },
        "score_artifact": {
            "path": str(score_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(score_path),
        },
        "A_labels_opened": False,
        "dev_test_records_labels_qrels_opened": False,
    }
    atomic_json(report_path, report)
    return report


def run_a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_signal_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    target = root / "a0_report.json"
    reports = {
        domain: json.loads((root / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
        for domain in ("kuai", "amazon")
    }
    checks = {
        "both_domain_score_checks": all(all(report["checks"].values()) for report in reports.values()),
        "score_hashes": all(
            sha256_file(REPO_ROOT / report["score_artifact"]["path"])
            == report["score_artifact"]["sha256"]
            for report in reports.values()
        ),
        "same_execution_lock": all(report["signal_execution_lock_sha256"] == lock_hash for report in reports.values()),
        "A_labels_closed_during_features_and_scoring": all(report["A_labels_opened"] is False for report in reports.values()),
        "dev_test_qrels_closed": all(report["dev_test_records_labels_qrels_opened"] is False for report in reports.values()),
    }
    value = {
        "candidate_id": "c47",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "A0_label_release_gate",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "signal_execution_lock_sha256": lock_hash,
        "checks": checks,
        "domain_reports_sha256": {
            domain: sha256_file(root / f"{domain}_fixed_score_report.json")
            for domain in reports
        },
        "score_artifacts": {domain: report["score_artifact"] for domain, report in reports.items()},
        "A_labels_opened": False,
        "A_labels_authorized_after_this_report": all(checks.values()),
        "dev_test_records_labels_qrels_opened": False,
    }
    atomic_json(target, value)
    return value


def load_score_rows(root: Path, report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C47 fixed score artifact changed")
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def rankings(
    request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], scores: Sequence[np.ndarray]
) -> list[list[str]]:
    return [
        [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)])]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def ndcg_rows(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for ranked, items, label in zip(rankings(request_ids, item_ids, scores), item_ids, labels):
        positives = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def kuai_labels(config: Mapping[str, Any], store: KuaiStore, indices: Sequence[int]) -> list[np.ndarray]:
    path = REPO_ROOT / config["paths"]["kuai_candidate_labels"]
    if sha256_file(path) != config["integrity"]["kuai_candidate_labels_sha256"]:
        raise RuntimeError("C47 Kuai train labels changed")
    source = np.load(path, mmap_mode="r")
    output = []
    for index in indices:
        start = int(store.data.candidate_offsets[index])
        stop = int(store.data.candidate_offsets[index + 1])
        output.append(np.asarray(source[start:stop], dtype=np.float32).copy())
    return output


def amazon_labels(
    config: Mapping[str, Any], store: AmazonStore, indices: Sequence[int]
) -> list[np.ndarray]:
    compact = open_role_labels(
        records_train_path=REPO_ROOT / config["paths"]["amazon_records_train"],
        records_train_sha256=config["integrity"]["amazon_records_train_sha256"],
        selection_path=REPO_ROOT / config["paths"]["amazon_adapter_selection"],
        selection_sha256=sha256_file(REPO_ROOT / config["paths"]["amazon_adapter_selection"]),
        store=store.store,
        role="internal_A",
    )
    return [compact.row(index, len(store.candidate_ids(index))) for index in indices]


def aggregate_domain(
    config: Mapping[str, Any], domain: str, scores: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, Any]:
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    if domain == "kuai":
        store: Any = KuaiStore(config)
        role = "kuai_internal_A"
        expected_hash = config["integrity"]["kuai_candidate_key_sha256"]
    else:
        store = AmazonStore(config)
        role = "amazon_internal_A"
        expected_hash = config["integrity"]["amazon_candidate_key_sha256"]
    indices = [int(value) for value in selection["roles"][role]["indices"]]
    if candidate_key_sha256(store, indices) != expected_hash:
        raise RuntimeError(f"C47 {domain} candidate key changed immediately before evaluation")
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(config, store, indices) if domain == "kuai" else amazon_labels(config, store, indices)
    ndcg = {
        name: ndcg_rows(request_ids, item_ids, values, labels)
        for name, values in scores.items()
        if name in {"base", "posterior_supported", "plain_ridge", "softmax_attention", "wrong_posterior"}
    }
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids,
        ndcg["posterior_supported"],
        {
            "query_base": ndcg["base"],
            "plain_ridge": ndcg["plain_ridge"],
            "softmax_attention": ndcg["softmax_attention"],
            "wrong_history": ndcg["wrong_posterior"],
        },
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    clicked_true_values = clicked_direction(scores["correction"], labels)
    clicked_wrong_values = clicked_direction(scores["wrong_correction"], labels)
    clicked = bootstrap(
        clicked_true_values,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]) + 20,
    )
    clicked_specific = bootstrap(
        clicked_true_values - clicked_wrong_values,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]) + 21,
    )
    thresholds = {
        "query_base": float(evaluation["primary_minus_base_min"]),
        "plain_ridge": float(evaluation["primary_minus_plain_ridge_min"]),
        "softmax_attention": float(evaluation["primary_minus_softmax_min"]),
        "wrong_history": float(evaluation["true_minus_wrong_min"]),
    }
    checks: dict[str, bool] = {"candidate_key_asserted": True}
    for name, minimum in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_effect"] = row["mean"] >= minimum
        checks[f"{name}_ci"] = row["percentile_95_ci"][0] > 0
        checks[f"{name}_all_folds_positive"] = all(
            fold["mean_difference"] > 0 for fold in row["hash_folds"]
        )
    checks["clicked_direction_ci"] = clicked["percentile_95_ci"][0] > 0
    checks["clicked_true_minus_wrong_ci"] = clicked_specific["percentile_95_ci"][0] > 0
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "requests": len(indices),
        "candidate_key_sha256": expected_hash,
        "checks": checks,
        "mean_ndcg10": {name: float(values.mean()) for name, values in ndcg.items()},
        "comparisons": comparisons,
        "clicked_correction_direction": clicked,
        "clicked_true_minus_wrong": clicked_specific,
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_signal_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0_path = root / "a0_report.json"
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    if a0.get("status") != "passed" or a0.get("A_labels_authorized_after_this_report") is not True:
        raise PermissionError("C47 A0 did not authorize train-label access")
    if a0.get("A_labels_opened") is not False:
        raise PermissionError("C47 A labels were opened before aggregate")
    reports = {
        domain: json.loads((root / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
        for domain in ("kuai", "amazon")
    }
    scores = {domain: load_score_rows(root, report) for domain, report in reports.items()}
    domains = {
        domain: aggregate_domain(config, domain, scores[domain])
        for domain in ("kuai", "amazon")
    }
    result = {
        "candidate_id": "c47",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_id": config["gate_id"],
        "status": "passed_S0_signal_only" if all(row["status"] == "passed" for row in domains.values()) else "failed_S0_terminal",
        "decision": "authorize_matched_control_GPU_architecture_training" if all(row["status"] == "passed" for row in domains.values()) else "close_C47_before_trainable_implementation",
        "signal_execution_lock_sha256": lock_hash,
        "selection_sha256": config["integrity"]["selection_sha256"],
        "A0_report_sha256": sha256_file(a0_path),
        "domains": domains,
        "A_labels_opened_after_A0": True,
        "dev_test_records_labels_qrels_opened": False,
        "claims": {
            "fixed_operator_signal_only": True,
            "trained_architecture_result": False,
            "dev_test_result": False,
        },
    }
    output = root / "signal_gate_report.json"
    promoted = REPO_ROOT / config["paths"]["promoted_report"]
    atomic_json(output, result)
    atomic_json(promoted, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("collect-amazon", "encode-amazon", "finalize-amazon", "score", "a0", "aggregate"),
    )
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    args = parser.parse_args()
    config = load_config(args.config)
    verify_signal_lock(config)
    if args.stage == "collect-amazon":
        value = collect_amazon(config)
    elif args.stage == "encode-amazon":
        if args.shard_id is None:
            raise ValueError("C47 Amazon encoding requires --shard-id")
        value = encode_amazon(config, shard_id=args.shard_id, device=args.device)
    elif args.stage == "finalize-amazon":
        value = finalize_amazon(config)
    elif args.stage == "score":
        if args.domain is None:
            raise ValueError("C47 scoring requires --domain")
        value = score_domain(config, args.domain)
    elif args.stage == "a0":
        value = run_a0(config)
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
