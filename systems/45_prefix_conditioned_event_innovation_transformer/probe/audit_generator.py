"""Pre-lock audit of C45 synthetic task headroom; no candidate is trained."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.synthetic import generate, ndcg10  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = generate(
        config,
        requests=int(config["generator"]["validation_requests"]),
        split_offset=200_000,
    )
    base = ndcg10(rows.query_only_scores, rows.labels)
    oracle = ndcg10(rows.oracle_scores, rows.labels)
    result = {
        "candidate_id": "c45",
        "trained_model_used": False,
        "repository_data_or_labels_used": False,
        "requests": len(rows.query),
        "finite": bool(
            rows.query.isfinite().all()
            and rows.candidates.isfinite().all()
            and rows.history.isfinite().all()
            and rows.oracle_scores.isfinite().all()
        ),
        "positive_positions_vary": int(rows.labels.argmax(1).unique().numel()) > 1,
        "query_only_ndcg10": float(base.mean()),
        "oracle_ndcg10": float(oracle.mean()),
        "oracle_minus_query_only": float((oracle - base).mean()),
        "history_nonconstant": float(rows.history.std()) > 0.0,
    }
    result["status"] = "passed" if all(
        [
            result["finite"],
            result["positive_positions_vary"],
            result["history_nonconstant"],
            result["oracle_minus_query_only"] >= 0.15,
        ]
    ) else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
