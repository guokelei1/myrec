#!/usr/bin/env python3
"""Build the qrels-blind Q2/Q3 native-readout descriptive appendix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myrec.mechanism.native_readout_diagnostics import (
    summarize_native_readout_diagnostics,
)
from myrec.mechanism.native_readout_runtime import (
    Q2_METHOD_ID,
    Q2_NATIVE_READOUT_CONDITIONS,
)
from myrec.mechanism.q3_native_readout_runtime import (
    Q3_METHOD_ID,
    Q3_NATIVE_READOUT_CONDITIONS,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q2-bundle",
        default="runs/20260718_kuaisearch_mech_d6_q2_native_readout_v1",
    )
    parser.add_argument(
        "--q3-bundle",
        default="runs/20260718_kuaisearch_mech_d6_q3_native_readout_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/20260719_kuaisearch_mech_d6_native_readout_diagnostics_v1",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"native-readout diagnostics output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {}
    requests = {}
    for name, root_value, method_id, conditions in (
        ("q2", args.q2_bundle, Q2_METHOD_ID, Q2_NATIVE_READOUT_CONDITIONS),
        ("q3", args.q3_bundle, Q3_METHOD_ID, Q3_NATIVE_READOUT_CONDITIONS),
    ):
        root = Path(root_value)
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scores_path = root / "scores.jsonl"
        if (
            metadata.get("status") != "completed"
            or metadata.get("method_id") != method_id
            or metadata.get("qrels_read") is not False
            or metadata.get("source_test_opened") is not False
            or metadata.get("complete_finite_score_coverage") is not True
            or metadata.get("request_count") != 8000
            or metadata.get("score_rows") != 160753
            or metadata.get("score_conditions") != list(conditions)
            or metadata.get("scores_path") != str(scores_path)
            or metadata.get("scores_sha256") != sha256_file(scores_path)
        ):
            raise ValueError(f"native-readout {name} completed source boundary differs")
        requests[name] = list(iter_jsonl(scores_path))
        sources[name] = {
            "root": str(root),
            "metadata_sha256": sha256_file(metadata_path),
            "scores_sha256": sha256_file(scores_path),
        }

    result = summarize_native_readout_diagnostics(
        requests["q2"],
        requests["q3"],
        expected_requests=8000,
        expected_score_rows=160753,
        qrels_read=False,
        source_test_opened=False,
    )
    result.update({"sources": sources, "command": [str(value) for value in os.sys.argv]})
    output_path = output_dir / "metrics.json"
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "request_count": result["request_count"],
                "score_rows": result["score_rows"],
                "qrels_read": result["qrels_read"],
                "sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
