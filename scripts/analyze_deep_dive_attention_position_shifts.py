#!/usr/bin/env python3
"""Audit full/null semantic position shifts on the frozen D3 sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_observation_runtime import (
    SAMPLE_MANIFEST_SHA256,
    _audit_sample,
    _build_observation_paths,
)
from myrec.mechanism.attention_position_shift_analysis import (
    MODELS,
    summarize_attention_position_shifts,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


DEFAULT_STANDARDIZED_DIR = Path(
    "data/standardized/kuaisearch/full_confirm_preceding40k_v11"
)
DEFAULT_SAMPLE_MANIFEST = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/manifest.json"
)
DEFAULTS = {
    MODELS[0]: {
        "config": Path(
            "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
        ),
        "checkpoint": Path(
            "artifacts/motivation_v1_2/checkpoints/"
            "q2_recranker_generalqwen_seed20260714"
        ),
        "metadata": Path(
            "runs/20260718_kuaisearch_mech_d3_q2_attention_heads_b13_v2/metadata.json"
        ),
    },
    MODELS[1]: {
        "config": Path(
            "configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
        ),
        "checkpoint": Path(
            "artifacts/motivation_v1_2/checkpoints/"
            "q3_tallrec_generalqwen_seed20260714"
        ),
        "metadata": Path(
            "runs/20260718_kuaisearch_mech_d3_q3_attention_heads_b13_v2/metadata.json"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", default=str(DEFAULT_STANDARDIZED_DIR))
    parser.add_argument("--sample-manifest", default=str(DEFAULT_SAMPLE_MANIFEST))
    parser.add_argument(
        "--output-dir",
        default="runs/20260719_kuaisearch_mech_d3_position_shift_audit_v1",
    )
    for prefix, model_id in (("q2", MODELS[0]), ("q3", MODELS[1])):
        parser.add_argument(
            f"--{prefix}-config", default=str(DEFAULTS[model_id]["config"])
        )
        parser.add_argument(
            f"--{prefix}-checkpoint-root",
            default=str(DEFAULTS[model_id]["checkpoint"]),
        )
        parser.add_argument(
            f"--{prefix}-source-metadata",
            default=str(DEFAULTS[model_id]["metadata"]),
        )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"position-shift output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    standardized_dir = Path(args.standardized_dir)
    records_path = standardized_dir / "records_dev.jsonl"
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("position-shift audit requires frozen 8000-request dev")
    records_by_id = {record.request_id: record for record in records}

    sample_manifest_path = Path(args.sample_manifest)
    if sha256_file(sample_manifest_path) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("position-shift sample manifest differs")
    sample_manifest = _read_json(sample_manifest_path)
    sample_path = Path(sample_manifest["path"])
    if (
        sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("sha256") != sha256_file(sample_path)
    ):
        raise ValueError("position-shift sample is not frozen qrels/score blind")
    samples = list(iter_jsonl(sample_path))
    _audit_sample(samples, records_by_id)

    model_rows = {}
    source_metadata = {}
    for prefix, model_id in (("q2", MODELS[0]), ("q3", MODELS[1])):
        config_path = Path(getattr(args, f"{prefix}_config"))
        checkpoint_root = Path(getattr(args, f"{prefix}_checkpoint_root"))
        metadata_path = Path(getattr(args, f"{prefix}_source_metadata"))
        config = load_v12_ranker_config(config_path)
        metadata = _read_json(metadata_path)
        model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
        _validate_source(
            metadata,
            method_id=model_id,
            config_path=config_path,
            config_sha256=config["_config_sha256"],
            records_path=records_path,
            sample_manifest_path=sample_manifest_path,
            sample_path=sample_path,
            model_dir=model_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True
        )
        rows = []
        for sample in samples:
            record = records_by_id[str(sample["request_id"])]
            paths = _build_observation_paths(
                tokenizer,
                record,
                int(sample["candidate_ordinal"]),
                config,
                device="cpu",
            )
            row_paths = {}
            for path in paths:
                selected = int(path["selected_batch_row"])
                row_paths[path["name"]] = {
                    "full_positions": [
                        int(value)
                        for value in path["full"]["capture_positions"][selected].tolist()
                    ],
                    "null_positions": [
                        int(value)
                        for value in path["null"]["capture_positions"][selected].tolist()
                    ],
                    "full_sequence_length": int(path["full"]["ids"].shape[1]),
                    "null_sequence_length": int(path["null"]["ids"].shape[1]),
                }
            rows.append(
                {
                    "request_id": record.request_id,
                    "candidate_item_id": str(sample["candidate_item_id"]),
                    "paths": row_paths,
                }
            )
        model_rows[model_id] = rows
        source_metadata[model_id] = {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "tokenizer_json_sha256": sha256_file(model_dir / "tokenizer.json"),
        }

    result = summarize_attention_position_shifts(
        model_rows, expected_rows=512, qrels_read=False, source_test_opened=False
    )
    result.update(
        {
            "records_path": str(records_path),
            "records_sha256": sha256_file(records_path),
            "sample_manifest_path": str(sample_manifest_path),
            "sample_manifest_sha256": sha256_file(sample_manifest_path),
            "sample_rows_path": str(sample_path),
            "sample_rows_sha256": sha256_file(sample_path),
            "source_metadata": source_metadata,
            "qwen_modeling_path": str(Path(modeling_qwen3.__file__)),
            "qwen_modeling_sha256": sha256_file(modeling_qwen3.__file__),
            "position_id_policy": (
                "No explicit position_ids in the registered observation forward; "
                "installed Qwen3Model constructs arange(padded_sequence_length)."
            ),
            "model_weights_loaded": False,
            "model_scores_read": False,
            "command": [str(value) for value in os.sys.argv],
        }
    )
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
                "rows_per_model": result["rows_per_model"],
                "path_cells": len(result["path_cells"]),
                "qrels_read": result["qrels_read"],
                "sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _validate_source(
    metadata,
    *,
    method_id,
    config_path,
    config_sha256,
    records_path,
    sample_manifest_path,
    sample_path,
    model_dir,
):
    checkpoint_files = {
        row["name"]: row["sha256"] for row in metadata.get("checkpoint_files", [])
    }
    if (
        metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or metadata.get("method_id") != method_id
        or metadata.get("observation_rows") != 512
        or metadata.get("config_sha256") != config_sha256
        or metadata.get("config_sha256") != sha256_file(config_path)
        or metadata.get("records_sha256") != sha256_file(records_path)
        or metadata.get("sample_manifest_sha256")
        != sha256_file(sample_manifest_path)
        or metadata.get("sample_rows_sha256") != sha256_file(sample_path)
        or checkpoint_files.get("tokenizer.json")
        != sha256_file(model_dir / "tokenizer.json")
        or checkpoint_files.get("tokenizer_config.json")
        != sha256_file(model_dir / "tokenizer_config.json")
    ):
        raise ValueError("position-shift source binding differs")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
