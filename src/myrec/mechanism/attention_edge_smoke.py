"""Qrels-blind production-model smoke for Q2 attention edge controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _checkpoint_identity,
    _load_model_and_tokenizer,
    _single_token_id,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_interventions import (
    EDGE_MODES,
    QwenAttentionEdgeIntervention,
)
from myrec.mechanism.representation_probe import instrument_pointwise_prompt
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def run_q2_attention_edge_smoke(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = (
        "experiments/motivation/transformer_deep_dive_manifest.yaml"
    ),
) -> dict[str, Any]:
    """Exercise native-backend edge interventions on one frozen Q2 row."""

    import torch
    import yaml

    _validate_run_id(run_id)
    block = int(block)
    if block not in {13, 20, 27}:
        raise ValueError("attention edge smoke block must be 13, 20, or 27")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"attention edge smoke run is not empty: {run_dir}")
    manifest_path = Path(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id != "q2_recranker_generalqwen":
        raise ValueError("this production attention edge smoke is Q2-only")
    frozen_model = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("attention smoke config differs from frozen manifest")
    records_path = standardized_dir / "records_dev.jsonl"
    if sha256_file(records_path) != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("attention smoke records differ from frozen manifest")
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, method_id
    )
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("attention smoke checkpoint differs from frozen manifest")

    neutral = manifest["frozen_qrels_blind_controls"]["content_neutral"]
    control_manifest_path = Path(neutral["manifest_path"])
    if sha256_file(control_manifest_path) != neutral["manifest_sha256"]:
        raise ValueError("attention smoke content-neutral manifest drifted")
    control_manifest = _read_json(control_manifest_path)
    control_info = control_manifest["methods"][method_id]
    control_path = Path(control_info["path"])
    if sha256_file(control_path) != control_info["sha256"]:
        raise ValueError("attention smoke content-neutral rows drifted")
    controls = {str(row["request_id"]): row for row in iter_jsonl(control_path)}
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    eligible = [row for row in records if controls[row.request_id]["eligible"]]
    eligible.sort(
        key=lambda row: (
            hashlib.sha256(
                f"attention-edge-smoke-v1\0{row.request_id}".encode("utf-8")
            ).hexdigest(),
            row.request_id,
        )
    )
    record = eligible[0]
    candidate = record.candidates[0]
    control = controls[record.request_id]
    tokenizer, model = _load_model_and_tokenizer(
        config,
        device=device,
        training=False,
        checkpoint_model_dir=checkpoint_model_dir,
    )
    model.eval()
    prompt = instrument_pointwise_prompt(
        tokenizer,
        method_id,
        record,
        candidate,
        history=record.history,
        history_budget=int(config["training"]["history_budget"]),
        max_length=int(config["training"]["max_length"]),
    )
    ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    readout = torch.tensor([prompt.candidate_readout], dtype=torch.long, device=device)
    starts = torch.tensor(
        [int(control["history_span_start"])], dtype=torch.long, device=device
    )
    ends = torch.tensor(
        [int(control["history_span_end_exclusive"])], dtype=torch.long, device=device
    )
    if int(ends[0]) > prompt.candidate_start:
        raise ValueError("attention smoke frozen history span crosses candidate")
    yes_id = _single_token_id(tokenizer, "yes")
    no_id = _single_token_id(tokenizer, "no")

    def score() -> float:
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = output.logits[0, -1]
        return float((logits[yes_id] - logits[no_id]).float().cpu().item())

    rows: dict[str, Any] = {}
    with torch.inference_mode():
        baseline = score()
        for mode in EDGE_MODES:
            with QwenAttentionEdgeIntervention(model, block, mode) as intervention:
                intervention.arm(
                    readout,
                    starts,
                    ends,
                    sequence_length=ids.shape[1],
                )
                value = score()
                summary = intervention.disarm()
            rows[mode] = {
                "score": value,
                "score_minus_baseline": value - baseline,
                "attention_summary": summary,
            }
    maximum_identity = max(
        abs(rows[mode]["score_minus_baseline"])
        for mode in ("zero_additive_delta", "mask_then_restore_output")
    )
    status = "completed" if maximum_identity <= 1.0e-5 else "mechanical_failure"
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d3_attention_edge_smoke",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_sha256": config["_config_sha256"],
        "deep_dive_manifest_sha256": sha256_file(manifest_path),
        "content_neutral_manifest_sha256": sha256_file(control_manifest_path),
        "request_id": record.request_id,
        "candidate_item_id": str(candidate["item_id"]),
        "block_zero_based": block,
        "history_span": [int(starts[0]), int(ends[0])],
        "readout_position": int(readout[0]),
        "baseline_score": baseline,
        "conditions": rows,
        "maximum_identity_error": maximum_identity,
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": "mechanical_smoke_non_result",
        "result_eligible": False,
        "status": status,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "metadata.json", metadata)
    if status != "completed":
        raise RuntimeError(f"attention edge identity smoke failed: {maximum_identity}")
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

