"""Qrels-blind four-model Transformer instrumentation identity smoke."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _answer_target_tokens,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _score_instructrec_request,
    _score_yes_no_request,
    _single_token_id,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    build_prompt_sections,
    encode_instructrec_selection_prompt,
    encode_prompt_sections,
    instructrec_template_index,
    load_v12_ranker_config,
)
from myrec.mechanism.transformer_instrumentation import (
    BLOCK_NODE_IDS,
    FINAL_NODE_IDS,
    NodeSpec,
    QwenNodeCallAudit,
    QwenNodeCapture,
    QwenNodePatch,
    QwenPostAttentionStatePatch,
    resolve_qwen_backbone,
)
from myrec.mechanism.attention_instrumentation import (
    QwenAttentionInterfaceAudit,
    attention_audit_summary,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


DEEP_DIVE_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
MAX_SMOKE_REQUESTS = 32
MAX_IDENTITY_GATE_REQUESTS = 128
IDENTITY_TOLERANCE = 1.0e-5
DETAILED_BLOCKS = (13, 20, 27)


def run_deep_dive_identity_smoke(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    max_requests: int = 1,
    identity_gate: bool = False,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Exercise native scoring, every direct hook, and identity patching.

    The output is permanently marked ``mechanical_smoke_non_result`` and never
    reads any qrels file.
    """

    _validate_run_id(run_id)
    max_requests = int(max_requests)
    maximum_requests = (
        MAX_IDENTITY_GATE_REQUESTS if identity_gate else MAX_SMOKE_REQUESTS
    )
    if not 0 < max_requests <= maximum_requests:
        raise ValueError(f"max_requests must be in [1, {maximum_requests}]")
    if not str(device).strip():
        raise ValueError("an explicit smoke device is required")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"smoke run directory is not empty: {run_dir}")

    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    frozen_model = manifest["frozen_inputs"]["models"].get(method_id)
    if frozen_model is None:
        raise ValueError("deep-dive smoke admits only frozen Q0--Q3")
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("deep-dive config differs from frozen manifest")

    records_path = standardized_dir / "records_dev.jsonl"
    request_manifest_path = standardized_dir / "request_manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    for path in (
        records_path,
        request_manifest_path,
        candidate_manifest_path,
        dataset_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        records_path: manifest["frozen_inputs"]["records_dev_sha256"],
        request_manifest_path: manifest["frozen_inputs"]["request_manifest_sha256"],
        candidate_manifest_path: manifest["frozen_inputs"]["candidate_manifest_sha256"],
        dataset_manifest_path: manifest["frozen_inputs"]["dataset_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"deep-dive frozen input hash mismatch: {path.name}")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(
        training_metadata,
        config,
        allow_smoke=False,
    )
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, method_id
    )
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("deep-dive checkpoint differs from frozen manifest")
    if training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("deep-dive checkpoint changed after training metadata")

    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    records.sort(
        key=lambda row: (
            hashlib.sha256(
                f"transformer-deep-dive-smoke-v1\0{row.request_id}".encode("utf-8")
            ).hexdigest(),
            row.request_id,
        )
    )
    selected = records[:max_requests]
    if len(selected) != max_requests:
        raise ValueError("deep-dive smoke request coverage is incomplete")

    tokenizer, model = _load_model_and_tokenizer(
        config,
        device=device,
        training=False,
        checkpoint_model_dir=checkpoint_model_dir,
    )
    torch = _torch()
    backbone = resolve_qwen_backbone(model)
    backend = str(backbone.layers[0].self_attn.config._attn_implementation)
    specs = _smoke_capture_specs()
    patch_specs = tuple(
        NodeSpec(node_id=node_id, block=13)
        for node_id in BLOCK_NODE_IDS
    ) + tuple(NodeSpec(node_id=node_id, block=None) for node_id in FINAL_NODE_IDS)
    node_max_abs_delta = {spec.key: 0.0 for spec in patch_specs}
    capture_max_abs_delta = 0.0
    attention_wrapper_max_abs_delta = 0.0
    native_attention_wrapper_max_abs_delta = 0.0
    native_max_abs_delta = 0.0
    algebra_max_abs_error = {
        "post_attention_recomposition": 0.0,
        "block_output_recomposition": 0.0,
        "swiglu_recomposition": 0.0,
        "o_projection_recomposition": 0.0,
        "final_rmsnorm_recomposition": 0.0,
        "post_rope_query_norm": 0.0,
        "post_rope_key_norm": 0.0,
    }
    algebra_max_allowed_error = {key: 0.0 for key in algebra_max_abs_error}
    calls = 0
    request_rows: list[dict[str, Any]] = []
    attention_summaries: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for record in selected:
                candidate = record.candidates[0]
                native_call_audit = None
                if method_id == "q1_instructrec_generalqwen":
                    with QwenNodeCallAudit(model) as call_audit:
                        native_score = _native_candidate_score(
                            model,
                            tokenizer,
                            record,
                            candidate,
                            config,
                            device=device,
                        )
                        native_call_audit = call_audit.result()
                    with QwenAttentionInterfaceAudit(model) as native_attention_audit:
                        native_attention_audit.arm_all_calls()
                        wrapped_native_score = _native_candidate_score(
                            model,
                            tokenizer,
                            record,
                            candidate,
                            config,
                            device=device,
                        )
                        native_attention_call_audit = (
                            native_attention_audit.disarm_all_calls()
                        )
                    native_attention_wrapper_max_abs_delta = max(
                        native_attention_wrapper_max_abs_delta,
                        abs(wrapped_native_score - native_score),
                    )
                else:
                    native_score = _native_candidate_score(
                        model,
                        tokenizer,
                        record,
                        candidate,
                        config,
                        device=device,
                    )
                    native_attention_call_audit = None
                sequences = _instrumented_sequences(
                    tokenizer,
                    record,
                    candidate,
                    config,
                )
                direct_score = 0.0
                captured_score = 0.0
                patched_scores = {spec.key: 0.0 for spec in patch_specs}
                call_rows = []
                for sequence in sequences:
                    input_ids = torch.tensor(
                        [sequence["input_ids"]], dtype=torch.long, device=device
                    )
                    attention_mask = torch.ones_like(input_ids)
                    positions = torch.tensor(
                        [sequence["score_positions"]],
                        dtype=torch.long,
                        device=device,
                    )
                    baseline_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=int(sequence["logits_to_keep"]),
                    )
                    baseline_value = _score_output(
                        baseline_output,
                        sequence,
                        tokenizer,
                    )
                    with QwenAttentionInterfaceAudit(model) as attention_audit:
                        attention_audit.arm(
                            positions,
                            sequence_length=input_ids.shape[1],
                        )
                        wrapped_output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            logits_to_keep=int(sequence["logits_to_keep"]),
                        )
                        attention_values = attention_audit.disarm()
                    wrapped_value = _score_output(
                        wrapped_output,
                        sequence,
                        tokenizer,
                    )
                    attention_wrapper_max_abs_delta = max(
                        attention_wrapper_max_abs_delta,
                        abs(wrapped_value - baseline_value),
                    )
                    attention_summaries.append(attention_audit_summary(attention_values))
                    with QwenNodeCapture(model, specs) as capture:
                        captured_output, values = capture.capture_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            positions=positions,
                            model_kwargs={
                                "logits_to_keep": int(sequence["logits_to_keep"])
                            },
                        )
                    captured_value = _score_output(
                        captured_output,
                        sequence,
                        tokenizer,
                    )
                    capture_max_abs_delta = max(
                        capture_max_abs_delta,
                        abs(captured_value - baseline_value),
                    )
                    _update_algebra_errors(
                        model,
                        values,
                        algebra_max_abs_error,
                        algebra_max_allowed_error,
                        block=13,
                    )
                    _update_rope_norm_errors(
                        values,
                        attention_values,
                        algebra_max_abs_error,
                        algebra_max_allowed_error,
                        block=13,
                    )
                    weight = float(sequence["aggregate_weight"])
                    direct_score += weight * baseline_value
                    captured_score += weight * captured_value
                    for spec in patch_specs:
                        if spec.node_id == "post_attention_residual":
                            context: Any = QwenPostAttentionStatePatch(model, 13)
                            context.__enter__()
                            context.arm(
                                positions,
                                values[spec.key],
                                sequence_length=input_ids.shape[1],
                            )
                        else:
                            context = QwenNodePatch(model, spec)
                            context.__enter__()
                            context.arm(
                                positions,
                                values[spec.key],
                                sequence_length=input_ids.shape[1],
                            )
                        try:
                            patched_output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                logits_to_keep=int(sequence["logits_to_keep"]),
                            )
                            context.disarm()
                        finally:
                            context.__exit__(None, None, None)
                        patched_value = _score_output(
                            patched_output,
                            sequence,
                            tokenizer,
                        )
                        delta = abs(patched_value - baseline_value)
                        node_max_abs_delta[spec.key] = max(
                            node_max_abs_delta[spec.key], delta
                        )
                        patched_scores[spec.key] += weight * patched_value
                    calls += 1
                    call_rows.append(
                        {
                            "branch": sequence["branch"],
                            "input_tokens": len(sequence["input_ids"]),
                            "score_positions": list(sequence["score_positions"]),
                        }
                    )
                native_max_abs_delta = max(
                    native_max_abs_delta,
                    abs(native_score - direct_score),
                )
                for spec in patch_specs:
                    node_max_abs_delta[spec.key] = max(
                        node_max_abs_delta[spec.key],
                        abs(patched_scores[spec.key] - direct_score),
                    )
                request_rows.append(
                    {
                        "request_id": record.request_id,
                        "candidate_id": str(candidate["item_id"]),
                        "native_score": native_score,
                        "direct_score": direct_score,
                        "captured_score": captured_score,
                        "native_decoder_call_audit": native_call_audit,
                        "native_attention_call_audit": native_attention_call_audit,
                        "calls": call_rows,
                    }
                )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    identity_errors = [
        capture_max_abs_delta,
        attention_wrapper_max_abs_delta,
        native_attention_wrapper_max_abs_delta,
        *node_max_abs_delta.values(),
    ]
    if method_id != "q1_instructrec_generalqwen":
        identity_errors.append(native_max_abs_delta)
    maximum_identity_error = max(identity_errors)
    algebra_passed = all(
        algebra_max_abs_error[key] <= algebra_max_allowed_error[key]
        for key in algebra_max_abs_error
    )
    status = (
        "completed"
        if maximum_identity_error <= IDENTITY_TOLERANCE and algebra_passed
        else "failed_identity"
    )
    metadata = {
        "schema_version": 1,
        "analysis_stage": (
            "transformer_deep_dive_d0_numerical_identity_gate"
            if identity_gate
            else "transformer_deep_dive_d0_identity_smoke"
        ),
        "status": status,
        "run_id": run_id,
        "evidence_mode": (
            "numerical_identity_gate"
            if identity_gate
            else "mechanical_smoke_non_result"
        ),
        "result_eligible": False,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "deep_dive_manifest_path": str(manifest["path"]),
        "deep_dive_manifest_sha256": manifest["sha256"],
        "records_dev_sha256": sha256_file(records_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "selected_request_ids": [row.request_id for row in selected],
        "selected_request_ids_sha256": sha256_text(
            json.dumps([row.request_id for row in selected], separators=(",", ":"))
        ),
        "request_count": len(selected),
        "sequence_calls": calls,
        "detailed_blocks": list(DETAILED_BLOCKS),
        "captured_node_count": len(specs),
        "patched_node_count": len(patch_specs),
        "actual_attention_backend": backend,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "native_vs_direct_max_abs_score_delta": native_max_abs_delta,
        "native_vs_direct_role": (
            "descriptive_execution_mode_crosscheck"
            if method_id == "q1_instructrec_generalqwen"
            else "identity_gate"
        ),
        "native_attention_wrapper_noop_max_abs_score_delta": (
            native_attention_wrapper_max_abs_delta
        ),
        "capture_noop_max_abs_score_delta": capture_max_abs_delta,
        "attention_wrapper_noop_max_abs_score_delta": attention_wrapper_max_abs_delta,
        "attention_wrapper_summaries": attention_summaries,
        "node_identity_max_abs_score_delta": node_max_abs_delta,
        "algebra_max_abs_error": algebra_max_abs_error,
        "algebra_max_allowed_error": algebra_max_allowed_error,
        "algebra_recomposition_passed": algebra_passed,
        "maximum_identity_error": maximum_identity_error,
        "requests": request_rows,
        "qrels_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "metadata.json", metadata)
    if status != "completed":
        raise RuntimeError(
            "deep-dive identity smoke failed: "
            f"identity={maximum_identity_error}, algebra_passed={algebra_passed}"
        )
    return metadata


def _smoke_capture_specs() -> tuple[NodeSpec, ...]:
    specs: list[NodeSpec] = []
    for block in range(28):
        specs.append(NodeSpec("block_input_residual", block))
        specs.append(NodeSpec("block_output_residual", block))
    for block in DETAILED_BLOCKS:
        for node_id in BLOCK_NODE_IDS:
            spec = NodeSpec(node_id, block)
            if spec not in specs:
                specs.append(spec)
    specs.extend(NodeSpec(node_id, None) for node_id in FINAL_NODE_IDS)
    return tuple(specs)


def _instrumented_sequences(
    tokenizer: Any,
    record: Any,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    method_id = str(config["method_id"])
    training = config["training"]
    if method_id == "q1_instructrec_generalqwen":
        max_target = int(training.get("max_target_length", 96))
        prompt, response_by_item, _audit = encode_instructrec_selection_prompt(
            tokenizer,
            record,
            record.candidates,
            history=record.history,
            history_budget=int(training["history_budget"]),
            template_index=instructrec_template_index(
                record.request_id, seed=int(training["seed"])
            ),
            max_length=int(training["max_length"]) - max_target,
            context_token_budget=int(training["context_token_budget"]),
            max_target_length=max_target,
        )
        target = response_by_item[str(candidate["item_id"])]
        return [_target_sequence(prompt, target, branch="candidate_response", weight=1.0)]

    yes_target = _answer_target_tokens(tokenizer, "Yes")
    no_target = _answer_target_tokens(tokenizer, "No")
    reserve = max(len(yes_target), len(no_target)) if method_id == "q3_tallrec_generalqwen" else 0
    sections = build_prompt_sections(
        method_id,
        record,
        candidate,
        history=record.history,
        history_budget=int(training["history_budget"]),
    )
    prompt = encode_prompt_sections(
        tokenizer,
        sections,
        max_length=int(training["max_length"]) - reserve,
    )
    if method_id == "q3_tallrec_generalqwen":
        return [
            _target_sequence(prompt, yes_target, branch="Yes", weight=1.0),
            _target_sequence(prompt, no_target, branch="No", weight=-1.0),
        ]
    return [
        {
            "branch": "prompt",
            "input_ids": list(prompt),
            "score_positions": [len(prompt) - 1],
            "logits_to_keep": 1,
            "target": None,
            "aggregate_weight": 1.0,
        }
    ]


def _target_sequence(
    prompt: Sequence[int],
    target: Sequence[int],
    *,
    branch: str,
    weight: float,
) -> dict[str, Any]:
    if not prompt or not target:
        raise ValueError("deep-dive target sequence is empty")
    return {
        "branch": branch,
        "input_ids": [*prompt, *target],
        "score_positions": list(range(len(prompt) - 1, len(prompt) + len(target) - 1)),
        "logits_to_keep": len(target) + 1,
        "target": list(target),
        "aggregate_weight": float(weight),
    }


def _score_output(output: Any, sequence: Mapping[str, Any], tokenizer: Any) -> float:
    torch = _torch()
    target = sequence["target"]
    if target is None:
        logits = output.logits[0, -1]
        yes_id = _single_token_id(tokenizer, "yes")
        no_id = _single_token_id(tokenizer, "no")
        return float((logits[yes_id] - logits[no_id]).float().cpu().item())
    length = len(target)
    logits = output.logits[0, -(length + 1) : -1].float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    expected = torch.tensor(target, dtype=torch.long, device=logits.device)
    return float(log_probs.gather(1, expected[:, None]).mean().cpu().item())


def _native_candidate_score(
    model: Any,
    tokenizer: Any,
    record: Any,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: str,
) -> float:
    if config["method_id"] == "q1_instructrec_generalqwen":
        scores, _boundary = _score_instructrec_request(
            model,
            tokenizer,
            record,
            record.history,
            dict(config),
            device=device,
            batch_size=1,
        )
    else:
        scores, _boundary = _score_yes_no_request(
            model,
            tokenizer,
            record,
            record.history,
            dict(config),
            device=device,
            batch_size=1,
        )
    return float(scores[str(candidate["item_id"])])


def _update_algebra_errors(
    model: Any,
    values: Mapping[str, Any],
    errors: dict[str, float],
    allowed: dict[str, float],
    *,
    block: int,
) -> None:
    torch = _torch()
    key = lambda node: NodeSpec(node, block).key
    r_raw = values[key("block_input_residual")]
    a_raw = values[key("attention_o_projection")]
    u_raw = values[key("post_attention_residual")]
    m_raw = values[key("mlp_down_projection")]
    r_next_raw = values[key("block_output_residual")]
    gate_raw = values[key("mlp_gate_projection")]
    up_raw = values[key("mlp_up_projection")]
    product_raw = values[key("mlp_swiglu_product")]
    r, a, u = r_raw.float(), a_raw.float(), u_raw.float()
    m, r_next = m_raw.float(), r_next_raw.float()
    gate, up, product = gate_raw.float(), up_raw.float(), product_raw.float()
    _record_algebra(
        "post_attention_recomposition", u, r + a, u_raw, errors, allowed
    )
    _record_algebra(
        "block_output_recomposition", r_next, u + m, r_next_raw, errors, allowed
    )
    _record_algebra(
        "swiglu_recomposition",
        product,
        torch.nn.functional.silu(gate) * up,
        product_raw,
        errors,
        allowed,
    )
    backbone = resolve_qwen_backbone(model)
    layer = backbone.layers[block]
    pre_o = values[key("attention_head_output_pre_o")].float()
    o_expected = torch.nn.functional.linear(
        pre_o,
        layer.self_attn.o_proj.weight.float(),
        None if layer.self_attn.o_proj.bias is None else layer.self_attn.o_proj.bias.float(),
    )
    _record_algebra(
        "o_projection_recomposition", a, o_expected, a_raw, errors, allowed
    )
    final_in_raw = values[NodeSpec("final_rmsnorm_input", None).key]
    final_out_raw = values[NodeSpec("final_rmsnorm_output", None).key]
    final_in = final_in_raw.float()
    final_out = final_out_raw.float()
    norm = backbone.final_norm
    expected = final_in * torch.rsqrt(
        final_in.pow(2).mean(-1, keepdim=True) + float(norm.variance_epsilon)
    )
    expected = expected * norm.weight.float()
    _record_algebra(
        "final_rmsnorm_recomposition",
        final_out,
        expected,
        final_out_raw,
        errors,
        allowed,
    )


def _update_rope_norm_errors(
    values: Mapping[str, Any],
    attention_values: Mapping[int, Mapping[str, Any]],
    errors: dict[str, float],
    allowed: dict[str, float],
    *,
    block: int,
) -> None:
    q_pre_raw = values[NodeSpec("q_post_norm_pre_rope", block).key]
    k_pre_raw = values[NodeSpec("k_post_norm_pre_rope", block).key]
    q_post_raw = attention_values[block]["post_rope_query"]
    k_post_raw = attention_values[block]["post_rope_key"]
    q_pre, k_pre = q_pre_raw.float(), k_pre_raw.float()
    q_post, k_post = q_post_raw.float(), k_post_raw.float()
    _record_algebra(
        "post_rope_query_norm",
        q_pre.pow(2).sum(-1).sqrt(),
        q_post.pow(2).sum(-1).sqrt(),
        q_post_raw,
        errors,
        allowed,
    )
    _record_algebra(
        "post_rope_key_norm",
        k_pre.pow(2).sum(-1).sqrt(),
        k_post.pow(2).sum(-1).sqrt(),
        k_post_raw,
        errors,
        allowed,
    )


def _record_algebra(
    key: str,
    observed: Any,
    expected: Any,
    dtype_reference: Any,
    errors: dict[str, float],
    allowed: dict[str, float],
) -> None:
    errors[key] = max(errors[key], _max_abs(observed, expected))
    allowed[key] = max(
        allowed[key], _dtype_recomposition_bound(dtype_reference, observed)
    )


def _dtype_recomposition_bound(dtype_reference: Any, scale_reference: Any) -> float:
    torch = _torch()
    epsilon = float(torch.finfo(dtype_reference.dtype).eps)
    scale = max(
        1.0, float(scale_reference.float().abs().max().cpu().item())
    )
    return 4.0 * epsilon * scale


def _max_abs(left: Any, right: Any) -> float:
    value = (left - right).abs().max().cpu().item()
    if not math.isfinite(float(value)):
        raise FloatingPointError("deep-dive algebra error is non-finite")
    return float(value)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deep-dive manifest is not a mapping")
    if payload.get("status") != "frozen_before_transformer_deep_dive_outcomes":
        raise ValueError("deep-dive manifest is not frozen")
    return {**payload, "path": path, "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _torch() -> Any:
    import torch

    return torch
