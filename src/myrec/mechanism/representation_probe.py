"""Frozen M2 representation-probe contracts and train-only linear fitter.

This module deliberately separates three boundaries:

* prompt instrumentation and activation-bundle validation never accept qrels;
* the fitter may open only ``qrels_train.jsonl`` after auditing a train bundle;
* internal-dev labels are owned by :mod:`representation_evaluator` and are not
  imported here.

The preference classifier uses only the causal history-summary position.  The
query endpoint and candidate readout are retained for layer-shift/mediation
diagnostics, never as classifier features.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import (
    ModelRecord,
    build_prompt_sections,
    encode_prompt_sections,
    sanitize_record_for_model,
    serialize_history,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


MECHANISM_PROBE_MANIFEST_PATH = Path("experiments/motivation/probe_manifest.yaml")
MECHANISM_PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
M2_ANCHORS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
M2_CONDITIONS = ("full", "null", "relevant_6", "irrelevant_6")
M2_BLOCKS = (6, 13, 20, 27)
M2_HIDDEN_STATE_INDICES = (0, 7, 14, 21, 28)
M2_PATCH_BLOCKS = (13, 27)
M2_TRAIN_REQUESTS = 8192
M2_MAX_CLASSES = 64
M2_MIN_CLASS_FREQUENCY = 100
M2_SEED = 20_260_717
REQUEST_POSITIONS = ("query_end", "history_summary_end")
CANDIDATE_POSITIONS = ("candidate_readout",)
PREFERENCE_POSITION = "history_summary_end"
PREFIX_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
)
ANSWER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class MechanicalPositionError(ValueError):
    """A prompt could not satisfy the preregistered mechanical position rule."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class InstrumentedPrompt:
    """Frozen prompt IDs plus causal positions in the unpadded sequence."""

    token_ids: tuple[int, ...]
    query_end: int
    history_summary_end: int
    candidate_readout: int
    candidate_start: int
    context_tokens: int
    candidate_tokens: int
    prompt_at_max_boundary: bool

    @property
    def request_positions(self) -> tuple[int, int]:
        return (self.query_end, self.history_summary_end)


@dataclass(frozen=True)
class AuditedActivationBundle:
    """Integrity-checked activation bundle descriptor."""

    root: Path
    metadata: dict[str, Any]
    index: dict[str, Any]
    request_ids: tuple[str, ...]
    candidate_count: int
    hidden_size: int


@dataclass(frozen=True)
class LinearReadout:
    """A standardized multiclass linear readout saved without pickle."""

    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("linear-readout feature shape mismatch")
        standardized = (values - self.mean) / self.scale
        scores = standardized @ self.coefficient.T + self.intercept
        return np.asarray(self.classes, dtype=np.str_)[scores.argmax(axis=1)]


def load_m2_probe_manifest(
    path: str | Path = MECHANISM_PROBE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load the exact frozen M2 manifest and reject any recipe drift."""

    import yaml

    path = Path(path)
    observed = sha256_file(path)
    if path.as_posix() != MECHANISM_PROBE_MANIFEST_PATH.as_posix():
        raise ValueError("M2 requires the repository-frozen probe manifest path")
    if observed != MECHANISM_PROBE_MANIFEST_SHA256:
        raise ValueError("frozen mechanism probe manifest hash mismatch")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("mechanism probe manifest must be an object")
    m2 = manifest.get("m2_representation_and_mediation")
    if not isinstance(m2, dict):
        raise ValueError("mechanism probe manifest is missing M2")
    expected = {
        "anchors": list(M2_ANCHORS),
        "transformer_blocks_zero_based": list(M2_BLOCKS),
        "hidden_state_indices": list(M2_HIDDEN_STATE_INDICES),
        "representation_conditions": list(M2_CONDITIONS),
        "representation_negative_controls": [
            "random_labels",
            "embedding_state_index_0",
        ],
        "patch_blocks_zero_based": list(M2_PATCH_BLOCKS),
        "patch_negative_controls": [
            "full_to_full_identity",
            "cross_request_same_layer_patch",
        ],
    }
    for key, value in expected.items():
        observed_value = m2.get(key)
        # YAML 1.1/1.2 loaders parse the unquoted condition token ``null`` as
        # None.  Normalize only this registered condition list; the file hash
        # remains the primary byte-identity boundary.
        if key == "representation_conditions" and isinstance(observed_value, list):
            observed_value = [
                "null" if item is None else item for item in observed_value
            ]
        if observed_value != value:
            raise ValueError(f"frozen M2 recipe drift: {key}")
    return {
        "path": path.as_posix(),
        "sha256": observed,
        "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "verified": True,
        "manifest_id": manifest.get("probe_manifest_id"),
        "m2": m2,
        "frozen_inputs": manifest.get("frozen_inputs"),
    }


def representation_probe_implementation_identity() -> dict[str, Any]:
    """Bind the train fitter/core contract and its production CLI bytes."""

    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/representation_probe.py",
        root / "scripts/fit_m2_representation_probe.py",
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {"files": files, "digest": _canonical_object_sha256(files)}


def instrument_pointwise_prompt(
    tokenizer: Any,
    method_id: str,
    record: ModelRecord,
    candidate: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]],
    history_budget: int,
    max_length: int,
) -> InstrumentedPrompt:
    """Locate exact query/history/readout positions in the frozen Q2/Q3 prompt.

    Offset failure, query/history truncation, or tokenization mismatch is a
    mechanical failure.  The function never substitutes a nearby token.
    """

    if method_id not in M2_ANCHORS:
        raise ValueError(f"M2 activation anchors exclude method_id={method_id}")
    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    sections = build_prompt_sections(
        method_id,
        record,
        dict(candidate),
        history=list(history),
        history_budget=history_budget,
    )
    prefix = PREFIX_TEMPLATE.format(system=sections.system)
    prefix_ids = _encode(tokenizer, prefix)
    context_ids, context_offsets = _encode_with_offsets(tokenizer, sections.context)
    candidate_ids = _encode(tokenizer, sections.candidate)
    suffix_ids = _encode(tokenizer, ANSWER_SUFFIX)

    body_budget = max_length - len(prefix_ids) - len(suffix_ids)
    if body_budget < 16:
        raise ValueError("max_length leaves no room for prompt body")
    candidate_budget = min(len(candidate_ids), max(8, body_budget // 2))
    context_budget = body_budget - candidate_budget
    if len(context_ids) < context_budget:
        candidate_budget = min(len(candidate_ids), body_budget - len(context_ids))
        context_budget = body_budget - candidate_budget
    elif len(candidate_ids) < candidate_budget:
        context_budget = body_budget - len(candidate_ids)
        candidate_budget = len(candidate_ids)

    history_text = serialize_history(history, history_budget=history_budget)
    query_span, history_span = _context_spans(
        method_id,
        sections.context,
        query=record.query,
        history_text=history_text,
    )
    query_token = _token_covering_span_end(
        context_offsets, query_span[1], name="query_end"
    )
    history_token = _token_covering_span_end(
        context_offsets, history_span[1], name="history_summary_end"
    )
    if query_token >= context_budget:
        raise MechanicalPositionError(
            "query_endpoint_truncated",
            f"query token {query_token} outside context budget {context_budget}",
        )
    if history_token >= context_budget:
        raise MechanicalPositionError(
            "history_endpoint_truncated",
            f"history token {history_token} outside context budget {context_budget}",
        )
    if candidate_budget != len(candidate_ids):
        raise MechanicalPositionError(
            "candidate_text_truncated",
            f"visible candidate tokens {candidate_budget}/{len(candidate_ids)}",
        )

    token_ids = (
        prefix_ids
        + context_ids[:context_budget]
        + candidate_ids[:candidate_budget]
        + suffix_ids
    )
    frozen = encode_prompt_sections(tokenizer, sections, max_length=max_length)
    if token_ids != frozen:
        raise MechanicalPositionError(
            "frozen_encoder_mismatch",
            "instrumented token IDs differ from encode_prompt_sections",
        )
    query_position = len(prefix_ids) + query_token
    history_position = len(prefix_ids) + history_token
    candidate_start = len(prefix_ids) + min(len(context_ids), context_budget)
    readout = len(token_ids) - 1
    if not (0 <= query_position < history_position < candidate_start <= readout):
        raise MechanicalPositionError(
            "noncausal_position_order",
            "expected query < history < candidate block <= readout",
        )
    return InstrumentedPrompt(
        token_ids=tuple(token_ids),
        query_end=query_position,
        history_summary_end=history_position,
        candidate_readout=readout,
        candidate_start=candidate_start,
        context_tokens=min(len(context_ids), context_budget),
        candidate_tokens=candidate_budget,
        prompt_at_max_boundary=len(token_ids) == max_length,
    )


def select_train_probe_records(
    records: Iterable[Mapping[str, Any]],
    *,
    limit: int = M2_TRAIN_REQUESTS,
) -> tuple[list[ModelRecord], dict[str, Any]]:
    """Label-free strict-transfer superset, then stable-hash first ``limit``.

    Eligibility is deliberately stronger than target-level non-recurrence: a
    request must have visible history and *no* candidate/history ID overlap.
    Thus the selection is independent of which candidate qrels later identify
    as positive.
    """

    if limit <= 0:
        raise ValueError("train probe selection limit must be positive")
    eligible: list[ModelRecord] = []
    seen: set[str] = set()
    total = 0
    no_history = 0
    overlap = 0
    for raw in records:
        total += 1
        record = sanitize_record_for_model(dict(raw))
        if record.request_id in seen:
            raise ValueError(f"duplicate train request_id={record.request_id}")
        seen.add(record.request_id)
        history_ids = {str(row["item_id"]) for row in record.history}
        candidate_ids = {str(row["item_id"]) for row in record.candidates}
        if not history_ids:
            no_history += 1
            continue
        if history_ids & candidate_ids:
            overlap += 1
            continue
        eligible.append(record)
    eligible.sort(key=lambda row: (_stable_hex("m2_train", row.request_id), row.request_id))
    selected = eligible[:limit]
    if not selected:
        raise ValueError("M2 train selection is empty")
    return selected, {
        "selection": "history_present_and_full_slate_history_id_disjoint_then_sha256_first",
        "selection_label_free": True,
        "requested_limit": limit,
        "source_requests": total,
        "eligible_requests": len(eligible),
        "selected_requests": len(selected),
        "skipped_no_history": no_history,
        "skipped_candidate_history_overlap": overlap,
        "selected_request_ids_sha256": sha256_text(
            json.dumps(
                [row.request_id for row in selected],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    }


def normalize_query(query: str) -> str:
    """Frozen normalized-query cluster: Unicode casefold, remove whitespace."""

    return "".join(str(query).casefold().split())


def representation_holdout(query: str) -> bool:
    """Stable normalized-query 25% probe holdout (one hash bucket of four)."""

    cluster = normalize_query(query)
    if not cluster:
        raise ValueError("normalized query cluster is empty")
    return int(_stable_hex("m2_representation_split", cluster), 16) % 4 == 0


def normalized_query_fold(query: str) -> int:
    """Two-fold reporting assignment from the frozen normalized query."""

    cluster = normalize_query(query)
    if not cluster:
        raise ValueError("normalized query cluster is empty")
    return int(hashlib.sha256(cluster.encode("utf-8")).hexdigest(), 16) % 2


def write_activation_shard(
    path: str | Path,
    *,
    request_ids: Sequence[str],
    normalized_queries: Sequence[str],
    request_activations: np.ndarray,
    candidate_offsets: Sequence[int],
    candidate_ids: Sequence[str],
    candidate_activations: np.ndarray,
) -> dict[str, Any]:
    """Write one non-pickle activation shard atomically and return its identity."""

    path = Path(path)
    request_values = np.asarray(request_activations)
    candidate_values = np.asarray(candidate_activations)
    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    if request_values.ndim != 4 or request_values.shape[1:3] != (
        len(REQUEST_POSITIONS),
        len(M2_HIDDEN_STATE_INDICES),
    ):
        raise ValueError("request activation shard has invalid shape")
    if candidate_values.ndim != 3 or candidate_values.shape[1] != len(
        M2_HIDDEN_STATE_INDICES
    ):
        raise ValueError("candidate activation shard has invalid shape")
    if request_values.shape[0] != len(request_ids):
        raise ValueError("request activation/identity count mismatch")
    if len(normalized_queries) != len(request_ids):
        raise ValueError("normalized-query/request count mismatch")
    if offsets.shape != (len(request_ids) + 1,):
        raise ValueError("candidate offsets have invalid shape")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("candidate offsets must be monotone from zero")
    if int(offsets[-1]) != len(candidate_ids) or len(candidate_ids) != candidate_values.shape[0]:
        raise ValueError("candidate offsets/identity/activation count mismatch")
    if request_values.shape[-1] != candidate_values.shape[-1]:
        raise ValueError("request/candidate hidden sizes differ")
    if not np.isfinite(request_values).all() or not np.isfinite(candidate_values).all():
        raise FloatingPointError("activation shard contains non-finite values")
    if len(set(str(value) for value in request_ids)) != len(request_ids):
        raise ValueError("activation shard contains duplicate request IDs")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez(
        temporary,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=np.asarray(normalized_queries, dtype=np.str_),
        request_activations=request_values.astype(np.float16, copy=False),
        candidate_offsets=offsets,
        candidate_ids=np.asarray(candidate_ids, dtype=np.str_),
        candidate_activations=candidate_values.astype(np.float16, copy=False),
        hidden_state_indices=np.asarray(M2_HIDDEN_STATE_INDICES, dtype=np.int64),
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "request_count": len(request_ids),
        "candidate_count": len(candidate_ids),
        "first_request_id": str(request_ids[0]) if request_ids else None,
        "last_request_id": str(request_ids[-1]) if request_ids else None,
    }


def audit_activation_bundle(
    bundle_dir: str | Path,
    *,
    expected_records: Sequence[ModelRecord],
    expected_role: str,
    expected_condition: str,
    require_result_eligible: bool = True,
) -> AuditedActivationBundle:
    """Audit every shard before any evaluator is allowed to open dev qrels."""

    root = Path(bundle_dir)
    metadata = _read_json(root / "metadata.json")
    index = _read_json(root / "index.json")
    if metadata.get("schema_version") != 1 or index.get("schema_version") != 1:
        raise ValueError("unsupported M2 activation-bundle schema")
    expected_meta = {
        "analysis_stage": "m2_representation",
        "bundle_role": expected_role,
        "condition_id": expected_condition,
        "qrels_read": False,
        "request_positions": list(REQUEST_POSITIONS),
        "hidden_state_indices": list(M2_HIDDEN_STATE_INDICES),
        "preference_classifier_position": PREFERENCE_POSITION,
        "candidate_text_visible_to_preference_classifier": False,
    }
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            raise ValueError(f"activation bundle metadata mismatch: {key}")
    if metadata.get("method_id") not in M2_ANCHORS:
        raise ValueError("activation bundle method is not an M2 anchor")
    passes = metadata.get("activation_passes")
    if not isinstance(passes, dict) or passes.get("positions_share_same_forward") is not False:
        raise ValueError("activation bundle does not separate request/readout passes")
    request_pass = passes.get("request_level_query_history")
    candidate_pass = passes.get("candidate_readout_donor")
    if not isinstance(request_pass, dict) or not isinstance(candidate_pass, dict):
        raise ValueError("activation bundle pass contract is incomplete")
    if request_pass.get("context") != "prompt_only":
        raise ValueError("request-level query/history states must be prompt-only")
    if request_pass.get("causal_before_candidate_text") is not True:
        raise ValueError("request-level activation pass lacks causal attestation")
    if metadata.get("method_id") == "q3_tallrec_generalqwen":
        if candidate_pass.get("context") != "prompt_plus_fixed_yes_target_scoring_kernel":
            raise ValueError("Q3 donor pass does not match the frozen scoring kernel")
        if candidate_pass.get("q3_yes_no_target_length_equal") is not True:
            raise ValueError("Q3 donor pass lacks equal Yes/No target attestation")
        if int(candidate_pass.get("q3_target_length_tokens", 0)) <= 0:
            raise ValueError("Q3 donor pass target length is invalid")
    elif candidate_pass.get("context") != "prompt_only_frozen_scoring_kernel":
        raise ValueError("Q2 donor pass does not match the frozen scoring kernel")
    probe_identity = metadata.get("mechanism_probe_manifest")
    if not isinstance(probe_identity, dict):
        raise ValueError("activation bundle lacks mechanism manifest identity")
    if probe_identity.get("sha256") != MECHANISM_PROBE_MANIFEST_SHA256:
        raise ValueError("activation bundle mechanism manifest hash mismatch")
    if probe_identity.get("expected_sha256") != MECHANISM_PROBE_MANIFEST_SHA256:
        raise ValueError("activation bundle expected manifest hash mismatch")
    if probe_identity.get("verified") is not True:
        raise ValueError("activation bundle mechanism manifest was not verified")
    if require_result_eligible and metadata.get("result_eligible") is not True:
        raise ValueError("activation bundle is a smoke/non-result")
    if index.get("metadata_sha256") != sha256_file(root / "metadata.json"):
        raise ValueError("activation index metadata hash mismatch")

    expected_ids = [row.request_id for row in expected_records]
    if int(index.get("request_count", -1)) != len(expected_ids):
        raise ValueError("activation bundle request count mismatch")
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("activation bundle has no shards")
    observed_ids: list[str] = []
    total_candidates = 0
    hidden_size: int | None = None
    expected_candidate_ids: list[str] = []
    for record in expected_records:
        expected_candidate_ids.extend(str(row["item_id"]) for row in record.candidates)
    observed_candidate_ids: list[str] = []
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError("activation shard index row must be an object")
        relative = str(shard.get("path") or "")
        if Path(relative).name != relative or not relative.endswith(".npz"):
            raise ValueError("activation shard path must be a local .npz filename")
        path = root / "shards" / relative
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"activation shard hash mismatch: {relative}")
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "request_ids",
                "normalized_queries",
                "request_activations",
                "candidate_offsets",
                "candidate_ids",
                "candidate_activations",
                "hidden_state_indices",
            }
            if set(payload.files) != required:
                raise ValueError(f"activation shard fields mismatch: {relative}")
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            queries = [str(value) for value in payload["normalized_queries"].tolist()]
            request_values = np.asarray(payload["request_activations"])
            offsets = np.asarray(payload["candidate_offsets"], dtype=np.int64)
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
            candidate_values = np.asarray(payload["candidate_activations"])
            states = tuple(int(value) for value in payload["hidden_state_indices"].tolist())
            if states != M2_HIDDEN_STATE_INDICES:
                raise ValueError("activation hidden-state order drift")
            if request_values.ndim != 4 or request_values.shape[1:3] != (
                len(REQUEST_POSITIONS),
                len(M2_HIDDEN_STATE_INDICES),
            ):
                raise ValueError("invalid request activation shape")
            if candidate_values.ndim != 3 or candidate_values.shape[1] != len(
                M2_HIDDEN_STATE_INDICES
            ):
                raise ValueError("invalid candidate activation shape")
            if request_values.shape[0] != len(request_ids) or len(queries) != len(request_ids):
                raise ValueError("activation request arrays are misaligned")
            if offsets.shape != (len(request_ids) + 1,) or offsets[0] != 0:
                raise ValueError("activation candidate offsets are invalid")
            if np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(candidate_ids):
                raise ValueError("activation candidate offsets are inconsistent")
            if candidate_values.shape[0] != len(candidate_ids):
                raise ValueError("activation candidate arrays are misaligned")
            if not np.isfinite(request_values).all() or not np.isfinite(candidate_values).all():
                raise ValueError("activation bundle contains non-finite values")
            if any(query != normalize_query(record.query) for query, record in zip(
                queries,
                expected_records[len(observed_ids) : len(observed_ids) + len(request_ids)],
            )):
                raise ValueError("activation normalized-query identity mismatch")
            current_hidden = int(request_values.shape[-1])
            if candidate_values.shape[-1] != current_hidden:
                raise ValueError("request/candidate hidden sizes differ")
            if hidden_size is None:
                hidden_size = current_hidden
            elif hidden_size != current_hidden:
                raise ValueError("hidden size differs between activation shards")
        if int(shard.get("request_count", -1)) != len(request_ids):
            raise ValueError("activation shard request count metadata mismatch")
        if int(shard.get("candidate_count", -1)) != len(candidate_ids):
            raise ValueError("activation shard candidate count metadata mismatch")
        if ordinal == 0 and shard.get("first_request_id") != request_ids[0]:
            raise ValueError("activation shard first request metadata mismatch")
        observed_ids.extend(request_ids)
        observed_candidate_ids.extend(candidate_ids)
        total_candidates += len(candidate_ids)
    if observed_ids != expected_ids:
        raise ValueError("activation request identity/order coverage mismatch")
    if expected_role == "dev_representation":
        if metadata.get("candidate_positions") != list(CANDIDATE_POSITIONS):
            raise ValueError("dev activation bundle lacks candidate readout position")
        if observed_candidate_ids != expected_candidate_ids:
            raise ValueError("activation candidate identity/order coverage mismatch")
    elif observed_candidate_ids:
        raise ValueError("train probe activation bundle must not store candidate readouts")
    if int(index.get("candidate_count", -1)) != total_candidates:
        raise ValueError("activation index candidate count mismatch")
    if hidden_size is None or hidden_size <= 0:
        raise ValueError("activation bundle hidden size is invalid")
    return AuditedActivationBundle(
        root=root,
        metadata=metadata,
        index=index,
        request_ids=tuple(observed_ids),
        candidate_count=total_candidates,
        hidden_size=hidden_size,
    )


def load_request_activations(
    bundle: AuditedActivationBundle,
    *,
    position: str = PREFERENCE_POSITION,
) -> dict[int, np.ndarray]:
    """Load one request position for all preregistered hidden states."""

    if position not in REQUEST_POSITIONS:
        raise ValueError(f"unknown request activation position={position}")
    position_index = REQUEST_POSITIONS.index(position)
    values: list[np.ndarray] = []
    for shard in bundle.index["shards"]:
        with np.load(bundle.root / "shards" / shard["path"], allow_pickle=False) as payload:
            values.append(
                np.asarray(payload["request_activations"][:, position_index], dtype=np.float32)
            )
    matrix = np.concatenate(values, axis=0)
    if matrix.shape != (
        len(bundle.request_ids),
        len(M2_HIDDEN_STATE_INDICES),
        bundle.hidden_size,
    ):
        raise ValueError("loaded request activation matrix has invalid shape")
    return {
        state: matrix[:, index]
        for index, state in enumerate(M2_HIDDEN_STATE_INDICES)
    }


def fit_train_representation_probes(
    standardized_dir: str | Path,
    activation_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_method_id: str,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit fixed ridge probes after opening train qrels only.

    This entry point rejects smoke activation bundles.  Label vocabulary and
    the 75/25 normalized-query split are constructed on the frozen 8192 train
    requests without any dev access.
    """

    if expected_method_id not in M2_ANCHORS:
        raise ValueError("M2 fitter only supports the two frozen anchors")
    manifest_identity = load_m2_probe_manifest()
    standardized_dir = Path(standardized_dir)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    for path in (records_path, qrels_path, dataset_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = manifest_identity["frozen_inputs"]
    # Keep every train-qrels byte read behind the complete label-free activation
    # bundle audit.  Existence checks above are metadata-only; hashing is a read
    # and therefore belongs on the supervised side of this boundary.
    expected_hashes = {
        records_path: frozen["records_train_sha256"],
        dataset_manifest_path: frozen["dataset_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen M2 train input hash mismatch: {path.name}")

    selected, selection_audit = select_train_probe_records(iter_jsonl(records_path))
    bundle = audit_activation_bundle(
        activation_bundle_dir,
        expected_records=selected,
        expected_role="train_probe",
        expected_condition="full",
        require_result_eligible=True,
    )
    if bundle.metadata.get("method_id") != expected_method_id:
        raise ValueError("train activation bundle method mismatch")
    # The first train-qrels byte read occurs only after all label-free bundle
    # checks above have succeeded.
    qrels_train_sha256 = sha256_file(qrels_path)
    if qrels_train_sha256 != frozen["qrels_train_sha256"]:
        raise ValueError("frozen M2 train input hash mismatch: qrels_train.jsonl")
    qrels = _load_positive_qrels(qrels_path)
    labels, label_audit = build_preference_labels(
        selected,
        qrels,
        max_classes=M2_MAX_CLASSES,
        min_frequency=M2_MIN_CLASS_FREQUENCY,
    )
    activations = load_request_activations(bundle, position=PREFERENCE_POSITION)
    holdout = np.asarray([representation_holdout(row.query) for row in selected])
    if not holdout.any() or holdout.all():
        raise ValueError("normalized-query 75/25 split produced an empty side")
    clusters = [normalize_query(row.query) for row in selected]
    train_clusters = {value for value, is_holdout in zip(clusters, holdout) if not is_holdout}
    test_clusters = {value for value, is_holdout in zip(clusters, holdout) if is_holdout}
    if train_clusters & test_clusters:
        raise AssertionError("normalized-query clusters crossed the 75/25 split")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"probe output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    probe_rows: list[dict[str, Any]] = []
    for task in ("brand", "category"):
        task_labels = labels[task]
        eligible = np.asarray([value is not None for value in task_labels], dtype=bool)
        real = np.asarray([value or "" for value in task_labels], dtype=np.str_)
        random_values = _permuted_labels(real, eligible, task=task)
        for control, target in (("real_labels", real), ("random_labels", random_values)):
            train_mask = eligible & ~holdout
            test_mask = eligible & holdout
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                raise ValueError(f"empty {task} train/holdout population")
            for state in M2_HIDDEN_STATE_INDICES:
                readout = fit_linear_readout(activations[state][train_mask], target[train_mask])
                prediction = readout.predict(activations[state][test_mask])
                accuracy = float(np.mean(prediction == target[test_mask]))
                balanced = _balanced_accuracy(target[test_mask], prediction)
                key = f"{task}__state_{state}__{control}"
                arrays[f"{key}__classes"] = np.asarray(readout.classes, dtype=np.str_)
                arrays[f"{key}__mean"] = readout.mean.astype(np.float32)
                arrays[f"{key}__scale"] = readout.scale.astype(np.float32)
                arrays[f"{key}__coefficient"] = readout.coefficient.astype(np.float32)
                arrays[f"{key}__intercept"] = readout.intercept.astype(np.float32)
                probe_rows.append(
                    {
                        "task": task,
                        "hidden_state_index": state,
                        "label_control": control,
                        "feature_position": PREFERENCE_POSITION,
                        "candidate_text_visible": False,
                        "classes": len(readout.classes),
                        "train_requests": int(train_mask.sum()),
                        "holdout_requests": int(test_mask.sum()),
                        "train_query_clusters": len(
                            {clusters[index] for index in np.flatnonzero(train_mask)}
                        ),
                        "holdout_query_clusters": len(
                            {clusters[index] for index in np.flatnonzero(test_mask)}
                        ),
                        "holdout_accuracy": accuracy,
                        "holdout_balanced_accuracy": balanced,
                    }
                )

    weights_path = output_dir / "probe_weights.npz"
    temporary = output_dir / ".probe_weights.writing.npz"
    np.savez(temporary, **arrays)
    temporary.replace(weights_path)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "m2_representation",
        "artifact_role": "train_only_linear_probe",
        "method_id": expected_method_id,
        "mechanism_probe_manifest": manifest_identity,
        "implementation_identity": representation_probe_implementation_identity(),
        "activation_bundle_path": str(Path(activation_bundle_dir)),
        "activation_bundle_metadata_sha256": sha256_file(
            Path(activation_bundle_dir) / "metadata.json"
        ),
        "activation_bundle_index_sha256": sha256_file(
            Path(activation_bundle_dir) / "index.json"
        ),
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": qrels_train_sha256,
        "qrels_read": "train_only_after_activation_bundle_integrity",
        "dev_qrels_read": False,
        "selection_audit": selection_audit,
        "label_audit": label_audit,
        "split": {
            "unit": "normalized_query_cluster",
            "rule": "sha256(namespace, normalized_query) mod 4; bucket 0 holdout",
            "train_fraction": 0.75,
            "holdout_fraction": 0.25,
            "cluster_overlap": 0,
        },
        "classifier": {
            "type": "standardized_multiclass_ridge_linear_readout",
            "alpha": 1.0,
            "solver": "lsqr",
            "tuning": False,
            "feature_position": PREFERENCE_POSITION,
            "candidate_text_visible": False,
        },
        "hidden_state_indices": list(M2_HIDDEN_STATE_INDICES),
        "negative_controls": ["random_labels", "embedding_state_index_0"],
        "probe_rows": probe_rows,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "command": list(command or []),
        "result_eligible": True,
    }
    metadata["probe_checkpoint_id"] = (
        f"m2_{expected_method_id}@{metadata['weights_sha256'][:20]}"
    )
    _write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata


def build_preference_labels(
    records: Sequence[ModelRecord],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    max_classes: int,
    min_frequency: int,
) -> tuple[dict[str, list[str | None]], dict[str, Any]]:
    """Create deterministic best-positive brand/category targets and vocabularies."""

    if max_classes <= 1 or min_frequency <= 0:
        raise ValueError("invalid preference-label vocabulary limits")
    raw: dict[str, list[str]] = {"brand": [], "category": []}
    targets: list[dict[str, str]] = []
    missing_positive = 0
    for record in records:
        if record.request_id not in qrels:
            raise ValueError(f"missing train qrels request_id={record.request_id}")
        gains = qrels[record.request_id]
        candidate_ids = {str(row["item_id"]) for row in record.candidates}
        if set(gains) - candidate_ids:
            raise ValueError(f"train qrels contains out-of-slate item: {record.request_id}")
        values = [float(gains.get(str(row["item_id"]), 0.0)) for row in record.candidates]
        # The manifest freezes class frequency over positive brands/categories,
        # not over one outcome-selected request target.  Count every positive
        # candidate occurrence, then use the deterministic first max-gain
        # candidate as the single request-level classification label.
        for candidate, gain in zip(record.candidates, values):
            if gain <= 0:
                continue
            positive_brand = _normalize_label(candidate.get("brand"))
            positive_categories = candidate.get("cat") or []
            positive_category = _normalize_label(
                positive_categories[-1] if positive_categories else ""
            )
            if positive_brand:
                raw["brand"].append(positive_brand)
            if positive_category:
                raw["category"].append(positive_category)
        maximum = max(values, default=0.0)
        if maximum <= 0:
            targets.append({"brand": "", "category": ""})
            missing_positive += 1
            continue
        index = next(position for position, value in enumerate(values) if value == maximum)
        candidate = record.candidates[index]
        brand = _normalize_label(candidate.get("brand"))
        categories = candidate.get("cat") or []
        category = _normalize_label(categories[-1] if categories else "")
        targets.append({"brand": brand, "category": category})
    vocabularies: dict[str, tuple[str, ...]] = {}
    labels: dict[str, list[str | None]] = {}
    audit: dict[str, Any] = {"requests_without_positive": missing_positive}
    for task in ("brand", "category"):
        counts = Counter(raw[task])
        vocabulary = tuple(
            label
            for label, count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))
            if count >= min_frequency
        )[:max_classes]
        if len(vocabulary) < 2:
            raise ValueError(f"{task} preference vocabulary has fewer than two classes")
        admitted = set(vocabulary)
        values = [row[task] if row[task] in admitted else None for row in targets]
        vocabularies[task] = vocabulary
        labels[task] = values
        audit[task] = {
            "vocabulary": list(vocabulary),
            "vocabulary_size": len(vocabulary),
            "frequency_unit": "positive_candidate_occurrence",
            "request_label": "first_maximum_gain_candidate_in_frozen_candidate_order",
            "minimum_train_frequency": min_frequency,
            "maximum_classes": max_classes,
            "admitted_requests": sum(value is not None for value in values),
            "frequency": {label: counts[label] for label in vocabulary},
        }
    return labels, audit


def fit_linear_readout(matrix: np.ndarray, labels: Sequence[str]) -> LinearReadout:
    """Fit the preregistered deterministic standardized ridge classifier."""

    from sklearn.linear_model import RidgeClassifier

    values = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(labels, dtype=np.str_)
    if values.ndim != 2 or values.shape[0] != target.size or values.shape[0] < 2:
        raise ValueError("invalid linear-readout training arrays")
    if not np.isfinite(values).all():
        raise ValueError("linear-readout training features are non-finite")
    classes = tuple(str(value) for value in np.unique(target))
    if len(classes) < 2:
        raise ValueError("linear readout requires at least two classes")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (values - mean) / scale
    classifier = RidgeClassifier(alpha=1.0, fit_intercept=True, solver="lsqr", tol=1.0e-6)
    classifier.fit(standardized, target)
    coefficient = np.asarray(classifier.coef_, dtype=np.float64)
    if coefficient.ndim == 1:
        coefficient = coefficient[None, :]
    intercept = np.asarray(classifier.intercept_, dtype=np.float64).reshape(-1)
    learned_classes = tuple(str(value) for value in classifier.classes_.tolist())
    if coefficient.shape[0] == 1 and len(learned_classes) == 2:
        coefficient = np.concatenate((-coefficient, coefficient), axis=0)
        intercept = np.asarray([-intercept[0], intercept[0]], dtype=np.float64)
    if coefficient.shape != (len(learned_classes), values.shape[1]):
        raise ValueError("ridge classifier returned unexpected coefficient shape")
    return LinearReadout(
        classes=learned_classes,
        mean=mean,
        scale=scale,
        coefficient=coefficient,
        intercept=intercept,
    )


def load_fitted_readout(
    model_dir: str | Path,
    *,
    task: str,
    hidden_state_index: int,
    label_control: str = "real_labels",
) -> tuple[LinearReadout, dict[str, Any]]:
    """Load and hash-check one saved train-only probe."""

    model_dir = Path(model_dir)
    metadata = _read_json(model_dir / "metadata.json")
    weights_path = model_dir / "probe_weights.npz"
    if metadata.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("M2 probe weights changed after fitting")
    if metadata.get("dev_qrels_read") is not False:
        raise ValueError("M2 fitted probe crossed the dev qrels boundary")
    key = f"{task}__state_{int(hidden_state_index)}__{label_control}"
    with np.load(weights_path, allow_pickle=False) as payload:
        names = {
            "classes": f"{key}__classes",
            "mean": f"{key}__mean",
            "scale": f"{key}__scale",
            "coefficient": f"{key}__coefficient",
            "intercept": f"{key}__intercept",
        }
        if any(value not in payload.files for value in names.values()):
            raise ValueError(f"saved M2 probe is missing {key}")
        readout = LinearReadout(
            classes=tuple(str(value) for value in payload[names["classes"]].tolist()),
            mean=np.asarray(payload[names["mean"]], dtype=np.float64),
            scale=np.asarray(payload[names["scale"]], dtype=np.float64),
            coefficient=np.asarray(payload[names["coefficient"]], dtype=np.float64),
            intercept=np.asarray(payload[names["intercept"]], dtype=np.float64),
        )
    return readout, metadata


def _context_spans(
    method_id: str,
    context: str,
    *,
    query: str,
    history_text: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if method_id == "q2_recranker_generalqwen":
        query_prefix = "Query: "
        history_marker = "\nUser history (newest first):\n"
    elif method_id == "q3_tallrec_generalqwen":
        query_prefix = "Current query: "
        history_marker = "\nPast interactions (newest first):\n"
    else:
        raise ValueError(f"unsupported M2 method_id={method_id}")
    if not context.startswith(query_prefix):
        raise MechanicalPositionError("context_template_drift", "query prefix changed")
    query_start = len(query_prefix)
    query_end = query_start + len(query)
    if context[query_start:query_end] != query:
        raise MechanicalPositionError("query_span_mismatch", "query bytes changed")
    if not context.startswith(history_marker, query_end):
        raise MechanicalPositionError("context_template_drift", "history marker changed")
    history_start = query_end + len(history_marker)
    history_end = history_start + len(history_text)
    if context[history_start:history_end] != history_text:
        raise MechanicalPositionError(
            "history_span_mismatch", "serialized history bytes changed"
        )
    if not query or not history_text:
        raise MechanicalPositionError("empty_span", "query/history span cannot be empty")
    return (query_start, query_end), (history_start, history_end)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def _encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise MechanicalPositionError(
            "offset_mapping_unavailable", str(exc)
        ) from exc
    try:
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanicalPositionError(
            "offset_mapping_invalid", "tokenizer returned malformed offsets"
        ) from exc
    if ids != _encode(tokenizer, text) or len(ids) != len(offsets):
        raise MechanicalPositionError(
            "offset_tokenization_mismatch", "offset and encode token IDs differ"
        )
    return ids, offsets


def _token_covering_span_end(
    offsets: Sequence[tuple[int, int]], char_end: int, *, name: str
) -> int:
    if char_end <= 0:
        raise MechanicalPositionError("empty_span", f"{name} has no terminal character")
    matches = [
        index
        for index, (left, right) in enumerate(offsets)
        if left < char_end <= right and right > left
    ]
    if not matches:
        raise MechanicalPositionError(
            "offset_endpoint_unresolved",
            f"{name} expected at least one covering token, observed {matches}",
        )
    if len(matches) > 1:
        contiguous = matches == list(range(matches[0], matches[-1] + 1))
        shared_right = len({offsets[index][1] for index in matches}) == 1
        monotone_left = all(
            offsets[left_index][0] <= offsets[right_index][0]
            for left_index, right_index in zip(matches, matches[1:])
        )
        if not (contiguous and shared_right and monotone_left):
            raise MechanicalPositionError(
                "offset_endpoint_unresolved",
                f"{name} has ambiguous covering-token offsets: {matches}",
            )
    # Qwen's byte-level UTF-8 fallback may emit two or three consecutive
    # subtokens whose offsets overlap the same source character.  The causal
    # endpoint is the last tokenizer-ordered piece covering that exact terminal
    # character, as frozen in the activation metadata contract.  This is not a
    # nearby-token fallback: an uncovered endpoint still fails above.
    return matches[-1]


def _load_positive_qrels(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"empty or duplicate train qrels request_id={request_id!r}")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError("train qrels relevance must be an object")
        gains = {
            str(item_id): float(value)
            for item_id, value in relevance.items()
            if float(value) > 0
        }
        if not gains:
            gains = {str(item_id): 1.0 for item_id in row.get("clicked", [])}
            gains.update({str(item_id): 2.0 for item_id in row.get("purchased", [])})
        if any(not math.isfinite(value) or value <= 0 for value in gains.values()):
            raise ValueError(f"invalid train qrels gain: {request_id}")
        result[request_id] = gains
    if not result:
        raise ValueError("empty train qrels")
    return result


def _permuted_labels(
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    task: str,
) -> np.ndarray:
    result = labels.copy()
    indexes = np.flatnonzero(eligible).tolist()
    values = [str(labels[index]) for index in indexes]
    random.Random(_stable_int("m2_random_labels", task, str(M2_SEED))).shuffle(values)
    for index, value in zip(indexes, values):
        result[index] = value
    return result


def _balanced_accuracy(target: Sequence[str], prediction: Sequence[str]) -> float:
    target_values = np.asarray(target, dtype=np.str_)
    prediction_values = np.asarray(prediction, dtype=np.str_)
    classes = np.unique(target_values)
    return float(
        np.mean(
            [
                np.mean(prediction_values[target_values == value] == value)
                for value in classes
            ]
        )
    )


def _normalize_label(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def _stable_hex(*parts: str) -> str:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_object_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _stable_int(*parts: str) -> int:
    return int(_stable_hex(*parts), 16)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
