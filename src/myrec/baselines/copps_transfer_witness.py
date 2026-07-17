"""CoPPS-style non-LLM structural witness for preference transfer.

The witness consumes only the shared standardized JSONL interface.  A frozen
BGE feature store supplies label-free query and item vectors; the trainable
component is a small shared projection with query-aware history pooling.
Supervised ranking retains visible event/query context, while semantic
replacement views use canonical item-only title/brand/category vectors.  Two
views independently replace ``ceil(0.3 * history_length)`` selected positions
with different items from the same deepest category, choosing the most
semantically similar item from a deterministic train-only shortlist.  Training
combines listwise train-qrels supervision with an in-batch contrastive loss
between the two replacement-history profiles.

This is an independent structural witness, not an LLM baseline and not an
official CoPPS reproduction.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from myrec.baselines.frozen_text_features import (
    FrozenTextFeatureStore,
    serialize_item_semantic_content,
)
from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


METHOD_ID = "w0_copps_style_transfer_witness"
PILOT_SEED = 20260714
HISTORY_BUDGET = 6
EPOCHS = 2
BATCH_REQUESTS = 128
LEARNING_RATE = 1.0e-3
PROJECTION_DIM = 128
CONTRASTIVE_TEMPERATURE = 0.1
CONTRASTIVE_LOSS_WEIGHT = 0.2
REPLACEMENT_RATIO = 0.3
SEMANTIC_SHORTLIST_SIZE = 64
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
MAX_CONTINUOUS_JOB_SECONDS = 14_400
SAFE_EXIT_SECONDS = 13_500
CHECKPOINT_EVERY_OPTIMIZER_STEPS = 50
RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9][a-z0-9_]*$")

REQUEST_INPUT_WHITELIST = (
    "request_id",
    "user_id",
    "session_id",
    "ts",
    "query",
    "history",
    "candidates",
    "masks",
)
HISTORY_INPUT_WHITELIST = (
    "item_id",
    "title",
    "brand",
    "cat",
    "event",
    "query",
    "ts",
)
CANDIDATE_INPUT_WHITELIST = ("item_id", "title", "brand", "cat")
MASK_INPUT_WHITELIST = (
    "history_present",
    "text_coverage",
    "history_text_coverage",
    "strict_nonrepeat",
)
FORBIDDEN_MODEL_INPUTS = (
    "clicked",
    "purchased",
    "relevance",
    "is_clicked",
    "is_purchased",
    "label",
    "labels",
    "target",
)


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    feature_row: int
    category: tuple[str, ...]


@dataclass(frozen=True)
class FrozenWitnessRequest:
    request_id: str
    query_row: int
    history_rows: tuple[int, ...]
    augmented_history_rows_a: tuple[int, ...]
    augmented_history_rows_b: tuple[int, ...]
    candidate_item_ids: tuple[str, ...]
    candidate_rows: tuple[int, ...]
    positive_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class WitnessBatch:
    query_features: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    augmented_history_features_a: torch.Tensor
    augmented_history_mask_a: torch.Tensor
    augmented_history_features_b: torch.Tensor
    augmented_history_mask_b: torch.Tensor
    positive_mask: torch.Tensor | None

    def to(self, device: str | torch.device) -> "WitnessBatch":
        return WitnessBatch(
            query_features=self.query_features.to(device),
            candidate_features=self.candidate_features.to(device),
            candidate_mask=self.candidate_mask.to(device),
            history_features=self.history_features.to(device),
            history_mask=self.history_mask.to(device),
            augmented_history_features_a=self.augmented_history_features_a.to(device),
            augmented_history_mask_a=self.augmented_history_mask_a.to(device),
            augmented_history_features_b=self.augmented_history_features_b.to(device),
            augmented_history_mask_b=self.augmented_history_mask_b.to(device),
            positive_mask=(
                self.positive_mask.to(device)
                if self.positive_mask is not None
                else None
            ),
        )


def project_visible_record(
    record: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project one unified record through the frozen explicit input whitelist."""

    projected = {
        key: _copy_visible_value(record[key])
        for key in REQUEST_INPUT_WHITELIST
        if key in record and key not in {"history", "candidates", "masks"}
    }
    masks = record.get("masks", {})
    if masks is None:
        masks = {}
    if not isinstance(masks, Mapping):
        raise ValueError("masks must be an object")
    projected["masks"] = {
        key: _copy_visible_value(masks[key])
        for key in MASK_INPUT_WHITELIST
        if key in masks
    }
    history_rows = record.get("history", []) if history is None else history
    candidate_rows = record.get("candidates", [])
    if not isinstance(history_rows, (list, tuple)):
        raise ValueError("history must be a sequence")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise ValueError("candidates must be a non-empty list")
    projected["history"] = [
        _project_item(row, HISTORY_INPUT_WHITELIST) for row in history_rows
    ]
    projected["candidates"] = [
        _project_item(row, CANDIDATE_INPUT_WHITELIST) for row in candidate_rows
    ]
    return projected


def serialize_visible_item(row: Mapping[str, Any], *, history: bool) -> str:
    """Serialize only the allowed history or candidate fields."""

    whitelist = HISTORY_INPUT_WHITELIST if history else CANDIDATE_INPUT_WHITELIST
    return serialize_item_content(_project_item(row, whitelist))


class CoPPSTransferWitness(nn.Module):
    """Small shared-projection ranker with query-aware history pooling."""

    def __init__(self, content_dim: int, projection_dim: int = PROJECTION_DIM) -> None:
        super().__init__()
        if content_dim <= 0 or projection_dim <= 0:
            raise ValueError("content_dim and projection_dim must be positive")
        self.content_dim = int(content_dim)
        self.projection_dim = int(projection_dim)
        self.content_projection = nn.Linear(content_dim, projection_dim, bias=False)
        self.log_query_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_history_scale = nn.Parameter(torch.tensor(0.0))

    def history_profile(
        self,
        query_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = nn.functional.normalize(
            self.content_projection(query_features), dim=-1
        )
        history = nn.functional.normalize(
            self.content_projection(history_features), dim=-1
        )
        logits = torch.einsum("bd,bhd->bh", query, history)
        weights = torch.softmax(logits.masked_fill(~history_mask, -1.0e4), dim=-1)
        weights = weights * history_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        profile = torch.einsum("bh,bhd->bd", weights, history)
        return nn.functional.normalize(profile, dim=-1)

    def score(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = nn.functional.normalize(
            self.content_projection(query_features), dim=-1
        )
        candidates = nn.functional.normalize(
            self.content_projection(candidate_features), dim=-1
        )
        profile = self.history_profile(
            query_features, history_features, history_mask
        )
        query_scores = torch.einsum("bd,bcd->bc", query, candidates)
        history_scores = torch.einsum("bd,bcd->bc", profile, candidates)
        scores = (
            self.log_query_scale.exp() * query_scores
            + self.log_history_scale.exp() * history_scores
        )
        return scores, profile


def multilabel_listwise_loss(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability mass assigned to all train-qrels positives."""

    if scores.shape != candidate_mask.shape or scores.shape != positive_mask.shape:
        raise ValueError("score/candidate/positive mask shapes differ")
    if bool((positive_mask & ~candidate_mask).any()):
        raise ValueError("positive mask includes padded candidates")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every training request must have a positive")
    minimum = torch.finfo(scores.dtype).min
    denominator = torch.logsumexp(scores.masked_fill(~candidate_mask, minimum), dim=1)
    numerator = torch.logsumexp(scores.masked_fill(~positive_mask, minimum), dim=1)
    return (denominator - numerator).mean()


def history_view_contrastive_loss(
    left_profiles: torch.Tensor,
    right_profiles: torch.Tensor,
    eligible: torch.Tensor,
    *,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE for two replacement history views."""

    if left_profiles.shape != right_profiles.shape:
        raise ValueError("left and right profile shapes differ")
    if eligible.ndim != 1 or eligible.shape[0] != left_profiles.shape[0]:
        raise ValueError("contrastive eligibility shape differs from profiles")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    left = left_profiles[eligible]
    right = right_profiles[eligible]
    if left.shape[0] == 0:
        return left_profiles.sum() * 0.0
    left = nn.functional.normalize(left, dim=-1)
    right = nn.functional.normalize(right, dim=-1)
    if left.shape[0] == 1:
        return (1.0 - (left * right).sum(dim=-1)).mean()
    logits = left @ right.T / temperature
    targets = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, targets)
        + nn.functional.cross_entropy(logits.T, targets)
    )


def copps_training_objective(
    model: CoPPSTransferWitness,
    batch: WitnessBatch,
    *,
    temperature: float = CONTRASTIVE_TEMPERATURE,
    contrastive_weight: float = CONTRASTIVE_LOSS_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if batch.positive_mask is None:
        raise ValueError("training objective requires positive masks")
    scores, _original_profiles = model.score(
        batch.query_features,
        batch.candidate_features,
        batch.history_features,
        batch.history_mask,
    )
    augmented_profiles_a = model.history_profile(
        batch.query_features,
        batch.augmented_history_features_a,
        batch.augmented_history_mask_a,
    )
    augmented_profiles_b = model.history_profile(
        batch.query_features,
        batch.augmented_history_features_b,
        batch.augmented_history_mask_b,
    )
    ranking = multilabel_listwise_loss(
        scores, batch.candidate_mask, batch.positive_mask
    )
    eligible = (
        batch.augmented_history_mask_a.any(dim=1)
        & batch.augmented_history_mask_b.any(dim=1)
    )
    contrastive = history_view_contrastive_loss(
        augmented_profiles_a,
        augmented_profiles_b,
        eligible,
        temperature=temperature,
    )
    total = ranking + contrastive_weight * contrastive
    return total, {"ranking": ranking, "contrastive": contrastive}


def build_train_catalog(
    records_path: str | Path,
    feature_store: FrozenTextFeatureStore,
) -> tuple[dict[tuple[str, ...], tuple[CatalogItem, ...]], dict[str, int]]:
    """Build a train-visible, label-free catalog indexed by deepest category."""

    pools: dict[tuple[str, ...], dict[str, CatalogItem]] = {}
    item_observations = 0
    for raw_record in iter_jsonl(records_path):
        record = project_visible_record(raw_record)
        for rows in (record["history"], record["candidates"]):
            for row in rows:
                item_observations += 1
                category = _category_key(row)
                if not category:
                    continue
                item_id = _required_text(row, "item_id")
                feature_row = _feature_row(
                    feature_store, serialize_item_semantic_content(row)
                )
                pools.setdefault(category, {}).setdefault(
                    item_id,
                    CatalogItem(
                        item_id=item_id,
                        feature_row=feature_row,
                        category=category,
                    ),
                )
    frozen = {
        category: tuple(sorted(values.values(), key=lambda row: row.item_id))
        for category, values in pools.items()
    }
    return frozen, {
        "category_pools": len(frozen),
        "catalog_items_across_categories": sum(len(rows) for rows in frozen.values()),
        "visible_item_observations": item_observations,
    }


def select_semantic_replacement(
    source: CatalogItem,
    category_pool: Sequence[CatalogItem],
    feature_store: FrozenTextFeatureStore,
    *,
    excluded_item_ids: Iterable[str] = (),
    deterministic_key: str = "default",
    seed: int = PILOT_SEED,
    shortlist_size: int = SEMANTIC_SHORTLIST_SIZE,
) -> CatalogItem | None:
    """Choose a different-ID same-category semantic neighbor deterministically."""

    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    if any(row.category != source.category for row in category_pool):
        raise ValueError("semantic replacement pool contains a different category")
    excluded = {str(value) for value in excluded_item_ids}
    excluded.add(source.item_id)
    if len(category_pool) <= shortlist_size + 1:
        shortlist = [row for row in category_pool if row.item_id not in excluded]
    else:
        payload = (
            f"{seed}|copps-semantic-shortlist|{deterministic_key}|{source.item_id}|"
            + "\x1f".join(source.category)
        )
        start = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
        ) % len(category_pool)
        shortlist = []
        offset = 0
        while len(shortlist) < shortlist_size and offset < len(category_pool):
            row = category_pool[(start + offset) % len(category_pool)]
            offset += 1
            if row.item_id not in excluded:
                shortlist.append(row)
    if not shortlist:
        return None
    source_vector = np.asarray(
        feature_store.vectors[source.feature_row], dtype=np.float32
    )
    candidate_vectors = np.asarray(
        feature_store.vectors[[row.feature_row for row in shortlist]],
        dtype=np.float32,
    )
    similarities = candidate_vectors @ source_vector
    best = min(
        range(len(shortlist)),
        key=lambda index: (-float(similarities[index]), shortlist[index].item_id),
    )
    replacement = shortlist[best]
    if replacement.item_id in excluded:
        raise AssertionError("CoPPS replacement used an excluded sequence item ID")
    if replacement.category != source.category:
        raise AssertionError("CoPPS replacement changed the deepest category")
    return replacement


def build_training_requests(
    records_path: str | Path,
    qrels_train_path: str | Path,
    feature_store: FrozenTextFeatureStore,
    *,
    history_budget: int = HISTORY_BUDGET,
    seed: int = PILOT_SEED,
    shortlist_size: int = SEMANTIC_SHORTLIST_SIZE,
    replacement_ratio: float = REPLACEMENT_RATIO,
) -> tuple[list[FrozenWitnessRequest], dict[str, Any]]:
    """Build listwise examples and train-only semantic replacement views."""

    labels = _load_train_qrels(qrels_train_path)
    catalog, catalog_stats = build_train_catalog(records_path, feature_store)
    examples: list[FrozenWitnessRequest] = []
    seen: set[str] = set()
    skipped_no_positive = 0
    skipped_no_negative = 0
    history_events = 0
    selected_events = {"a": 0, "b": 0}
    replacement_events = {"a": 0, "b": 0}
    dropped_events = {"a": 0, "b": 0}
    requests_with_replacement = {"a": 0, "b": 0}
    selected_position_overlap = 0
    replacement_item_overlap = 0
    identical_augmented_views = 0
    identical_nonempty_augmented_views = 0
    contrastive_eligible_requests = 0
    requests_with_nonempty_history = 0
    views_without_replacement = {"a": 0, "b": 0}
    for raw_record in iter_jsonl(records_path):
        record = project_visible_record(raw_record)
        request_id = _required_text(record, "request_id")
        if request_id in seen:
            raise ValueError(f"duplicate train record request_id={request_id}")
        seen.add(request_id)
        if request_id not in labels:
            raise ValueError(f"missing train qrels request_id={request_id}")
        candidates = record["candidates"]
        candidate_ids = tuple(_required_text(row, "item_id") for row in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"duplicate candidate item for request_id={request_id}")
        unknown = labels[request_id] - set(candidate_ids)
        if unknown:
            raise ValueError(
                f"train qrels positives outside candidate slate for {request_id}: "
                f"{sorted(unknown)[:5]}"
            )
        positive_mask = tuple(item_id in labels[request_id] for item_id in candidate_ids)
        if not any(positive_mask):
            skipped_no_positive += 1
            continue
        if all(positive_mask):
            skipped_no_negative += 1
            continue
        all_original_history_ids = {
            _required_text(event, "item_id") for event in record["history"]
        }
        retained = list(record["history"][-history_budget:]) if history_budget else []
        if retained:
            requests_with_nonempty_history += 1
        original_rows: list[int] = []
        sources: list[CatalogItem] = []
        for event in retained:
            history_events += 1
            feature_row = _feature_row(
                feature_store, serialize_visible_item(event, history=True)
            )
            original_rows.append(feature_row)
            category = _category_key(event)
            sources.append(
                CatalogItem(
                    item_id=_required_text(event, "item_id"),
                    feature_row=_feature_row(
                        feature_store, serialize_item_semantic_content(event)
                    ),
                    category=category,
                )
            )
        excluded_item_ids = all_original_history_ids | set(candidate_ids)
        augmented_views: dict[str, list[int]] = {}
        selected_by_view: dict[str, set[int]] = {}
        replacements_by_view: dict[str, set[str]] = {}
        for view in ("a", "b"):
            selected = _replacement_positions(
                request_id,
                sources,
                view=view,
                replacement_ratio=replacement_ratio,
                seed=seed,
            )
            selected_by_view[view] = selected
            selected_events[view] += len(selected)
            augmented_rows: list[int] = []
            request_replacements = 0
            replacement_ids: set[str] = set()
            for index, source in enumerate(sources):
                if index not in selected:
                    augmented_rows.append(source.feature_row)
                    continue
                replacement = None
                if source.category:
                    replacement = select_semantic_replacement(
                        source,
                        catalog.get(source.category, ()),
                        feature_store,
                        excluded_item_ids=excluded_item_ids,
                        deterministic_key=f"{request_id}|view={view}|position={index}",
                        seed=seed,
                        shortlist_size=shortlist_size,
                    )
                if replacement is None:
                    dropped_events[view] += 1
                    continue
                augmented_rows.append(replacement.feature_row)
                replacement_ids.add(replacement.item_id)
                replacement_events[view] += 1
                request_replacements += 1
            if request_replacements:
                requests_with_replacement[view] += 1
            else:
                views_without_replacement[view] += 1
            replacements_by_view[view] = replacement_ids
            augmented_views[view] = augmented_rows
        selected_position_overlap += len(
            selected_by_view["a"] & selected_by_view["b"]
        )
        replacement_item_overlap += len(
            replacements_by_view["a"] & replacements_by_view["b"]
        )
        if augmented_views["a"] and augmented_views["b"]:
            contrastive_eligible_requests += 1
        if augmented_views["a"] == augmented_views["b"]:
            identical_augmented_views += 1
            if augmented_views["a"]:
                identical_nonempty_augmented_views += 1
        query = _required_text(record, "query")
        examples.append(
            FrozenWitnessRequest(
                request_id=request_id,
                query_row=_feature_row(feature_store, f"query: {query}"),
                history_rows=tuple(original_rows),
                augmented_history_rows_a=tuple(augmented_views["a"]),
                augmented_history_rows_b=tuple(augmented_views["b"]),
                candidate_item_ids=candidate_ids,
                candidate_rows=tuple(
                    _feature_row(
                        feature_store, serialize_visible_item(row, history=False)
                    )
                    for row in candidates
                ),
                positive_mask=positive_mask,
            )
        )
    if seen != set(labels):
        raise ValueError("train records and qrels have different request coverage")
    if not examples:
        raise ValueError("no labeled CoPPS witness requests were constructed")
    selected_total = sum(selected_events.values())
    replacement_total = sum(replacement_events.values())
    replacement_coverage = replacement_total / selected_total if selected_total else 0.0
    return examples, {
        **catalog_stats,
        "requests": len(seen),
        "labeled_requests": len(examples),
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_negative": skipped_no_negative,
        "retained_history_events": history_events,
        "augmentation_views_per_request": 2,
        "selected_history_events": selected_events,
        "replacement_history_events": replacement_events,
        "dropped_selected_events_no_valid_donor": dropped_events,
        "replacement_event_coverage_over_selected": replacement_coverage,
        "requests_with_replacement": requests_with_replacement,
        "requests_with_nonempty_history": requests_with_nonempty_history,
        "contrastive_eligible_requests": contrastive_eligible_requests,
        "views_without_replacement": views_without_replacement,
        "selected_position_overlap_across_views": selected_position_overlap,
        "replacement_item_overlap_across_views": replacement_item_overlap,
        "identical_augmented_views": identical_augmented_views,
        "identical_nonempty_augmented_views": identical_nonempty_augmented_views,
        "replacement_contract": {
            "catalog_scope": "records_train visible history and candidates only",
            "category": "exact deepest non-empty category path",
            "different_item_id_required": True,
            "replacement_ratio": replacement_ratio,
            "positions_per_view": "ceil(replacement_ratio * retained_history_length)",
            "views_per_request": 2,
            "donor_exclusions": "all original history item IDs and all current candidate IDs",
            "semantic_similarity": "frozen BGE cosine/dot-product",
            "semantic_feature": "canonical title/brand/category item-only text",
            "ranking_history_feature": "visible event/query/title/brand/category context",
            "shortlist": "seeded deterministic cyclic train-category shortlist",
            "shortlist_size": shortlist_size,
            "unmatched_policy": "drop event from augmented view and audit coverage",
            "seed": seed,
        },
    }


def _replacement_positions(
    request_id: str,
    sources: Sequence[CatalogItem],
    *,
    view: str,
    replacement_ratio: float,
    seed: int,
) -> set[int]:
    if not 0.0 < replacement_ratio <= 1.0:
        raise ValueError("replacement_ratio must be in (0, 1]")
    if not sources:
        return set()
    count = math.ceil(replacement_ratio * len(sources))
    ranked = sorted(
        range(len(sources)),
        key=lambda index: hashlib.sha256(
            (
                f"{seed}|copps-replacement-position|{view}|{request_id}|"
                f"{index}|{sources[index].item_id}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    return set(ranked[:count])


def collate_witness_requests(
    rows: Sequence[FrozenWitnessRequest],
    feature_store: FrozenTextFeatureStore,
) -> WitnessBatch:
    if not rows:
        raise ValueError("cannot collate an empty witness batch")
    batch_size = len(rows)
    content_dim = feature_store.dimension
    max_candidates = max(len(row.candidate_rows) for row in rows)
    max_history = max(1, max(len(row.history_rows) for row in rows))
    max_augmented_a = max(1, max(len(row.augmented_history_rows_a) for row in rows))
    max_augmented_b = max(1, max(len(row.augmented_history_rows_b) for row in rows))
    query = np.zeros((batch_size, content_dim), dtype=np.float32)
    candidates = np.zeros(
        (batch_size, max_candidates, content_dim), dtype=np.float32
    )
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    history = np.zeros((batch_size, max_history, content_dim), dtype=np.float32)
    history_mask = np.zeros((batch_size, max_history), dtype=np.bool_)
    augmented_a = np.zeros(
        (batch_size, max_augmented_a, content_dim), dtype=np.float32
    )
    augmented_mask_a = np.zeros((batch_size, max_augmented_a), dtype=np.bool_)
    augmented_b = np.zeros(
        (batch_size, max_augmented_b, content_dim), dtype=np.float32
    )
    augmented_mask_b = np.zeros((batch_size, max_augmented_b), dtype=np.bool_)
    has_labels = rows[0].positive_mask is not None
    if any((row.positive_mask is not None) != has_labels for row in rows):
        raise ValueError("mixed labeled and unlabeled requests in one batch")
    positive_mask = (
        np.zeros((batch_size, max_candidates), dtype=np.bool_) if has_labels else None
    )
    for index, row in enumerate(rows):
        query[index] = np.asarray(
            feature_store.vectors[row.query_row], dtype=np.float32
        )
        candidate_count = len(row.candidate_rows)
        candidates[index, :candidate_count] = np.asarray(
            feature_store.vectors[list(row.candidate_rows)], dtype=np.float32
        )
        candidate_mask[index, :candidate_count] = True
        if row.history_rows:
            history[index, : len(row.history_rows)] = np.asarray(
                feature_store.vectors[list(row.history_rows)], dtype=np.float32
            )
            history_mask[index, : len(row.history_rows)] = True
        if row.augmented_history_rows_a:
            augmented_a[index, : len(row.augmented_history_rows_a)] = np.asarray(
                feature_store.vectors[list(row.augmented_history_rows_a)],
                dtype=np.float32,
            )
            augmented_mask_a[index, : len(row.augmented_history_rows_a)] = True
        if row.augmented_history_rows_b:
            augmented_b[index, : len(row.augmented_history_rows_b)] = np.asarray(
                feature_store.vectors[list(row.augmented_history_rows_b)],
                dtype=np.float32,
            )
            augmented_mask_b[index, : len(row.augmented_history_rows_b)] = True
        if positive_mask is not None:
            if row.positive_mask is None:
                raise AssertionError("unreachable mixed-label batch")
            positive_mask[index, :candidate_count] = row.positive_mask
    return WitnessBatch(
        query_features=torch.from_numpy(query),
        candidate_features=torch.from_numpy(candidates),
        candidate_mask=torch.from_numpy(candidate_mask),
        history_features=torch.from_numpy(history),
        history_mask=torch.from_numpy(history_mask),
        augmented_history_features_a=torch.from_numpy(augmented_a),
        augmented_history_mask_a=torch.from_numpy(augmented_mask_a),
        augmented_history_features_b=torch.from_numpy(augmented_b),
        augmented_history_mask_b=torch.from_numpy(augmented_mask_b),
        positive_mask=(
            torch.from_numpy(positive_mask) if positive_mask is not None else None
        ),
    )


def audit_witness_candidate_scores(
    model: CoPPSTransferWitness,
    rows: Sequence[FrozenWitnessRequest],
    feature_store: FrozenTextFeatureStore,
    *,
    device: str | torch.device,
    max_requests: int = 256,
) -> dict[str, Any]:
    """Run a label-free finite/nonconstant checkpoint sanity audit."""

    selected = list(rows[:max_requests])
    if not selected:
        raise ValueError("cannot audit W0 scores without requests")
    was_training = model.training
    model.eval()
    batch = collate_witness_requests(selected, feature_store).to(device)
    with torch.inference_mode():
        scores, _ = model.score(
            batch.query_features,
            batch.candidate_features,
            batch.history_features,
            batch.history_mask,
        )
    finite = bool(torch.isfinite(scores[batch.candidate_mask]).all())
    spans = []
    values = []
    for index in range(len(selected)):
        row_scores = scores[index][batch.candidate_mask[index]]
        values.append(row_scores)
        spans.append(float((row_scores.max() - row_scores.min()).detach().cpu()))
    flat = torch.cat(values)
    score_std = float(flat.float().std(unbiased=False).detach().cpu())
    nonconstant = sum(span > 1.0e-8 for span in spans)
    if was_training:
        model.train()
    return {
        "requests": len(selected),
        "finite_scores": finite,
        "nonconstant_candidate_requests": nonconstant,
        "score_standard_deviation": score_std,
        "passed": finite and nonconstant > 0 and score_std > 0.0,
    }


def train_copps_transfer_witness(
    standardized_dir: str | Path,
    feature_store_dir: str | Path,
    output_model_dir: str | Path,
    run_id: str,
    *,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    device: str = "cuda:0",
    resume: bool = False,
    seed: int = PILOT_SEED,
    history_budget: int = HISTORY_BUDGET,
    epochs: int = EPOCHS,
    batch_requests: int = BATCH_REQUESTS,
    learning_rate: float = LEARNING_RATE,
    projection_dim: int = PROJECTION_DIM,
    contrastive_temperature: float = CONTRASTIVE_TEMPERATURE,
    contrastive_loss_weight: float = CONTRASTIVE_LOSS_WEIGHT,
    replacement_ratio: float = REPLACEMENT_RATIO,
    semantic_shortlist_size: int = SEMANTIC_SHORTLIST_SIZE,
    safe_exit_seconds: int = SAFE_EXIT_SECONDS,
    command: Sequence[str] | None = None,
    max_optimizer_steps_this_job: int | None = None,
) -> dict[str, Any]:
    """Train or resume the frozen V1.2 W0 recipe using train qrels only."""

    started = time.perf_counter()
    _validate_run_id(run_id)
    implementation_identity = _implementation_identity()
    config = _load_witness_config(config_path) if config_path is not None else None
    _assert_frozen_recipe(
        seed=seed,
        history_budget=history_budget,
        epochs=epochs,
        batch_requests=batch_requests,
        learning_rate=learning_rate,
        projection_dim=projection_dim,
        contrastive_temperature=contrastive_temperature,
        contrastive_loss_weight=contrastive_loss_weight,
        replacement_ratio=replacement_ratio,
        semantic_shortlist_size=semantic_shortlist_size,
    )
    if not 0 < safe_exit_seconds <= MAX_CONTINUOUS_JOB_SECONDS:
        raise ValueError("safe_exit_seconds is outside the frozen job boundary")
    if max_optimizer_steps_this_job is not None and max_optimizer_steps_this_job <= 0:
        raise ValueError("max_optimizer_steps_this_job must be positive")
    standardized_dir = Path(standardized_dir)
    feature_store_dir = Path(feature_store_dir)
    output_model_dir = Path(output_model_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    if resume:
        if not (output_model_dir / "training_state.pt").exists():
            raise FileNotFoundError("resume requires output_model_dir/training_state.pt")
    elif output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    records_train = standardized_dir / "records_train.jsonl"
    qrels_train = standardized_dir / "qrels_train.jsonl"
    feature_store = FrozenTextFeatureStore(
        feature_store_dir, require_fingerprints=True
    )
    _assert_feature_store_contract(feature_store, required_records=(records_train,))
    dataset_manifest = _load_dataset_manifest(standardized_dir)
    if config is not None:
        _assert_frozen_train_inputs(
            config=config,
            standardized_dir=standardized_dir,
            feature_store_dir=feature_store_dir,
            feature_store=feature_store,
            dataset_manifest=dataset_manifest,
        )
    examples, augmentation_stats = build_training_requests(
        records_train,
        qrels_train,
        feature_store,
        history_budget=history_budget,
        seed=seed,
        shortlist_size=semantic_shortlist_size,
        replacement_ratio=replacement_ratio,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    recipe = _frozen_recipe()
    training_contract = {
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "records_train_sha256": sha256_file(records_train),
        "qrels_train_sha256": sha256_file(qrels_train),
        "config_sha256": config["_config_sha256"] if config is not None else None,
        "protocol_sha256": (
            config["protocol"]["sha256"] if config is not None else None
        ),
        "implementation_digest": implementation_identity["digest"],
        "feature_contract": feature_store.metadata["feature_contract"],
        "visible_text_contract": feature_store.metadata["visible_text_contract"],
        "feature_model": feature_store.metadata.get("model_name_or_path"),
        "feature_store_metadata_sha256": sha256_file(
            feature_store_dir / "metadata.json"
        ),
        "encoder_fingerprint_sha256": feature_store.encoder_fingerprint_sha256,
        "store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
        "content_dim": feature_store.dimension,
        "recipe": recipe,
    }

    _set_random_seed(seed)
    model = CoPPSTransferWitness(
        content_dim=feature_store.dimension, projection_dim=projection_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    batches_per_epoch = math.ceil(len(examples) / batch_requests)
    total_steps = batches_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine_lambda(total_steps, WARMUP_RATIO),
    )
    start_epoch = 0
    start_batch = 0
    optimizer_steps = 0
    losses: list[dict[str, float]] = []
    resumed_from: str | None = None
    prior_run_lineage: list[str] = []
    cumulative_elapsed_before = 0.0
    if resume:
        state_path = output_model_dir / "training_state.pt"
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("training_contract") != training_contract:
            raise ValueError("resume training contract differs from checkpoint")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["batch_cursor"])
        optimizer_steps = int(state["optimizer_steps"])
        losses = list(state.get("losses", []))
        prior_run_lineage = [str(value) for value in state.get("run_lineage", [])]
        cumulative_elapsed_before = float(
            state.get("cumulative_elapsed_seconds", 0.0)
        )
        if run_id in prior_run_lineage:
            raise ValueError("resume run_id already exists in checkpoint lineage")
        _restore_rng_state(state.get("rng_state", {}))
        resumed_from = sha256_file(state_path)

    status = "completed"
    next_epoch = start_epoch
    next_batch = start_batch
    optimizer_steps_at_job_start = optimizer_steps
    run_lineage = [*prior_run_lineage, run_id]

    def save_state() -> None:
        _save_training_state(
            output_model_dir / "training_state.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=next_epoch,
            batch_cursor=next_batch,
            optimizer_steps=optimizer_steps,
            losses=losses,
            training_contract=training_contract,
            run_lineage=run_lineage,
            cumulative_elapsed_seconds=(
                cumulative_elapsed_before + time.perf_counter() - started
            ),
        )

    model.train()
    for epoch in range(start_epoch, epochs):
        order = _epoch_order(len(examples), seed=seed, epoch=epoch)
        first_batch = start_batch if epoch == start_epoch else 0
        for batch_index in range(first_batch, batches_per_epoch):
            indices = order[
                batch_index * batch_requests : (batch_index + 1) * batch_requests
            ]
            batch = collate_witness_requests(
                [examples[int(index)] for index in indices], feature_store
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, components = copps_training_objective(
                model,
                batch,
                temperature=contrastive_temperature,
                contrastive_weight=contrastive_loss_weight,
            )
            if not bool(torch.isfinite(total)):
                raise ValueError("non-finite CoPPS witness loss")
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("non-finite CoPPS witness gradient")
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            losses.append(
                {
                    "epoch": epoch,
                    "batch": batch_index,
                    "total": float(total.detach().cpu()),
                    "ranking": float(components["ranking"].detach().cpu()),
                    "contrastive": float(
                        components["contrastive"].detach().cpu()
                    ),
                }
            )
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch == batches_per_epoch:
                next_epoch = epoch + 1
                next_batch = 0
            if optimizer_steps % CHECKPOINT_EVERY_OPTIMIZER_STEPS == 0:
                save_state()
            if (
                max_optimizer_steps_this_job is not None
                and optimizer_steps - optimizer_steps_at_job_start
                >= max_optimizer_steps_this_job
            ):
                status = "pending_step_limit"
                break
            if time.perf_counter() - started >= safe_exit_seconds:
                status = "pending_safe_exit"
                break
        save_state()
        if status != "completed":
            break
    if next_epoch < epochs and status == "completed":
        raise AssertionError("W0 training ended early without a pending status")
    save_state()
    nondegeneracy = audit_witness_candidate_scores(
        model, examples, feature_store, device=device
    )
    if status == "completed" and not nondegeneracy["passed"]:
        status = "mechanical_failure_degenerate_scores"

    model_path = output_model_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    elapsed = time.perf_counter() - started
    model_config = {
        "content_dim": feature_store.dimension,
        "projection_dim": projection_dim,
    }
    checkpoint_id = f"{METHOD_ID}@{sha256_file(model_path)[:20]}"
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "status": status,
        "seed": seed,
        "evidence_mode": "first_round_pilot",
        "role": "non_llm_structural_transfer_witness_outside_main_table",
        "reproduction_boundary": "independent_CoPPS_style_reimplementation_not_official",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "model_config": model_config,
        "recipe": recipe,
        "objective": (
            "multi_positive_listwise_train_qrels + 0.2 * symmetric_in_batch_"
            "InfoNCE(two different_ID_same_category_semantic_history_views)"
        ),
        "input_field_whitelist": _input_whitelist_metadata(),
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "model_input_fields_used": [
            "query",
            "history.title",
            "history.brand",
            "history.cat",
            "history.event",
            "history.query",
            "candidates.title",
            "candidates.brand",
            "candidates.cat",
        ],
        "feature_store": {
            "path": str(feature_store_dir),
            "metadata_sha256": sha256_file(feature_store_dir / "metadata.json"),
            "feature_contract": feature_store.metadata["feature_contract"],
            "visible_text_contract": feature_store.metadata["visible_text_contract"],
            "model_name_or_path": feature_store.metadata.get("model_name_or_path"),
            "encoder_fingerprint": feature_store.metadata["encoder_fingerprint"],
            "encoder_fingerprint_sha256": feature_store.encoder_fingerprint_sha256,
            "store_fingerprint": feature_store.metadata["store_fingerprint"],
            "store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
            "qrels_read": feature_store.metadata.get("qrels_read"),
            "trainable": False,
        },
        "augmentation": augmentation_stats,
        "nondegeneracy_audit": nondegeneracy,
        "training": {
            "requests": len(examples),
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds_this_job": elapsed,
            "cumulative_elapsed_seconds": cumulative_elapsed_before + elapsed,
            "run_lineage": run_lineage,
            "loss_rows": len(losses),
            "last_loss": losses[-1] if losses else None,
            "resumed": resume,
            "resumed_from_training_state_sha256": resumed_from,
            "next_epoch": next_epoch,
            "next_batch_cursor": next_batch,
            "safe_exit_seconds": safe_exit_seconds,
            "max_continuous_job_seconds": MAX_CONTINUOUS_JOB_SECONDS,
            "max_optimizer_steps_this_job": max_optimizer_steps_this_job,
        },
        "records_train_path": str(records_train),
        "records_train_sha256": sha256_file(records_train),
        "training_qrels_path": str(qrels_train),
        "training_qrels_sha256": sha256_file(qrels_train),
        "qrels_read": True,
        "qrels_scope": "qrels_train_only",
        "dev_qrels_read": False,
        "confirmation_qrels_read": False,
        "test_qrels_read": False,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "training_state_path": str(output_model_dir / "training_state.pt"),
        "training_state_sha256": sha256_file(
            output_model_dir / "training_state.pt"
        ),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": config["_config_sha256"] if config is not None else None,
        "protocol_path": (
            config["protocol"]["path"] if config is not None else None
        ),
        "protocol_sha256": (
            config["protocol"]["sha256"] if config is not None else None
        ),
        "implementation_identity": implementation_identity,
        "implementation_digest": implementation_identity["digest"],
        "implementation_sha256": sha256_file(Path(__file__)),
        "command": list(command) if command is not None else None,
        "code_revision": _code_revision_metadata(),
        "environment": _runtime_metadata(device),
        "input_manifests": _manifest_hash_metadata(standardized_dir),
    }
    write_json(output_model_dir / "metadata.json", metadata)
    write_json(run_dir / "metadata.json", metadata)
    _copy_config(config_path, run_dir)
    return metadata


def write_copps_transfer_witness_scores(
    standardized_dir: str | Path,
    feature_store_dir: str | Path,
    checkpoint_dir: str | Path,
    history_assignments_path: str | Path,
    run_id: str,
    *,
    history_condition: str,
    split: str = "dev",
    runs_dir: str | Path = "runs",
    device: str = "cuda:0",
    batch_requests: int = BATCH_REQUESTS,
    config_path: str | Path | None,
    command: Sequence[str] | None = None,
    _test_only_allow_unfrozen_config: bool = False,
) -> dict[str, Any]:
    """Score complete candidates for one full/null/wrong assignment, label-free."""

    _validate_run_id(run_id)
    implementation_identity = _implementation_identity()
    if config_path is None and not _test_only_allow_unfrozen_config:
        raise ValueError("production W0 scoring requires a frozen config_path")
    config = _load_witness_config(config_path) if config_path is not None else None
    normalized_condition = "true" if history_condition == "full" else history_condition
    if normalized_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    if split not in {"dev", "confirmation"}:
        raise ValueError("CoPPS witness scoring supports dev or confirmation only")
    if batch_requests <= 0:
        raise ValueError("batch_requests must be positive")
    standardized_dir = Path(standardized_dir)
    feature_store_dir = Path(feature_store_dir)
    checkpoint_dir = Path(checkpoint_dir)
    assignments_path = Path(history_assignments_path)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    checkpoint_metadata_path = checkpoint_dir / "metadata.json"
    with checkpoint_metadata_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if checkpoint.get("status") != "completed":
        raise ValueError("only a completed W0 checkpoint may be scored")
    if checkpoint.get("method_id") != METHOD_ID:
        raise ValueError("checkpoint is not the W0 CoPPS-style witness")
    if config is not None and checkpoint.get("config_sha256") != config[
        "_config_sha256"
    ]:
        raise ValueError("W0 scoring config differs from the training checkpoint")
    if config is not None and checkpoint.get("protocol_sha256") != config[
        "protocol"
    ]["sha256"]:
        raise ValueError("W0 scoring protocol differs from the training checkpoint")
    checkpoint_implementation = checkpoint.get("implementation_digest")
    checkpoint_identity = checkpoint.get("implementation_identity")
    if not isinstance(checkpoint_identity, Mapping) or (
        checkpoint_identity.get("digest") != checkpoint_implementation
    ):
        raise ValueError("W0 checkpoint lacks a self-consistent implementation identity")
    if checkpoint_implementation != implementation_identity["digest"]:
        raise ValueError("W0 scoring implementation differs from the training checkpoint")
    feature_store = FrozenTextFeatureStore(
        feature_store_dir, require_fingerprints=True
    )
    records_path = standardized_dir / f"records_{split}.jsonl"
    _assert_feature_store_contract(feature_store, required_records=(records_path,))
    feature_store_compatibility = _assert_scoring_feature_store_compatible(
        feature_store, checkpoint
    )

    assignments = _load_history_assignments(
        assignments_path, expected_condition=normalized_condition
    )
    assignment_manifest = _load_history_assignment_manifest(
        assignments_path,
        expected_condition=normalized_condition,
        records_path=records_path,
        assignment_count=len(assignments),
    )
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    expected_candidates, candidate_dataset_version = _load_candidate_contract(
        candidate_manifest_path, split=split
    )
    expected_requests, request_dataset_version = _load_request_contract(
        request_manifest_path, split=split
    )
    dataset_manifest = _load_dataset_manifest(standardized_dir)
    if candidate_dataset_version != dataset_manifest["dataset_version"]:
        raise ValueError("candidate manifest dataset version mismatch")
    if request_dataset_version != dataset_manifest["dataset_version"]:
        raise ValueError("request manifest dataset version mismatch")
    if set(expected_requests) != set(expected_candidates):
        raise ValueError("candidate and request manifest coverage differs")
    model_path = checkpoint_dir / "model.pt"
    declared_model_sha256 = str(checkpoint.get("model_sha256", ""))
    if len(declared_model_sha256) != 64 or sha256_file(model_path) != declared_model_sha256:
        raise ValueError("W0 checkpoint model SHA-256 differs from metadata")
    expected_checkpoint_id = f"{METHOD_ID}@{declared_model_sha256[:20]}"
    if checkpoint.get("checkpoint_id") != expected_checkpoint_id:
        raise ValueError("W0 checkpoint_id differs from the model SHA-256")
    holdout_integrity = None
    if config is not None:
        holdout_integrity = _assert_frozen_scoring_population(
            config=config,
            standardized_dir=standardized_dir,
            records_path=records_path,
            dataset_manifest=dataset_manifest,
            split=split,
            history_condition=normalized_condition,
            checkpoint=checkpoint,
            checkpoint_metadata_path=checkpoint_metadata_path,
            model_path=model_path,
            model_sha256=declared_model_sha256,
            implementation_digest=implementation_identity["digest"],
        )
    model = CoPPSTransferWitness(**checkpoint["model_config"]).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    run_dir.mkdir(parents=True, exist_ok=False)
    scores_path = run_dir / "scores.jsonl"
    pending: list[FrozenWitnessRequest] = []
    seen: set[str] = set()
    request_count = 0
    score_rows = 0
    finite_scores = True
    request_score_ranges: list[float] = []
    started = time.perf_counter()

    def flush(handle: Any) -> None:
        nonlocal request_count, score_rows, finite_scores
        if not pending:
            return
        batch = collate_witness_requests(pending, feature_store).to(device)
        with torch.inference_mode():
            values, _ = model.score(
                batch.query_features,
                batch.candidate_features,
                batch.history_features,
                batch.history_mask,
            )
        values = values.cpu()
        for row_index, request in enumerate(pending):
            request_values = values[
                row_index, : len(request.candidate_item_ids)
            ].float()
            request_score_ranges.append(
                float((request_values.max() - request_values.min()).item())
            )
            for candidate_index, item_id in enumerate(request.candidate_item_ids):
                value = float(values[row_index, candidate_index])
                if not math.isfinite(value):
                    finite_scores = False
                    raise ValueError(
                        f"non-finite score for {request.request_id} {item_id}"
                    )
                handle.write(
                    json.dumps(
                        {
                            "request_id": request.request_id,
                            "candidate_item_id": item_id,
                            "score": value,
                            "method_id": METHOD_ID,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                score_rows += 1
            request_count += 1
        pending.clear()

    with scores_path.open("w", encoding="utf-8") as handle:
        for raw_record in iter_jsonl(records_path):
            request_id = str(raw_record["request_id"])
            if request_id in seen:
                raise ValueError(f"duplicate scored request_id={request_id}")
            if request_id not in assignments:
                raise ValueError(f"assignment missing request_id={request_id}")
            seen.add(request_id)
            source_record = project_visible_record(raw_record)
            assigned_history = _validate_assigned_history(
                raw_record,
                source_record,
                assignments[request_id],
                history_condition=normalized_condition,
            )
            record = {**source_record, "history": assigned_history}
            candidate_ids = tuple(
                _required_text(row, "item_id") for row in record["candidates"]
            )
            if request_id not in expected_candidates:
                raise ValueError(f"candidate manifest missing request_id={request_id}")
            if candidate_ids != expected_candidates[request_id]:
                raise ValueError(
                    f"candidate identity/order mismatch for request_id={request_id}"
                )
            raw_query_text = str(raw_record.get("query", ""))
            query_text = _required_text(record, "query")
            expected_query_hash, expected_candidate_hash = expected_requests[request_id]
            # The request manifest binds the exact serialized query, including
            # leading/trailing whitespace.  W0 intentionally strips that
            # whitespace only for the frozen encoder text contract, so identity
            # verification must use the raw unified-record value.
            if sha256_text(raw_query_text) != expected_query_hash:
                raise ValueError(f"request query hash mismatch for request_id={request_id}")
            observed_candidate_hash = sha256_text(
                json.dumps(list(candidate_ids), separators=(",", ":"))
            )
            if observed_candidate_hash != expected_candidate_hash:
                raise ValueError(
                    f"request candidate hash mismatch for request_id={request_id}"
                )
            retained = list(record["history"][-HISTORY_BUDGET:])
            pending.append(
                FrozenWitnessRequest(
                    request_id=request_id,
                    query_row=_feature_row(
                        feature_store, f"query: {query_text}"
                    ),
                    history_rows=tuple(
                        _feature_row(
                            feature_store,
                            serialize_visible_item(event, history=True),
                        )
                        for event in retained
                    ),
                    augmented_history_rows_a=(),
                    augmented_history_rows_b=(),
                    candidate_item_ids=candidate_ids,
                    candidate_rows=tuple(
                        _feature_row(
                            feature_store,
                            serialize_visible_item(candidate, history=False),
                        )
                        for candidate in record["candidates"]
                    ),
                )
            )
            if len(pending) >= batch_requests:
                flush(handle)
        flush(handle)
    if seen != set(assignments):
        raise ValueError("assignment and scored request coverage differ")
    if seen != set(expected_candidates):
        raise ValueError("records and candidate manifest request coverage differ")
    expected_rows = sum(len(values) for values in expected_candidates.values())
    if score_rows != expected_rows:
        raise AssertionError("score rows do not exactly cover candidate manifest")
    nonconstant_requests = sum(
        value > 1.0e-8 for value in request_score_ranges
    )
    if nonconstant_requests == 0:
        shutil.rmtree(run_dir)
        raise ValueError(
            "uncapped W0 score run is globally degenerate at the frozen 1e-8 threshold"
        )

    feature_metadata_hash = sha256_file(feature_store_dir / "metadata.json")
    runtime_metadata = _runtime_metadata(device)
    scoring_signature = {
        "batch_requests": batch_requests,
        "method": METHOD_ID,
        "model_config": checkpoint["model_config"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_model_sha256": declared_model_sha256,
        "config_sha256": config["_config_sha256"] if config is not None else None,
        "protocol_sha256": (
            config["protocol"]["sha256"] if config is not None else None
        ),
        "implementation_digest": implementation_identity["digest"],
        "inference_dtype": "float32",
        "request_aligned_batches": True,
        "runtime_identity": {
            "python": runtime_metadata["python"],
            "platform": runtime_metadata["platform"],
            "numpy": runtime_metadata["numpy"],
            "torch": runtime_metadata["torch"],
            "device": runtime_metadata["device"],
            "cuda_version": runtime_metadata["cuda_version"],
            "cuda_device_name": runtime_metadata.get("cuda_device_name"),
        },
        "score_definition": (
            "exp(query_scale)*projected_query_candidate_cosine + "
            "exp(history_scale)*query_attended_history_candidate_cosine"
        ),
        "history_budget": HISTORY_BUDGET,
        "content_feature_contract": feature_store.metadata["feature_contract"],
        "content_visible_text_contract": feature_store.metadata["visible_text_contract"],
        "content_feature_model": feature_store.metadata.get("model_name_or_path"),
        "content_encoder_fingerprint_sha256": feature_store.encoder_fingerprint_sha256,
        "content_store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
        "content_feature_store_metadata_sha256": feature_metadata_hash,
        "input_field_whitelist": _input_whitelist_metadata(),
        "checkpoint_identity_manifest_sha256": (
            holdout_integrity["checkpoint_identity_manifest_sha256"]
            if holdout_integrity is not None
            else None
        ),
        "holdout_integrity_lock_sha256": (
            holdout_integrity["integrity_lock_sha256"]
            if holdout_integrity is not None
            else None
        ),
        "holdout_release_lock_sha256": (
            holdout_integrity[
                "post_selection_recipe_checkpoint_lock_sha256"
            ]
            if holdout_integrity is not None
            else None
        ),
    }
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": METHOD_ID,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_model_sha256": declared_model_sha256,
        "checkpoint_metadata_sha256": sha256_file(checkpoint_metadata_path),
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "seed": checkpoint["seed"],
        "evidence_mode": "first_round_pilot",
        "split": split,
        "history_condition": normalized_condition,
        "history_view": "full" if normalized_condition == "true" else normalized_condition,
        "history_assignments_path": str(assignments_path),
        "history_assignment_sha256": sha256_file(assignments_path),
        "history_assignment_manifest_path": assignment_manifest["path"],
        "history_assignment_manifest_sha256": assignment_manifest["sha256"],
        "history_assignment_semantics_verified": True,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "scoring_signature": scoring_signature,
        "holdout_integrity": holdout_integrity,
        "qrels_read": False,
        "request_count": request_count,
        "score_rows": score_rows,
        "scores_sha256": sha256_file(scores_path),
        "score_non_degeneracy": {
            "max_request_range": max(request_score_ranges),
            "mean_request_range": sum(request_score_ranges)
            / len(request_score_ranges),
            "nonconstant_requests_at_1e_8": nonconstant_requests,
            "threshold": 1.0e-8,
        },
        "coverage": {
            "expected_requests": len(expected_candidates),
            "actual_requests": request_count,
            "expected_candidate_scores": expected_rows,
            "actual_candidate_scores": score_rows,
            "complete_candidate_coverage": True,
            "finite_scores": finite_scores,
        },
        "input_field_whitelist": _input_whitelist_metadata(),
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "feature_store": {
            "path": str(feature_store_dir),
            "metadata_sha256": feature_metadata_hash,
            "encoder_fingerprint_sha256": feature_store.encoder_fingerprint_sha256,
            "store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
            "qrels_read": feature_store.metadata.get("qrels_read"),
            "trainable": False,
        },
        "feature_store_compatibility": feature_store_compatibility,
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": config["_config_sha256"] if config is not None else None,
        "protocol_path": (
            config["protocol"]["path"] if config is not None else None
        ),
        "protocol_sha256": (
            config["protocol"]["sha256"] if config is not None else None
        ),
        "implementation_identity": implementation_identity,
        "implementation_digest": implementation_identity["digest"],
        "implementation_sha256": sha256_file(Path(__file__)),
        "test_only_unfrozen_config": _test_only_allow_unfrozen_config,
        "role": "non_llm_structural_transfer_witness_outside_main_table",
        "command": list(command) if command is not None else None,
        "code_revision": _code_revision_metadata(),
        "environment": runtime_metadata,
        "input_manifests": _manifest_hash_metadata(standardized_dir),
    }
    write_json(run_dir / "metadata.json", metadata)
    _copy_config(config_path, run_dir)
    return metadata


def _project_item(row: Mapping[str, Any], whitelist: Iterable[str]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("item/history row must be an object")
    return {
        key: _copy_visible_value(row[key]) for key in whitelist if key in row
    }


def _copy_visible_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_visible_value(row) for row in value]
    if isinstance(value, dict):
        return {str(key): _copy_visible_value(row) for key, row in value.items()}
    return value


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing or empty {key}")
    return value


def _category_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("cat", [])
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return tuple(
        text
        for value in raw
        if (text := str(value).strip()) and text.upper() != "UNKNOWN"
    )


def _feature_row(feature_store: FrozenTextFeatureStore, text: str) -> int:
    digest = sha256_text(text)
    try:
        return int(feature_store.hash_to_row[digest])
    except KeyError as exc:
        raise KeyError(
            f"visible text is absent from frozen feature store: {digest}"
        ) from exc


def _load_train_qrels(path: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if path.name != "qrels_train.jsonl":
        raise ValueError("W0 training may read only qrels_train.jsonl")
    result: dict[str, set[str]] = {}
    for row in iter_jsonl(path):
        request_id = _required_text(row, "request_id")
        if request_id in result:
            raise ValueError(f"duplicate train qrels request_id={request_id}")
        positives = {
            *(str(value) for value in row.get("clicked", [])),
            *(str(value) for value in row.get("purchased", [])),
        }
        relevance = row.get("relevance", {})
        if isinstance(relevance, Mapping):
            positives.update(
                str(item_id)
                for item_id, gain in relevance.items()
                if float(gain) > 0
            )
        result[request_id] = positives
    if not result:
        raise ValueError("qrels_train.jsonl is empty")
    return result


def _load_history_assignments(
    path: Path, *, expected_condition: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        request_id = _required_text(row, "request_id")
        if request_id in result:
            raise ValueError(f"duplicate assignment request_id={request_id}")
        assignment = str(row.get("assignment", ""))
        if assignment == "full":
            assignment = "true"
        if assignment != expected_condition:
            raise ValueError(f"assignment condition mismatch for {request_id}")
        history = row.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"assignment history is not a list for {request_id}")
        result[request_id] = {
            "donor_request_id": row.get("donor_request_id"),
            "donor_user_id": row.get("donor_user_id"),
            "match_type": row.get("match_type"),
            "history": [
                _project_item(event, HISTORY_INPUT_WHITELIST) for event in history
            ],
        }
    if not result:
        raise ValueError("history assignment file is empty")
    return result


def _load_history_assignment_manifest(
    assignments_path: Path,
    *,
    expected_condition: str,
    records_path: Path,
    assignment_count: int,
) -> dict[str, Any]:
    """Verify the qrels-free assignment artifact and its source population."""

    manifest_path = assignments_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"history assignment manifest is required: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("qrels_read") is not False or manifest.get(
        "model_scores_read"
    ) is not False:
        raise ValueError("history assignment manifest crossed the label/model boundary")
    file_entry = manifest.get("files", {}).get(expected_condition, {})
    if file_entry.get("sha256") != sha256_file(assignments_path):
        raise ValueError("history assignment file differs from its manifest")
    if manifest.get("source_records_sha256") != sha256_file(records_path):
        raise ValueError("history assignment source records hash mismatch")
    if int(manifest.get("requests", -1)) != assignment_count:
        raise ValueError("history assignment manifest request count mismatch")
    if int(manifest.get("target_candidate_leakage_violations", -1)) != 0:
        raise ValueError("history assignment candidate leakage audit failed")
    if int(manifest.get("history_not_strictly_before_target_violations", -1)) != 0:
        raise ValueError("history assignment causality audit failed")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "evidence_mode": manifest.get("evidence_mode"),
        "seed": manifest.get("seed"),
    }


def _validate_assigned_history(
    raw_record: Mapping[str, Any],
    record: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    history_condition: str,
) -> list[dict[str, Any]]:
    """Enforce full/null/wrong semantics independently for every request."""

    history = list(assignment["history"])
    request_id = _required_text(record, "request_id")
    if history_condition == "true":
        if history != list(record["history"]):
            raise ValueError(f"true assignment differs from record history: {request_id}")
        return history
    if history_condition == "null":
        if history:
            raise ValueError(f"null assignment is non-empty: {request_id}")
        return history
    if history_condition != "wrong":
        raise ValueError(f"unsupported assignment condition={history_condition}")

    candidate_ids = {
        _required_text(candidate, "item_id") for candidate in record["candidates"]
    }
    if any(_required_text(event, "item_id") in candidate_ids for event in history):
        raise ValueError(f"wrong history contains a target candidate: {request_id}")
    try:
        request_ts = int(raw_record["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target request lacks an integer timestamp: {request_id}") from exc
    for event in history:
        try:
            event_ts = int(event["ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"wrong history event lacks an integer timestamp: {request_id}"
            ) from exc
        if event_ts >= request_ts:
            raise ValueError(f"wrong history is not causal: {request_id}")
    if history:
        donor_user_id = str(assignment.get("donor_user_id") or "").strip()
        target_user_id = _required_text(raw_record, "user_id")
        if not donor_user_id or donor_user_id == target_user_id:
            raise ValueError(f"wrong history is not cross-user: {request_id}")
        donor_request_id = str(assignment.get("donor_request_id") or "").strip()
        if not donor_request_id or donor_request_id == request_id:
            raise ValueError(f"wrong history reused the target request: {request_id}")
    return history


def _load_candidate_contract(
    path: Path, *, split: str
) -> tuple[dict[str, tuple[str, ...]], str]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result: dict[str, tuple[str, ...]] = {}
    for row in manifest.get("entries", []):
        if row.get("split") != split:
            continue
        request_id = str(row["request_id"])
        candidate_ids = tuple(str(value) for value in row["candidate_item_ids"])
        if request_id in result:
            raise ValueError(f"duplicate candidate manifest request_id={request_id}")
        if len(candidate_ids) < 2 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"invalid candidate slate for request_id={request_id}")
        result[request_id] = candidate_ids
    if not result:
        raise ValueError(f"candidate manifest has no split={split}")
    return result, str(manifest["dataset_version"])


def _load_request_contract(
    path: Path, *, split: str
) -> tuple[dict[str, tuple[str, str]], str]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result: dict[str, tuple[str, str]] = {}
    for row in manifest.get("entries", []):
        if row.get("split") != split:
            continue
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate request manifest request_id={request_id}")
        query_hash = str(row.get("query_sha256", ""))
        candidate_hash = str(row.get("candidate_item_ids_sha256", ""))
        if len(query_hash) != 64 or len(candidate_hash) != 64:
            raise ValueError(f"invalid request manifest hashes for request_id={request_id}")
        result[request_id] = (query_hash, candidate_hash)
    if not result:
        raise ValueError(f"request manifest has no split={split}")
    return result, str(manifest["dataset_version"])


def _load_dataset_manifest(standardized_dir: Path) -> dict[str, Any]:
    with (standardized_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_id") or not manifest.get("dataset_version"):
        raise ValueError("dataset manifest lacks dataset identity")
    return manifest


def _assert_feature_store_contract(
    feature_store: FrozenTextFeatureStore,
    *,
    required_records: Sequence[Path],
) -> None:
    if feature_store.metadata.get("qrels_read") is not False:
        raise ValueError("frozen feature store does not attest qrels_read=false")
    if feature_store.metadata.get("feature_contract") != "frozen_transformer_cls_l2_v1":
        raise ValueError("W0 requires frozen_transformer_cls_l2_v1 features")
    if (
        feature_store.metadata.get("visible_text_contract")
        != "query_context_and_canonical_item_semantics_v2"
    ):
        raise ValueError("W0 feature store lacks canonical item semantic texts")
    if not feature_store.encoder_fingerprint_sha256:
        raise ValueError("W0 feature store lacks a frozen encoder fingerprint")
    if not feature_store.store_fingerprint_sha256:
        raise ValueError("W0 feature store lacks a concrete store fingerprint")
    encoding = feature_store.metadata.get("encoder_fingerprint", {}).get(
        "encoding_recipe", {}
    )
    for key in (
        "batch_size",
        "requested_inference_dtype",
        "effective_compute_dtype",
        "device_identity",
        "package_versions",
    ):
        if key not in encoding:
            raise ValueError(
                f"W0 feature encoder fingerprint lacks effective identity field={key}"
            )
    for key in ("reused_text_rows", "new_text_rows", "store_ancestry"):
        if key not in feature_store.metadata:
            raise ValueError(f"W0 feature store lacks ancestry field={key}")
    recorded_hashes = {
        str(row.get("sha256"))
        for row in feature_store.metadata.get("record_files", [])
    }
    missing = [
        str(path) for path in required_records if sha256_file(path) not in recorded_hashes
    ]
    if missing:
        raise ValueError(
            "frozen feature store does not cover required standardized records: "
            + ", ".join(missing)
        )


def _assert_scoring_feature_store_compatible(
    feature_store: FrozenTextFeatureStore,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact training features or a bitwise-reuse descendant superset."""

    checkpoint_features = checkpoint.get("feature_store", {})
    if feature_store.dimension != int(checkpoint["model_config"]["content_dim"]):
        raise ValueError("scoring feature dimension differs from checkpoint")
    if (
        feature_store.metadata.get("feature_contract")
        != checkpoint_features.get("feature_contract")
    ):
        raise ValueError("scoring feature contract differs from checkpoint")
    if (
        feature_store.metadata.get("visible_text_contract")
        != checkpoint_features.get("visible_text_contract")
    ):
        raise ValueError("scoring visible-text contract differs from checkpoint")
    if (
        feature_store.encoder_fingerprint_sha256
        != checkpoint_features.get("encoder_fingerprint_sha256")
    ):
        raise ValueError("scoring frozen encoder fingerprint differs from checkpoint")

    training_store = checkpoint_features.get("store_fingerprint", {})
    training_store_sha = checkpoint_features.get("store_fingerprint_sha256")
    training_metadata_sha = checkpoint_features.get("metadata_sha256")
    if not isinstance(training_store_sha, str) or len(training_store_sha) != 64:
        raise ValueError("checkpoint lacks the training feature-store fingerprint")
    if not isinstance(training_metadata_sha, str) or len(training_metadata_sha) != 64:
        raise ValueError("checkpoint lacks the training feature metadata SHA")
    if feature_store.store_fingerprint_sha256 == training_store_sha:
        if sha256_file(feature_store.root / "metadata.json") != training_metadata_sha:
            raise ValueError("exact scoring feature metadata differs from checkpoint")
        return {
            "mode": "exact_training_store",
            "training_store_fingerprint_sha256": training_store_sha,
            "scoring_store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
            "training_record_hashes_subset": True,
            "bitwise_reuse_ancestry_verified": True,
        }

    ancestors = feature_store.metadata.get("store_ancestry", [])
    matching = [
        entry
        for entry in ancestors
        if entry.get("store_fingerprint_sha256") == training_store_sha
        and entry.get("metadata_sha256") == training_metadata_sha
    ]
    if len(matching) != 1:
        raise ValueError(
            "scoring feature store is neither the training store nor its "
            "verified bitwise-reuse descendant"
        )
    training_record_hashes = {
        str(value) for value in training_store.get("record_sha256s", [])
    }
    scoring_record_hashes = {
        str(row.get("sha256"))
        for row in feature_store.metadata.get("record_files", [])
        if isinstance(row, Mapping) and row.get("sha256")
    }
    if not training_record_hashes or not training_record_hashes <= scoring_record_hashes:
        raise ValueError(
            "scoring feature-store records are not a superset of training records"
        )
    training_store_path = Path(str(checkpoint_features.get("path", "")))
    if not training_store_path.is_dir():
        raise FileNotFoundError(
            "training feature store is unavailable for bitwise ancestry verification"
        )
    training_feature_store = FrozenTextFeatureStore(
        training_store_path, require_fingerprints=True
    )
    if (
        training_feature_store.store_fingerprint_sha256 != training_store_sha
        or sha256_file(training_store_path / "metadata.json") != training_metadata_sha
    ):
        raise ValueError("training feature store differs from the checkpoint identity")
    training_hashes = list(training_feature_store.hash_to_row)
    if not set(training_hashes) <= set(feature_store.hash_to_row):
        raise ValueError("scoring feature store is not a text superset of training")
    for start in range(0, len(training_hashes), 8192):
        chunk = training_hashes[start : start + 8192]
        training_rows = [training_feature_store.hash_to_row[value] for value in chunk]
        scoring_rows = [feature_store.hash_to_row[value] for value in chunk]
        training_vectors = np.asarray(
            training_feature_store.vectors[np.asarray(training_rows)]
        )
        scoring_vectors = np.asarray(feature_store.vectors[np.asarray(scoring_rows)])
        if not np.array_equal(training_vectors, scoring_vectors):
            raise ValueError(
                "scoring feature-store ancestry does not bitwise reuse training rows"
            )
    return {
        "mode": "bitwise_reuse_descendant_superset",
        "training_store_fingerprint_sha256": training_store_sha,
        "scoring_store_fingerprint_sha256": feature_store.store_fingerprint_sha256,
        "training_record_hashes_subset": True,
        "bitwise_reuse_ancestry_verified": True,
        "bitwise_reused_text_rows": len(training_hashes),
        "ancestor_relation": matching[0].get("relation"),
    }


def _assert_frozen_train_inputs(
    *,
    config: Mapping[str, Any],
    standardized_dir: Path,
    feature_store_dir: Path,
    feature_store: FrozenTextFeatureStore,
    dataset_manifest: Mapping[str, Any],
) -> None:
    """Bind a production W0 training invocation to the frozen protocol data."""

    configured_dataset = config.get("dataset", {})
    expected_standardized = Path(
        str(configured_dataset.get("standardized_dir", ""))
    ).resolve()
    if standardized_dir.resolve() != expected_standardized:
        raise ValueError("W0 standardized_dir differs from the frozen config")
    for key in ("dataset_id", "dataset_version"):
        if str(dataset_manifest.get(key)) != str(configured_dataset.get(key)):
            raise ValueError(f"W0 dataset manifest {key} differs from the config")

    protocol_data = config.get("_protocol_payload", {}).get("data", {})
    development = protocol_data.get("development_population", {})
    expected_hashes = {
        "manifest.json": development.get("manifest_sha256"),
        "candidate_manifest.json": development.get("candidate_manifest_sha256"),
        "request_manifest.json": development.get("request_manifest_sha256"),
        "records_train.jsonl": development.get("records_train_sha256"),
        "qrels_train.jsonl": development.get("qrels_train_sha256"),
    }
    for filename, expected in expected_hashes.items():
        path = standardized_dir / filename
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"W0 protocol lacks a frozen hash for {filename}")
        if sha256_file(path) != expected:
            raise ValueError(f"W0 frozen train input hash differs for {filename}")

    content = config.get("content_features", {})
    if feature_store_dir.resolve() != Path(str(content.get("store", ""))).resolve():
        raise ValueError("W0 training feature store differs from the frozen config")
    expected_content = {
        "feature_contract": content.get("contract"),
        "visible_text_contract": content.get("visible_text_contract"),
        "model_name_or_path": content.get("model"),
        "qrels_read": content.get("qrels_read"),
    }
    for metadata_key, expected in expected_content.items():
        if feature_store.metadata.get(metadata_key) != expected:
            raise ValueError(
                f"W0 feature metadata {metadata_key} differs from the config"
            )
    encoding = feature_store.metadata["encoder_fingerprint"].get(
        "encoding_recipe", {}
    )
    if int(encoding.get("max_length", -1)) != int(content.get("max_length", -2)):
        raise ValueError("W0 feature max_length differs from the frozen config")
    configured_encoder_sha = content.get("encoder_fingerprint_sha256")
    if configured_encoder_sha is not None and (
        configured_encoder_sha != feature_store.encoder_fingerprint_sha256
    ):
        raise ValueError("W0 feature encoder fingerprint differs from the config")
    configured_store_sha = content.get("store_fingerprint_sha256")
    if configured_store_sha != feature_store.store_fingerprint_sha256:
        raise ValueError("W0 feature store fingerprint differs from the config")


def _assert_frozen_dev_scoring_inputs(
    *,
    config: Mapping[str, Any],
    standardized_dir: Path,
    records_path: Path,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind internal-dev scoring to the exact qrels-free protocol population."""

    configured_dataset = config.get("dataset", {})
    expected_standardized = Path(
        str(configured_dataset.get("standardized_dir", ""))
    ).resolve()
    if standardized_dir.resolve() != expected_standardized:
        raise ValueError("W0 dev standardized_dir differs from the frozen config")
    for key in ("dataset_id", "dataset_version"):
        if str(dataset_manifest.get(key)) != str(configured_dataset.get(key)):
            raise ValueError(f"W0 dev dataset manifest {key} differs from the config")

    development = (
        config.get("_protocol_payload", {})
        .get("data", {})
        .get("development_population", {})
    )
    expected_hashes = {
        "manifest.json": development.get("manifest_sha256"),
        "candidate_manifest.json": development.get("candidate_manifest_sha256"),
        "request_manifest.json": development.get("request_manifest_sha256"),
        "records_dev.jsonl": development.get("records_dev_sha256"),
    }
    if records_path != standardized_dir / "records_dev.jsonl":
        raise ValueError("W0 internal-dev scoring did not select records_dev.jsonl")
    verified: dict[str, str] = {}
    for filename, expected in expected_hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"W0 protocol lacks a frozen dev hash for {filename}")
        actual = sha256_file(standardized_dir / filename)
        if actual != expected:
            raise ValueError(f"W0 frozen dev input hash differs for {filename}")
        verified[filename] = actual
    return {
        "mode": "frozen_internal_dev_qrels_free",
        "protocol_sha256": config["protocol"]["sha256"],
        "verified_file_sha256s": verified,
        "qrels_opened": False,
        "passed": True,
    }


def _assert_frozen_legacy_scoring_inputs(
    *,
    config: Mapping[str, Any],
    standardized_dir: Path,
    records_path: Path,
    dataset_manifest: Mapping[str, Any],
) -> None:
    configured_dataset = config.get("dataset", {})
    if standardized_dir.resolve() != Path(
        str(configured_dataset.get("standardized_dir", ""))
    ).resolve():
        raise ValueError("W0 legacy standardized_dir differs from the frozen config")
    for key in ("dataset_id", "dataset_version"):
        if str(dataset_manifest.get(key)) != str(configured_dataset.get(key)):
            raise ValueError(f"W0 legacy dataset manifest {key} differs from config")
    development = (
        config.get("_protocol_payload", {})
        .get("data", {})
        .get("development_population", {})
    )
    expected = {
        "manifest.json": development.get("manifest_sha256"),
        "candidate_manifest.json": development.get("candidate_manifest_sha256"),
        "request_manifest.json": development.get("request_manifest_sha256"),
        "records_confirmation.jsonl": development.get(
            "records_legacy_compatibility_sha256"
        ),
    }
    if records_path != standardized_dir / "records_confirmation.jsonl":
        raise ValueError("W0 legacy scoring did not select records_confirmation.jsonl")
    for filename, expected_sha in expected.items():
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"W0 protocol lacks a frozen legacy hash for {filename}")
        if sha256_file(standardized_dir / filename) != expected_sha:
            raise ValueError(f"W0 frozen legacy input hash differs for {filename}")


def _assert_frozen_scoring_population(
    *,
    config: Mapping[str, Any],
    standardized_dir: Path,
    records_path: Path,
    dataset_manifest: Mapping[str, Any],
    split: str,
    history_condition: str,
    checkpoint: Mapping[str, Any],
    checkpoint_metadata_path: Path,
    model_path: Path,
    model_sha256: str,
    implementation_digest: str,
) -> dict[str, Any] | None:
    """Verify development hashes or the qrels-free released holdout identity."""

    development = (
        config.get("_protocol_payload", {})
        .get("data", {})
        .get("development_population", {})
    )
    dataset_version = str(dataset_manifest.get("dataset_version"))
    if dataset_version == str(development.get("dataset_version")):
        if history_condition == "wrong":
            raise ValueError("development/legacy W0 scoring permits only full/null")
        if split == "dev":
            _assert_frozen_dev_scoring_inputs(
                config=config,
                standardized_dir=standardized_dir,
                records_path=records_path,
                dataset_manifest=dataset_manifest,
            )
        elif split == "confirmation":
            _assert_frozen_legacy_scoring_inputs(
                config=config,
                standardized_dir=standardized_dir,
                records_path=records_path,
                dataset_manifest=dataset_manifest,
            )
        else:
            raise ValueError("unsupported frozen W0 development split")
        return None

    from myrec.data.kuaisearch_holdout import (
        V12_DATASET_VERSION,
        verify_published_holdout,
    )

    if dataset_version != V12_DATASET_VERSION:
        raise ValueError(f"unregistered V1.2 W0 scoring dataset={dataset_version!r}")
    if split != "confirmation":
        raise ValueError("the registered V1.2 holdout may score confirmation only")
    audit = verify_published_holdout(
        standardized_dir,
        protocol_path=config["protocol"]["path"],
        open_qrels=False,
    )
    if audit.get("qrels_opened") is not False:
        raise ValueError("W0 holdout verifier unexpectedly opened qrels")
    frozen = audit.get("checkpoint_identities", {}).get(METHOD_ID)
    if not isinstance(frozen, Mapping):
        raise ValueError("holdout release lock lacks the W0 checkpoint identity")
    expected = {
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "checkpoint_sha256": model_sha256,
        "config_sha256": config["_config_sha256"],
        "implementation_digest": implementation_digest,
        "protocol_sha256": config["protocol"]["sha256"],
        "training_metadata_sha256": sha256_file(checkpoint_metadata_path),
    }
    for key, observed in expected.items():
        if frozen.get(key) != observed:
            raise ValueError(f"holdout frozen W0 checkpoint mismatch: {key}")
    if Path(str(frozen.get("training_metadata_path", ""))).resolve() != (
        checkpoint_metadata_path.resolve()
    ):
        raise ValueError("holdout frozen W0 training metadata path mismatch")
    frozen_files = frozen.get("checkpoint_files", [])
    if not isinstance(frozen_files, list) or len(frozen_files) != 1:
        raise ValueError("holdout frozen W0 checkpoint file list is invalid")
    frozen_model = frozen_files[0]
    expected_file = {
        "name": "model.pt",
        "sha256": model_sha256,
        "size_bytes": model_path.stat().st_size,
    }
    for key, observed in expected_file.items():
        if frozen_model.get(key) != observed:
            raise ValueError(f"holdout frozen W0 model artifact mismatch: {key}")
    if Path(str(frozen_model.get("path", ""))).resolve() != model_path.resolve():
        raise ValueError("holdout frozen W0 model artifact path mismatch")
    return {
        "checkpoint_identity_manifest_sha256": frozen[
            "identity_manifest_sha256"
        ],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "integrity_lock_sha256": audit["integrity_lock_sha256"],
        "manifest_sha256": audit["manifest_sha256"],
        "post_selection_recipe_checkpoint_lock_sha256": audit[
            "post_selection_recipe_checkpoint_lock_sha256"
        ],
        "protocol_sha256": audit["protocol_sha256"],
        "qrels_opened": False,
        "verified_before_model_load": True,
    }


def _assert_frozen_recipe(**values: Any) -> None:
    expected = {
        "seed": PILOT_SEED,
        "history_budget": HISTORY_BUDGET,
        "epochs": EPOCHS,
        "batch_requests": BATCH_REQUESTS,
        "learning_rate": LEARNING_RATE,
        "projection_dim": PROJECTION_DIM,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "contrastive_loss_weight": CONTRASTIVE_LOSS_WEIGHT,
        "replacement_ratio": REPLACEMENT_RATIO,
        "semantic_shortlist_size": SEMANTIC_SHORTLIST_SIZE,
    }
    if values != expected:
        differences = {
            key: {"expected": expected[key], "observed": values.get(key)}
            for key in expected
            if values.get(key) != expected[key]
        }
        raise ValueError(f"W0 V1.2 frozen recipe mismatch: {differences}")


def _frozen_recipe() -> dict[str, Any]:
    return {
        "seed": PILOT_SEED,
        "history_budget": HISTORY_BUDGET,
        "epochs": EPOCHS,
        "batch_requests": BATCH_REQUESTS,
        "learning_rate": LEARNING_RATE,
        "projection_dim": PROJECTION_DIM,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "contrastive_loss_weight": CONTRASTIVE_LOSS_WEIGHT,
        "replacement_ratio": REPLACEMENT_RATIO,
        "semantic_shortlist_size": SEMANTIC_SHORTLIST_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_continuous_job_seconds": MAX_CONTINUOUS_JOB_SECONDS,
        "safe_exit_seconds": SAFE_EXIT_SECONDS,
    }


def _input_whitelist_metadata() -> dict[str, list[str]]:
    return {
        "request": list(REQUEST_INPUT_WHITELIST),
        "history": list(HISTORY_INPUT_WHITELIST),
        "candidate": list(CANDIDATE_INPUT_WHITELIST),
        "masks": list(MASK_INPUT_WHITELIST),
    }


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_order(size: int, *, seed: int, epoch: int) -> np.ndarray:
    generator = np.random.default_rng(seed + epoch)
    return generator.permutation(size)


def _warmup_cosine_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return scale


def _save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_cursor: int,
    optimizer_steps: int,
    losses: list[dict[str, float]],
    training_contract: dict[str, Any],
    run_lineage: list[str],
    cumulative_elapsed_seconds: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "optimizer_steps": optimizer_steps,
            "losses": losses,
            "training_contract": training_contract,
            "run_lineage": run_lineage,
            "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        },
        temporary,
    )
    temporary.replace(path)


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _manifest_hash_metadata(standardized_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for filename in ("manifest.json", "candidate_manifest.json", "request_manifest.json"):
        path = standardized_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing standardized manifest: {path}")
        result[filename] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _implementation_identity() -> dict[str, Any]:
    """Hash every project-owned source file that defines W0 model inputs/scores."""

    baseline_dir = Path(__file__).parent
    files = sorted(
        [
            Path(__file__),
            baseline_dir / "frozen_text_features.py",
            baseline_dir / "representative_sequence_adapter.py",
        ]
    )
    values = [
        {"path": str(path), "sha256": sha256_file(path)} for path in files
    ]
    return {
        "digest": sha256_text(
            json.dumps(values, sort_keys=True, separators=(",", ":"))
        ),
        "files": values,
    }


def _code_revision_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "revision": revision,
        "dirty": bool(status),
        "status_sha256": sha256_text(status),
    }


def _runtime_metadata(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        result["cuda_device_name"] = torch.cuda.get_device_name(index)
    return result


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if config_path is None:
        return
    path = Path(config_path)
    if path.exists():
        shutil.copyfile(path, run_dir / f"config_snapshot{path.suffix}")


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must use YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>"
        )


def _load_witness_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("invalid W0 V1.2 config")
    if config.get("method_id") != METHOD_ID:
        raise ValueError("W0 config method_id mismatch")
    protocol = config.get("protocol", {})
    protocol_path = Path(str(protocol.get("path", "")))
    if not protocol_path.exists() or sha256_file(protocol_path) != protocol.get(
        "sha256"
    ):
        raise ValueError("W0 frozen protocol hash mismatch")
    protocol_payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol_payload, dict):
        raise ValueError("W0 frozen protocol is not an object")
    expected_training = {
        "seed": PILOT_SEED,
        "history_budget": HISTORY_BUDGET,
        "epochs": EPOCHS,
        "batch_requests": BATCH_REQUESTS,
        "learning_rate": LEARNING_RATE,
        "projection_dim": PROJECTION_DIM,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "contrastive_loss_weight": CONTRASTIVE_LOSS_WEIGHT,
    }
    observed_training = config.get("training", {})
    for key, expected in expected_training.items():
        if observed_training.get(key) != expected:
            raise ValueError(f"W0 config training.{key} drifted")
    expected_fixed_training = {
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_continuous_job_seconds": MAX_CONTINUOUS_JOB_SECONDS,
        "safe_exit_seconds": SAFE_EXIT_SECONDS,
    }
    for key, expected in expected_fixed_training.items():
        if observed_training.get(key) != expected:
            raise ValueError(f"W0 config training.{key} drifted")
    semantic = config.get("mechanism", {}).get("semantic_replacement", {})
    if semantic.get("replacement_ratio") != REPLACEMENT_RATIO:
        raise ValueError("W0 config replacement_ratio drifted")
    if semantic.get("deterministic_shortlist_size") != SEMANTIC_SHORTLIST_SIZE:
        raise ValueError("W0 config deterministic_shortlist_size drifted")
    expected_semantic = {
        "views_per_request": 2,
        "category_key": "deepest_complete_category",
        "donor_population": "train_visible_history_and_candidates_only",
        "different_item_id_required": True,
        "exclude_original_history_and_current_candidates": True,
        "selection": "highest_frozen_bge_cosine",
        "missing_replacement": "drop_selected_event_from_augmented_view_and_audit",
    }
    for key, expected in expected_semantic.items():
        if semantic.get(key) != expected:
            raise ValueError(f"W0 config semantic_replacement.{key} drifted")
    boundary = config.get("input_boundary", {})
    expected_boundary = {
        "request_fields": list(REQUEST_INPUT_WHITELIST),
        "history_fields": list(HISTORY_INPUT_WHITELIST),
        "candidate_fields": list(CANDIDATE_INPUT_WHITELIST),
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            raise ValueError(f"W0 config input_boundary.{key} drifted")
    protocol_method = protocol_payload.get("methods", {}).get(METHOD_ID, {})
    if protocol_method.get("history_budget") != HISTORY_BUDGET:
        raise ValueError("W0 protocol history_budget drifted")
    if protocol_method.get("semantic_replacement", {}).get(
        "replacement_ratio"
    ) != REPLACEMENT_RATIO:
        raise ValueError("W0 protocol replacement ratio drifted")
    content = config.get("content_features", {})
    for config_key, protocol_key in (
        ("contract", "feature_contract"),
        ("visible_text_contract", "content_feature_contract"),
        ("encoder_fingerprint_sha256", "content_encoder_fingerprint_sha256"),
        ("store_fingerprint_sha256", "content_store_fingerprint_sha256"),
    ):
        if content.get(config_key) != protocol_method.get(protocol_key):
            raise ValueError(
                f"W0 config content_features.{config_key} drifted from protocol"
            )
    config["_config_sha256"] = sha256_file(path)
    config["_protocol_payload"] = protocol_payload
    return config
