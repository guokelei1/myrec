"""Label-free M1 history intervention materialization.

The module consumes only the frozen unified ``records_train.jsonl`` and
``records_dev.jsonl`` interfaces plus a read-only BGE feature store.  It does
not accept a qrels path, a score path, or an evaluation object.  Train history
events are the sole donor population: unlike train candidate rows, they have
real event/query/timestamp provenance and can therefore remain valid unified
history rows after replacement.

Every emitted assignment contains exactly ``request_id``, ``condition_id``
and ``history``.  Query and candidates stay in the source dev record and are
bound to assignments by request-id/hash evidence in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

import numpy as np

from myrec.baselines.frozen_text_features import (
    FrozenTextFeatureStore,
    serialize_item_semantic_content,
)
from myrec.baselines.motivation_v12_contracts import (
    CANDIDATE_INPUT_FIELDS,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    HISTORY_INPUT_FIELDS,
)
from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


MECHANISM_INTERVENTION_SEED = 20_260_717
HISTORY_BUDGET = 6
SEMANTIC_SHORTLIST_SIZE = 64
CONDITION_IDS = (
    "relevant_6",
    "irrelevant_6",
    "recent_6_order_shuffle",
    "semantic_preserving_different_id",
    "semantic_breaking_different_id",
    "candidate_overlap_semantic_swap",
)

FeatureLookup = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class DonorEvent:
    """One canonical train-history event with auditable provenance."""

    item_id: str
    history: dict[str, Any]
    brand: str
    category: tuple[str, ...]
    top_category: str
    title: str
    semantic_text: str
    source_request_id: str
    source_history_position: int
    original_ts: int | float


@dataclass(frozen=True)
class DonorCatalog:
    """Deterministically ordered train-only donor pools."""

    all_events: tuple[DonorEvent, ...]
    by_category: Mapping[tuple[str, ...], tuple[DonorEvent, ...]]
    by_brand_category: Mapping[
        tuple[str, tuple[str, ...]], tuple[DonorEvent, ...]
    ]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class ReplacementLineage:
    condition_id: str
    target_request_id: str
    target_history_position: int
    replaced_item_id: str
    donor: DonorEvent
    emitted_ts: int | float
    timestamp_adjusted: bool
    fallback_used: bool


@dataclass(frozen=True)
class ConditionOutcome:
    history: tuple[dict[str, Any], ...]
    attempted_replacements: int = 0
    fallback_attempts: int = 0
    lineage: tuple[ReplacementLineage, ...] = ()


class _NormalizedFeatureLookup:
    """Finite, normalized BGE lookup with a bounded in-process cache."""

    def __init__(self, lookup: FeatureLookup, *, cache_size: int = 16_384) -> None:
        if cache_size <= 0:
            raise ValueError("feature cache size must be positive")
        self.lookup = lookup
        self.cache_size = cache_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.dimension: int | None = None

    def __call__(self, text: str) -> np.ndarray:
        cached = self.cache.pop(text, None)
        if cached is not None:
            self.cache[text] = cached
            return cached
        value = np.asarray(self.lookup(text), dtype=np.float32)
        if value.ndim != 1 or value.size == 0:
            raise ValueError("BGE feature lookup must return a non-empty vector")
        if not bool(np.isfinite(value).all()):
            raise ValueError("BGE feature lookup returned a non-finite vector")
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("BGE feature lookup returned a zero vector")
        value = value / norm
        if self.dimension is None:
            self.dimension = int(value.size)
        elif value.size != self.dimension:
            raise ValueError("BGE feature dimensions differ across texts")
        self.cache[text] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value


def build_train_donor_catalog(
    train_records: Iterable[Mapping[str, Any]],
) -> DonorCatalog:
    """Build a label-insensitive catalog from train history events only."""

    by_item: dict[str, tuple[tuple[Any, ...], DonorEvent]] = {}
    seen_requests: set[str] = set()
    history_observations = 0
    duplicate_item_observations = 0
    for raw_record in train_records:
        request_id = _required_text(raw_record, "request_id", scope="train record")
        if request_id in seen_requests:
            raise ValueError(f"duplicate train request_id={request_id}")
        seen_requests.add(request_id)
        raw_history = raw_record.get("history")
        if not isinstance(raw_history, list):
            raise ValueError(f"train request_id={request_id}: history must be a list")
        for position, raw_event in enumerate(raw_history):
            event = _project_item(raw_event, HISTORY_INPUT_FIELDS, scope="train history")
            item_id = _required_text(event, "item_id", scope="train history")
            original_ts = _required_timestamp(event, scope="train history")
            history_observations += 1
            category = _category_key(event)
            donor = DonorEvent(
                item_id=item_id,
                history=event,
                brand=_normalized_text(event.get("brand")),
                category=category,
                top_category=category[0] if category else "",
                title=_normalized_title(event.get("title")),
                semantic_text=serialize_item_semantic_content(event),
                source_request_id=request_id,
                source_history_position=position,
                original_ts=original_ts,
            )
            # The canonical source occurrence is independent of input row order.
            canonical_key = (
                request_id,
                position,
                _canonical_json(event),
            )
            prior = by_item.get(item_id)
            if prior is not None:
                duplicate_item_observations += 1
            if prior is None or canonical_key < prior[0]:
                by_item[item_id] = (canonical_key, donor)
    if not seen_requests:
        raise ValueError("train donor records are empty")
    donors = tuple(
        sorted((value[1] for value in by_item.values()), key=lambda row: row.item_id)
    )
    if not donors:
        raise ValueError("train records contain no history donor events")
    category_pools: dict[tuple[str, ...], list[DonorEvent]] = {}
    brand_category_pools: dict[
        tuple[str, tuple[str, ...]], list[DonorEvent]
    ] = {}
    for donor in donors:
        if not donor.category:
            continue
        category_pools.setdefault(donor.category, []).append(donor)
        if donor.brand:
            brand_category_pools.setdefault(
                (donor.brand, donor.category), []
            ).append(donor)
    return DonorCatalog(
        all_events=donors,
        by_category={key: tuple(value) for key, value in category_pools.items()},
        by_brand_category={
            key: tuple(value) for key, value in brand_category_pools.items()
        },
        audit={
            "source_scope": "records_train.jsonl history events only",
            "train_requests": len(seen_requests),
            "history_event_observations": history_observations,
            "unique_item_ids": len(donors),
            "duplicate_item_observations": duplicate_item_observations,
            "category_pools": len(category_pools),
            "brand_category_pools": len(brand_category_pools),
            "candidate_rows_used_as_donors": 0,
        },
    )


class HistoryInterventionEngine:
    """Apply all six preregistered interventions to one dev request."""

    def __init__(
        self,
        catalog: DonorCatalog,
        feature_lookup: FeatureLookup,
        *,
        seed: int = MECHANISM_INTERVENTION_SEED,
        history_budget: int = HISTORY_BUDGET,
        shortlist_size: int = SEMANTIC_SHORTLIST_SIZE,
    ) -> None:
        if seed != MECHANISM_INTERVENTION_SEED:
            raise ValueError(
                f"M1 intervention seed is frozen at {MECHANISM_INTERVENTION_SEED}"
            )
        if history_budget != HISTORY_BUDGET:
            raise ValueError(f"M1 visible history budget is frozen at {HISTORY_BUDGET}")
        if shortlist_size != SEMANTIC_SHORTLIST_SIZE:
            raise ValueError(
                "M1 semantic shortlist size is frozen at "
                f"{SEMANTIC_SHORTLIST_SIZE}"
            )
        self.catalog = catalog
        self.features = _NormalizedFeatureLookup(feature_lookup)
        self.seed = seed
        self.history_budget = history_budget
        self.shortlist_size = shortlist_size

    def apply(self, raw_record: Mapping[str, Any]) -> dict[str, ConditionOutcome]:
        record = _sanitize_dev_record(raw_record)
        request_id = record["request_id"]
        request_ts = record["ts"]
        history = record["history"]
        recent = tuple(_copy_value(row) for row in history[-self.history_budget :])
        candidate_ids = {str(row["item_id"]) for row in record["candidates"]}
        excluded_ids = {
            str(row["item_id"]) for row in history
        } | candidate_ids

        relevant = self._relevance_select(record["query"], history, highest=True)
        irrelevant = self._relevance_select(record["query"], history, highest=False)
        shuffled = self._shuffle_recent(request_id, recent)

        preserving = self._replace_history(
            request_id=request_id,
            request_ts=request_ts,
            recent=recent,
            candidate_ids=candidate_ids,
            excluded_ids=excluded_ids,
            condition_id="semantic_preserving_different_id",
            mode="preserving",
        )
        breaking = self._replace_history(
            request_id=request_id,
            request_ts=request_ts,
            recent=recent,
            candidate_ids=candidate_ids,
            excluded_ids=excluded_ids,
            condition_id="semantic_breaking_different_id",
            mode="breaking",
        )
        overlap = self._replace_history(
            request_id=request_id,
            request_ts=request_ts,
            recent=recent,
            candidate_ids=candidate_ids,
            excluded_ids=excluded_ids,
            condition_id="candidate_overlap_semantic_swap",
            mode="candidate_overlap",
        )
        return {
            "relevant_6": ConditionOutcome(history=relevant),
            "irrelevant_6": ConditionOutcome(history=irrelevant),
            "recent_6_order_shuffle": ConditionOutcome(history=shuffled),
            "semantic_preserving_different_id": preserving,
            "semantic_breaking_different_id": breaking,
            "candidate_overlap_semantic_swap": overlap,
        }

    def _relevance_select(
        self,
        query: str,
        history: Sequence[Mapping[str, Any]],
        *,
        highest: bool,
    ) -> tuple[dict[str, Any], ...]:
        if not history:
            return ()
        query_vector = self.features(f"query: {query}")
        similarities = [
            float(query_vector @ self.features(serialize_item_content(row)))
            for row in history
        ]
        count = min(self.history_budget, len(history))
        if highest:
            selected = sorted(
                range(len(history)), key=lambda index: (-similarities[index], index)
            )[:count]
        else:
            selected = sorted(
                range(len(history)), key=lambda index: (similarities[index], index)
            )[:count]
        # Relevance changes membership only; source temporal order is retained.
        return tuple(_copy_value(history[index]) for index in sorted(selected))

    def _shuffle_recent(
        self, request_id: str, recent: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        ranked = sorted(
            range(len(recent)),
            key=lambda index: _stable_digest(
                self.seed,
                "recent_6_order_shuffle",
                request_id,
                index,
                recent[index].get("item_id", ""),
            ),
        )
        return tuple(_copy_value(recent[index]) for index in ranked)

    def _replace_history(
        self,
        *,
        request_id: str,
        request_ts: int | float,
        recent: Sequence[Mapping[str, Any]],
        candidate_ids: set[str],
        excluded_ids: set[str],
        condition_id: str,
        mode: str,
    ) -> ConditionOutcome:
        output: list[dict[str, Any]] = []
        lineage: list[ReplacementLineage] = []
        attempted = 0
        fallback_attempts = 0
        for position, source in enumerate(recent):
            source_item_id = str(source["item_id"])
            eligible_position = mode != "candidate_overlap" or source_item_id in candidate_ids
            if not eligible_position:
                output.append(_copy_value(source))
                continue
            attempted += 1
            if mode in {"preserving", "candidate_overlap"}:
                donor, fallback_used = self._select_preserving_donor(
                    source,
                    excluded_ids=excluded_ids,
                    deterministic_key=f"{request_id}|{condition_id}|position={position}",
                )
                if fallback_used:
                    fallback_attempts += 1
            elif mode == "breaking":
                donor = self._select_breaking_donor(
                    source,
                    excluded_ids=excluded_ids,
                    deterministic_key=f"{request_id}|{condition_id}|position={position}",
                )
                fallback_used = False
            else:
                raise AssertionError(f"unknown replacement mode={mode}")
            if donor is None:
                output.append(_copy_value(source))
                continue
            emitted, adjusted = _causal_donor_history(donor, request_ts=request_ts)
            if donor.item_id in candidate_ids:
                raise AssertionError("semantic donor leaked a target candidate item ID")
            if donor.item_id in excluded_ids:
                raise AssertionError("semantic donor reused an original history item ID")
            output.append(emitted)
            lineage.append(
                ReplacementLineage(
                    condition_id=condition_id,
                    target_request_id=request_id,
                    target_history_position=position,
                    replaced_item_id=source_item_id,
                    donor=donor,
                    emitted_ts=emitted["ts"],
                    timestamp_adjusted=adjusted,
                    fallback_used=fallback_used,
                )
            )
        return ConditionOutcome(
            history=tuple(output),
            attempted_replacements=attempted,
            fallback_attempts=fallback_attempts,
            lineage=tuple(lineage),
        )

    def _select_preserving_donor(
        self,
        source: Mapping[str, Any],
        *,
        excluded_ids: set[str],
        deterministic_key: str,
    ) -> tuple[DonorEvent | None, bool]:
        category = _category_key(source)
        brand = _normalized_text(source.get("brand"))
        source_title = _normalized_title(source.get("title"))
        if not category:
            return None, False

        def eligible(donor: DonorEvent) -> bool:
            return (
                donor.item_id not in excluded_ids
                and donor.item_id != str(source.get("item_id", ""))
                and donor.title != source_title
            )

        if brand:
            primary = self.catalog.by_brand_category.get((brand, category), ())
            shortlist = self._stable_shortlist(
                primary,
                eligible=eligible,
                deterministic_key=f"{deterministic_key}|same-brand-category",
            )
            if shortlist:
                return self._highest_cosine(source, shortlist), False
        fallback = self.catalog.by_category.get(category, ())
        shortlist = self._stable_shortlist(
            fallback,
            eligible=eligible,
            deterministic_key=f"{deterministic_key}|same-category-fallback",
        )
        if shortlist:
            return self._highest_cosine(source, shortlist), True
        return None, bool(brand)

    def _select_breaking_donor(
        self,
        source: Mapping[str, Any],
        *,
        excluded_ids: set[str],
        deterministic_key: str,
    ) -> DonorEvent | None:
        source_category = _category_key(source)
        source_top = source_category[0] if source_category else ""
        source_brand = _normalized_text(source.get("brand"))
        if not source_top:
            return None

        def eligible(donor: DonorEvent) -> bool:
            return (
                donor.item_id not in excluded_ids
                and donor.item_id != str(source.get("item_id", ""))
                and bool(donor.top_category)
                and donor.top_category != source_top
                and donor.brand != source_brand
            )

        shortlist = self._stable_shortlist(
            self.catalog.all_events,
            eligible=eligible,
            deterministic_key=f"{deterministic_key}|different-top-category-brand",
        )
        if not shortlist:
            return None
        source_vector = self.features(serialize_item_semantic_content(source))
        similarities = [
            float(source_vector @ self.features(donor.semantic_text))
            for donor in shortlist
        ]
        # First restrict to the lowest-similarity quartile, then length-match
        # within that deliberately semantic-breaking set.
        low_count = max(1, math.ceil(len(shortlist) / 4))
        low_indices = sorted(
            range(len(shortlist)),
            key=lambda index: (similarities[index], shortlist[index].item_id),
        )[:low_count]
        source_length = _visible_history_characters((source,))
        best = min(
            low_indices,
            key=lambda index: (
                abs(_visible_history_characters((shortlist[index].history,)) - source_length),
                similarities[index],
                shortlist[index].item_id,
            ),
        )
        return shortlist[best]

    def _highest_cosine(
        self, source: Mapping[str, Any], donors: Sequence[DonorEvent]
    ) -> DonorEvent:
        source_vector = self.features(serialize_item_semantic_content(source))
        similarities = [
            float(source_vector @ self.features(donor.semantic_text)) for donor in donors
        ]
        best = min(
            range(len(donors)),
            key=lambda index: (-similarities[index], donors[index].item_id),
        )
        return donors[best]

    def _stable_shortlist(
        self,
        pool: Sequence[DonorEvent],
        *,
        eligible: Callable[[DonorEvent], bool],
        deterministic_key: str,
    ) -> tuple[DonorEvent, ...]:
        if not pool:
            return ()
        start = int(
            _stable_digest(self.seed, "semantic-shortlist", deterministic_key)[:16],
            16,
        ) % len(pool)
        selected: list[DonorEvent] = []
        for offset in range(len(pool)):
            donor = pool[(start + offset) % len(pool)]
            if eligible(donor):
                selected.append(donor)
                if len(selected) == self.shortlist_size:
                    break
        return tuple(selected)


class _RunAudit:
    def __init__(self) -> None:
        self.requests = 0
        self.seen_request_ids: set[str] = set()
        self.request_hasher = hashlib.sha256()
        self.query_hasher = hashlib.sha256()
        self.candidate_hasher = hashlib.sha256()
        self.query_candidate_hasher = hashlib.sha256()
        self.conditions: dict[str, dict[str, Any]] = {
            condition: {
                "requests": 0,
                "output_history_events": 0,
                "attempted_replacements": 0,
                "replacements": 0,
                "fallback_attempts": 0,
                "fallback_replacements": 0,
                "no_op_requests": 0,
                "selection_change_requests": 0,
                "character_length_deltas": [],
            }
            for condition in CONDITION_IDS
        }
        self.forbidden_fields = 0
        self.non_whitelisted_fields = 0
        self.candidate_leakage = 0
        self.causality_violations = 0
        self.donor_item_ids: Counter[str] = Counter()
        self.donor_source_requests: set[str] = set()
        self.donor_lineage_hasher = hashlib.sha256()
        self.donor_timestamp_adjusted = 0
        self.donor_timestamp_preserved = 0
        self.donor_by_condition: Counter[str] = Counter()

    def add_request(
        self,
        record: Mapping[str, Any],
        outcomes: Mapping[str, ConditionOutcome],
    ) -> None:
        request_id = str(record["request_id"])
        if request_id in self.seen_request_ids:
            raise ValueError(f"duplicate dev request_id={request_id}")
        self.seen_request_ids.add(request_id)
        self.requests += 1
        projected_candidates = record["candidates"]
        baseline = tuple(record["history"][-HISTORY_BUDGET:])
        _update_digest(self.request_hasher, request_id)
        _update_digest(
            self.query_hasher,
            {"request_id": request_id, "query": record["query"]},
        )
        _update_digest(
            self.candidate_hasher,
            {"request_id": request_id, "candidates": projected_candidates},
        )
        _update_digest(
            self.query_candidate_hasher,
            {
                "request_id": request_id,
                "query": record["query"],
                "candidates": projected_candidates,
            },
        )
        if set(outcomes) != set(CONDITION_IDS):
            raise AssertionError("intervention engine omitted or added a condition")
        candidate_ids = {str(row["item_id"]) for row in projected_candidates}
        baseline_characters = _visible_history_characters(baseline)
        for condition_id in CONDITION_IDS:
            outcome = outcomes[condition_id]
            if len(outcome.history) > HISTORY_BUDGET:
                raise AssertionError("intervention exceeded the visible history budget")
            stats = self.conditions[condition_id]
            stats["requests"] += 1
            stats["output_history_events"] += len(outcome.history)
            stats["attempted_replacements"] += outcome.attempted_replacements
            stats["replacements"] += len(outcome.lineage)
            stats["fallback_attempts"] += outcome.fallback_attempts
            stats["fallback_replacements"] += sum(
                int(row.fallback_used) for row in outcome.lineage
            )
            if list(outcome.history) == list(baseline):
                stats["no_op_requests"] += 1
            else:
                stats["selection_change_requests"] += 1
            stats["character_length_deltas"].append(
                _visible_history_characters(outcome.history) - baseline_characters
            )
            for event in outcome.history:
                keys = set(event)
                self.forbidden_fields += len(keys & FORBIDDEN_MODEL_INPUT_FIELDS)
                self.non_whitelisted_fields += len(keys - set(HISTORY_INPUT_FIELDS))
                event_ts = _required_timestamp(event, scope="emitted history")
                if event_ts >= record["ts"]:
                    self.causality_violations += 1
            for row in outcome.lineage:
                if row.donor.item_id in candidate_ids:
                    self.candidate_leakage += 1
                self.donor_item_ids[row.donor.item_id] += 1
                self.donor_source_requests.add(row.donor.source_request_id)
                self.donor_by_condition[row.condition_id] += 1
                if row.timestamp_adjusted:
                    self.donor_timestamp_adjusted += 1
                else:
                    self.donor_timestamp_preserved += 1
                _update_digest(
                    self.donor_lineage_hasher,
                    {
                        "condition_id": row.condition_id,
                        "target_request_id": row.target_request_id,
                        "target_history_position": row.target_history_position,
                        "replaced_item_id": row.replaced_item_id,
                        "donor_original_item_id": row.donor.item_id,
                        "donor_source_request_id": row.donor.source_request_id,
                        "donor_source_history_position": (
                            row.donor.source_history_position
                        ),
                        "donor_original_ts": row.donor.original_ts,
                        "emitted_ts": row.emitted_ts,
                    },
                )

    def finalize(self) -> dict[str, Any]:
        if self.requests <= 0:
            raise ValueError("dev intervention records are empty")
        condition_audit: dict[str, Any] = {}
        for condition_id, raw in self.conditions.items():
            requests = int(raw["requests"])
            attempted = int(raw["attempted_replacements"])
            replacements = int(raw["replacements"])
            fallback_attempts = int(raw["fallback_attempts"])
            fallback_replacements = int(raw["fallback_replacements"])
            condition_audit[condition_id] = {
                key: value
                for key, value in raw.items()
                if key != "character_length_deltas"
            }
            condition_audit[condition_id].update(
                {
                    "replacement_rate": _rate(replacements, attempted),
                    "no_op_rate": _rate(int(raw["no_op_requests"]), requests),
                    "fallback_rate": _rate(fallback_replacements, attempted),
                    "fallback_success_rate": _rate(
                        fallback_replacements, fallback_attempts
                    ),
                    "selection_change_rate": _rate(
                        int(raw["selection_change_requests"]), requests
                    ),
                    "character_length_delta": _summary(
                        raw["character_length_deltas"]
                    ),
                }
            )
        donor_ids_sorted = sorted(self.donor_item_ids)
        donor_requests_sorted = sorted(self.donor_source_requests)
        return {
            "requests": self.requests,
            "request_ids_sha256": self.request_hasher.hexdigest(),
            "query_binding_sha256": self.query_hasher.hexdigest(),
            "candidate_binding_sha256": self.candidate_hasher.hexdigest(),
            "query_candidate_binding_sha256": (
                self.query_candidate_hasher.hexdigest()
            ),
            "conditions": condition_audit,
            "integrity": {
                "forbidden_field_count": self.forbidden_fields,
                "non_whitelisted_history_field_count": self.non_whitelisted_fields,
                "candidate_leakage_count": self.candidate_leakage,
                "causality_violation_count": self.causality_violations,
            },
            "donor_audit": {
                "source_scope": "records_train.jsonl history events only",
                "source_kinds": {"train_history": sum(self.donor_item_ids.values())},
                "assignment_count": sum(self.donor_item_ids.values()),
                "unique_original_item_ids": len(donor_ids_sorted),
                "original_item_ids_sha256": sha256_text(
                    _canonical_json(donor_ids_sorted)
                ),
                "most_frequent_original_item_ids": [
                    {"item_id": item_id, "assignments": count}
                    for item_id, count in sorted(
                        self.donor_item_ids.items(),
                        key=lambda pair: (-pair[1], pair[0]),
                    )[:20]
                ],
                "unique_source_request_ids": len(donor_requests_sorted),
                "source_request_ids_sha256": sha256_text(
                    _canonical_json(donor_requests_sorted)
                ),
                "lineage_sha256": self.donor_lineage_hasher.hexdigest(),
                "assignments_by_condition": dict(sorted(self.donor_by_condition.items())),
                "original_timestamp_preserved": self.donor_timestamp_preserved,
                "timestamp_adjusted_to_precede_target": self.donor_timestamp_adjusted,
            },
        }


def generate_history_interventions(
    train_records: Iterable[Mapping[str, Any]],
    dev_records: Iterable[Mapping[str, Any]],
    feature_lookup: FeatureLookup,
    *,
    seed: int = MECHANISM_INTERVENTION_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """In-memory entry point intended for tests and small diagnostic fixtures."""

    rows = {condition: [] for condition in CONDITION_IDS}

    def emit(condition_id: str, row: dict[str, Any]) -> None:
        rows[condition_id].append(row)

    audit = _run_interventions(
        train_records,
        dev_records,
        feature_lookup,
        emit=emit,
        seed=seed,
    )
    return rows, audit


def materialize_history_interventions(
    records_train_path: str | Path,
    records_dev_path: str | Path,
    feature_store_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = MECHANISM_INTERVENTION_SEED,
    feature_lookup: FeatureLookup | None = None,
) -> dict[str, Any]:
    """Stream the six internal-dev assignments and write an evidence manifest.

    ``feature_lookup`` is injectable for a tiny test fixture.  Production calls
    omit it and use a fully fingerprint-verified ``FrozenTextFeatureStore``.
    Even with injection, the feature-store metadata and its qrels boundary are
    validated so the manifest retains a concrete BGE identity.
    """

    if seed != MECHANISM_INTERVENTION_SEED:
        raise ValueError(
            f"M1 intervention seed is frozen at {MECHANISM_INTERVENTION_SEED}"
        )
    train_path = _require_population_file(records_train_path, "records_train.jsonl")
    dev_path = _require_population_file(records_dev_path, "records_dev.jsonl")
    if train_path == dev_path:
        raise ValueError("train and internal-dev record paths must differ")
    standardized_dir = dev_path.parent
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    for path in (
        dataset_manifest_path,
        candidate_manifest_path,
        request_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source manifest: {path}")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_manifest, dict):
        raise ValueError("frozen dataset manifest must be an object")
    if str(dataset_manifest.get("dataset_id")) != "kuaisearch":
        raise ValueError("M1 interventions are restricted to KuaiSearch")
    dataset_version = str(dataset_manifest.get("dataset_version") or "")
    if not dataset_version:
        raise ValueError("frozen dataset manifest lacks dataset_version")
    feature_root = Path(feature_store_path).resolve()
    metadata_path = feature_root / "metadata.json"
    index_path = feature_root / "index.json"
    vectors_path = feature_root / "vectors.npy"
    if not metadata_path.is_file() or not index_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError("frozen BGE feature store is incomplete")

    store: FrozenTextFeatureStore | None = None
    if feature_lookup is None:
        store = FrozenTextFeatureStore(feature_root, require_fingerprints=True)
        feature_lookup = store
        metadata = store.metadata
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("frozen BGE metadata must be an object")
    train_sha256 = sha256_file(train_path)
    dev_sha256 = sha256_file(dev_path)
    _validate_feature_metadata(
        metadata,
        train_sha256=train_sha256,
        dev_sha256=dev_sha256,
    )

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"intervention output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {condition: output / f"{condition}.jsonl" for condition in CONDITION_IDS}
    temporary_paths = {
        condition: output / f".{condition}.jsonl.partial"
        for condition in CONDITION_IDS
    }
    handles: dict[str, TextIO] = {}
    try:
        for condition, path in temporary_paths.items():
            handles[condition] = path.open("x", encoding="utf-8")

        def emit(condition_id: str, row: dict[str, Any]) -> None:
            handles[condition_id].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

        audit = _run_interventions(
            iter_jsonl(train_path),
            iter_jsonl(dev_path),
            feature_lookup,
            emit=emit,
            seed=seed,
        )
    finally:
        for handle in handles.values():
            handle.close()
    integrity = audit["integrity"]
    if any(int(value) != 0 for value in integrity.values()):
        raise ValueError(f"intervention integrity audit failed: {integrity}")
    for condition in CONDITION_IDS:
        temporary_paths[condition].replace(paths[condition])

    conditions: dict[str, Any] = {}
    for condition in CONDITION_IDS:
        conditions[condition] = {
            "path": str(paths[condition]),
            "sha256": sha256_file(paths[condition]),
            "count": audit["requests"],
            "request_count": audit["requests"],
            "request_ids_sha256": audit["request_ids_sha256"],
            "candidate_leakage_violations": integrity[
                "candidate_leakage_count"
            ],
            "causality_violations": integrity["causality_violation_count"],
            **audit["conditions"][condition],
        }
    store_fingerprint = metadata.get("store_fingerprint", {})
    encoder_fingerprint = metadata.get("encoder_fingerprint", {})
    manifest = {
        "schema_version": 1,
        "probe_id": "m1_history_interventions_v1",
        "population_role": "train_only_internal_dev",
        "dataset_id": "kuaisearch",
        "dataset_version": dataset_version,
        "split": "dev",
        "seed": seed,
        "history_budget": HISTORY_BUDGET,
        "semantic_shortlist_size": SEMANTIC_SHORTLIST_SIZE,
        "catalog_source": "train_history_only",
        "condition_order": list(CONDITION_IDS),
        "conditions": conditions,
        "inputs": {
            "source_dev": {"path": str(dev_path), "sha256": dev_sha256},
            "train_donor": {"path": str(train_path), "sha256": train_sha256},
            "frozen_bge": {
                "path": str(feature_root),
                "metadata_sha256": sha256_file(metadata_path),
                "index_sha256": str(metadata.get("index_sha256")),
                "vectors_sha256": str(metadata.get("vectors_sha256")),
                "store_fingerprint_sha256": (
                    str(store_fingerprint.get("sha256"))
                    if isinstance(store_fingerprint, Mapping)
                    else None
                ),
                "encoder_fingerprint_sha256": (
                    str(encoder_fingerprint.get("sha256"))
                    if isinstance(encoder_fingerprint, Mapping)
                    else None
                ),
                "qrels_read": metadata.get("qrels_read"),
            },
        },
        # Flat identities make downstream scorer admission intentionally simple.
        "source_records_sha256": dev_sha256,
        "train_records_sha256": train_sha256,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source": {
            "dataset_id": "kuaisearch",
            "dataset_version": dataset_version,
            "split": "dev",
            "records_path": str(dev_path),
            "records_sha256": dev_sha256,
            "candidate_manifest_path": str(candidate_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "request_manifest_path": str(request_manifest_path),
            "request_manifest_sha256": sha256_file(request_manifest_path),
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        },
        "bge_store_fingerprint_sha256": (
            str(store_fingerprint.get("sha256"))
            if isinstance(store_fingerprint, Mapping)
            else None
        ),
        "request_coverage": {
            "source_request_count": audit["requests"],
            "source_request_ids_sha256": audit["request_ids_sha256"],
            "all_conditions_exact": True,
            "missing_requests": 0,
            "extra_requests": 0,
        },
        "query_candidate_immutability": {
            "query_binding_sha256": audit["query_binding_sha256"],
            "candidate_binding_sha256": audit["candidate_binding_sha256"],
            "query_candidate_binding_sha256": (
                audit["query_candidate_binding_sha256"]
            ),
            "assignment_payload_fields": [
                "request_id",
                "condition_id",
                "history",
            ],
            "query_changed_rows": 0,
            "candidate_changed_rows": 0,
            "proof": (
                "query and candidates are absent from assignment payloads and "
                "remain request-id-bound to the hashed source dev records"
            ),
        },
        "forbidden_field_count": integrity["forbidden_field_count"],
        "candidate_leakage_count": integrity["candidate_leakage_count"],
        "causality_violation_count": integrity["causality_violation_count"],
        "candidate_leakage_violations": integrity["candidate_leakage_count"],
        "target_candidate_leakage_violations": integrity[
            "candidate_leakage_count"
        ],
        "causality_violations": integrity["causality_violation_count"],
        "history_not_strictly_before_target_violations": integrity[
            "causality_violation_count"
        ],
        "integrity": integrity,
        "condition_audit": audit["conditions"],
        "donor_catalog": audit["donor_catalog"],
        "donor_audit": audit["donor_audit"],
        "qrels_read": False,
        "model_scores_read": False,
        "confirmation_records_read": False,
        "source_test_opened": False,
        "implementation": {
            "path": "src/myrec/mechanism/history_interventions.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "replacement_contract": {
            "donor_population": "records_train.jsonl history events only",
            "donor_exclusions": [
                "all current candidate item IDs",
                "all original history item IDs",
            ],
            "preserving_pool_order": [
                "same normalized brand plus exact complete category path",
                "exact complete category path fallback",
            ],
            "preserving_selection": (
                "highest frozen-BGE canonical-item cosine from seeded cyclic "
                "shortlist64; identical normalized title excluded"
            ),
            "breaking_constraints": [
                "different top-level category",
                "different normalized brand",
            ],
            "breaking_selection": (
                "seeded cyclic shortlist64, lowest-cosine quartile, then minimum "
                "visible-character delta"
            ),
            "donor_timestamp": (
                "preserve original train timestamp when already causal; otherwise "
                "clamp to immediately before target request and retain original "
                "timestamp in donor lineage aggregate"
            ),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _run_interventions(
    train_records: Iterable[Mapping[str, Any]],
    dev_records: Iterable[Mapping[str, Any]],
    feature_lookup: FeatureLookup,
    *,
    emit: Callable[[str, dict[str, Any]], None],
    seed: int,
) -> dict[str, Any]:
    catalog = build_train_donor_catalog(train_records)
    engine = HistoryInterventionEngine(catalog, feature_lookup, seed=seed)
    audit = _RunAudit()
    for raw_record in dev_records:
        record = _sanitize_dev_record(raw_record)
        outcomes = engine.apply(record)
        audit.add_request(record, outcomes)
        for condition_id in CONDITION_IDS:
            assignment = {
                "request_id": record["request_id"],
                "condition_id": condition_id,
                "history": list(outcomes[condition_id].history),
            }
            if set(assignment) != {"request_id", "condition_id", "history"}:
                raise AssertionError("assignment payload fields drifted")
            emit(condition_id, assignment)
    result = audit.finalize()
    result["donor_catalog"] = dict(catalog.audit)
    return result


def _sanitize_dev_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    request_id = _required_text(raw_record, "request_id", scope="dev record")
    query = _required_text(raw_record, "query", scope=f"dev request_id={request_id}")
    request_ts = _required_timestamp(raw_record, scope=f"dev request_id={request_id}")
    raw_history = raw_record.get("history")
    raw_candidates = raw_record.get("candidates")
    if not isinstance(raw_history, list):
        raise ValueError(f"dev request_id={request_id}: history must be a list")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"dev request_id={request_id}: candidates must be non-empty")
    history = [
        _project_item(row, HISTORY_INPUT_FIELDS, scope="dev history")
        for row in raw_history
    ]
    candidates = [
        _project_item(row, CANDIDATE_INPUT_FIELDS, scope="dev candidate")
        for row in raw_candidates
    ]
    for event in history:
        _required_text(event, "item_id", scope="dev history")
        if _required_timestamp(event, scope="dev history") >= request_ts:
            raise ValueError(
                f"dev request_id={request_id}: source history is not strictly causal"
            )
    candidate_ids = [
        _required_text(row, "item_id", scope="dev candidate") for row in candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"dev request_id={request_id}: duplicate candidate item_id")
    return {
        "request_id": request_id,
        "query": query,
        "ts": request_ts,
        "history": history,
        "candidates": candidates,
    }


def _causal_donor_history(
    donor: DonorEvent, *, request_ts: int | float
) -> tuple[dict[str, Any], bool]:
    emitted = _copy_value(donor.history)
    if donor.original_ts < request_ts:
        emitted["ts"] = donor.original_ts
        return emitted, False
    if isinstance(request_ts, int) and not isinstance(request_ts, bool):
        emitted_ts: int | float = request_ts - 1
    else:
        emitted_ts = math.nextafter(float(request_ts), -math.inf)
    if not emitted_ts < request_ts:
        raise AssertionError("could not make donor timestamp strictly causal")
    emitted["ts"] = emitted_ts
    return emitted, True


def _validate_feature_metadata(
    metadata: Mapping[str, Any], *, train_sha256: str, dev_sha256: str
) -> None:
    if metadata.get("qrels_read") is not False:
        raise ValueError("frozen BGE feature store crossed the qrels boundary")
    if metadata.get("feature_contract") != "frozen_transformer_cls_l2_v1":
        raise ValueError("unexpected frozen BGE feature contract")
    if (
        metadata.get("visible_text_contract")
        != "query_context_and_canonical_item_semantics_v2"
    ):
        raise ValueError("frozen BGE store lacks required canonical item semantics")
    record_hashes = {
        str(row.get("sha256"))
        for row in metadata.get("record_files", [])
        if isinstance(row, Mapping) and row.get("sha256")
    }
    missing = {train_sha256, dev_sha256} - record_hashes
    if missing:
        raise ValueError("frozen BGE store is not bound to supplied train/dev records")
    for field in ("index_sha256", "vectors_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"frozen BGE metadata lacks {field}")
    for field in ("encoder_fingerprint", "store_fingerprint"):
        value = metadata.get(field)
        if not isinstance(value, Mapping) or not isinstance(value.get("sha256"), str):
            raise ValueError(f"frozen BGE metadata lacks {field}")


def _require_population_file(path: str | Path, expected_name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing population records: {resolved}")
    if resolved.name != expected_name:
        raise ValueError(
            f"M1 accepts only {expected_name}; refusing {resolved.name}"
        )
    return resolved


def _project_item(
    row: Mapping[str, Any], fields: Sequence[str], *, scope: str
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{scope} row must be an object")
    return {key: _copy_value(row[key]) for key in fields if key in row}


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_value(row) for row in value]
    if isinstance(value, tuple):
        return [_copy_value(row) for row in value]
    if isinstance(value, Mapping):
        return {str(key): _copy_value(row) for key, row in value.items()}
    return value


def _required_text(row: Mapping[str, Any], key: str, *, scope: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{scope}: missing or empty {key}")
    return value


def _required_timestamp(row: Mapping[str, Any], *, scope: str) -> int | float:
    value = row.get("ts")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{scope}: ts must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{scope}: ts must be finite")
    return value


def _normalized_text(value: Any) -> str:
    text = str(value).strip().casefold() if value is not None else ""
    return "" if text.upper() == "UNKNOWN" else text


def _normalized_title(value: Any) -> str:
    return " ".join(_normalized_text(value).split())


def _category_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("cat", [])
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tuple(
        text
        for part in value
        if (text := _normalized_text(part))
    )


def _visible_history_characters(history: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(serialize_item_content(row)) for row in history)


def _stable_digest(*parts: Any) -> str:
    return sha256_text("\x1f".join(str(value) for value in parts))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _update_digest(digest: Any, value: Any) -> None:
    digest.update(_canonical_json(value).encode("utf-8"))
    digest.update(b"\n")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        position = int(round((len(ordered) - 1) * fraction))
        return ordered[position]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }
