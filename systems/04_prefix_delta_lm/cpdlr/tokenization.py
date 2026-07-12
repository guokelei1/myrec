"""Deterministic paired-prefix token construction for C04."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import torch

from .io import stable_hash


MARKERS = {
    "query": "[unused90]",
    "history": "[unused91]",
    "candidate": "[unused92]",
    "click": "[unused93]",
    "purchase": "[unused94]",
    "null": "[unused95]",
    "event_sep": "[unused96]",
    "brand": "[unused97]",
    "category": "[unused98]",
}


class PrefixTokenizer:
    """Build `[q,H,c]` and `[q,NULL_HISTORY,c]` with fixed field budgets.

    Item identity is represented by three stable `[unused1..80]` tokens.  The
    same mapping is used for history and candidate items; it exposes exact
    recurrence without introducing a learned item table or a generated ID.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        query_tokens: int,
        candidate_tokens: int,
        max_history_events: int,
        event_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.query_tokens = int(query_tokens)
        self.candidate_tokens = int(candidate_tokens)
        self.max_history_events = int(max_history_events)
        self.event_tokens = int(event_tokens)
        self.cls_id = int(tokenizer.cls_token_id)
        self.sep_id = int(tokenizer.sep_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        self.marker_ids = {
            name: int(tokenizer.convert_tokens_to_ids(token))
            for name, token in MARKERS.items()
        }
        if any(value == int(tokenizer.unk_token_id) for value in self.marker_ids.values()):
            raise ValueError("BGE tokenizer does not expose the frozen [unused] markers")

    @lru_cache(maxsize=250_000)
    def _encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
        )

    @lru_cache(maxsize=250_000)
    def _identity_tokens(self, item_id: str) -> tuple[int, int, int]:
        digest = bytes.fromhex(stable_hash("c04_item", item_id))
        tokens = []
        for byte in digest[:3]:
            marker = f"[unused{1 + byte % 80}]"
            token_id = int(self.tokenizer.convert_tokens_to_ids(marker))
            if token_id == int(self.tokenizer.unk_token_id):
                raise ValueError(f"missing identity marker: {marker}")
            tokens.append(token_id)
        return tuple(tokens)  # type: ignore[return-value]

    @staticmethod
    def _item_text(item: dict[str, Any], coarse_only: bool = False) -> str:
        categories = " ".join(str(value) for value in item.get("cat", []) if value)
        if coarse_only:
            return categories
        return " ".join(
            value
            for value in (
                str(item.get("title", "")),
                str(item.get("brand", "")),
                categories,
            )
            if value
        )

    def _event_ids(self, event: dict[str, Any], coarse_only: bool) -> list[int]:
        marker_name = "purchase" if str(event.get("event")) == "purchase" else "click"
        result = [self.marker_ids[marker_name]]
        if not coarse_only:
            result.extend(self._identity_tokens(str(event.get("item_id", ""))))
        result.extend(
            self._encode_text(self._item_text(event, coarse_only=coarse_only))[
                : self.event_tokens
            ]
        )
        result.append(self.marker_ids["event_sep"])
        return result

    def _candidate_ids(self, candidate: dict[str, Any]) -> list[int]:
        result = list(self._identity_tokens(str(candidate.get("item_id", ""))))
        result.extend(
            self._encode_text(self._item_text(candidate))[: self.candidate_tokens]
        )
        return result

    @staticmethod
    def _shuffled(history: list[dict[str, Any]], request_id: str, seed: int) -> list[dict[str, Any]]:
        return sorted(
            history,
            key=lambda row: stable_hash(
                "c04_shuffle", seed, request_id, row.get("item_id"), row.get("ts")
            ),
        )

    def encode(
        self,
        record: dict[str, Any],
        candidate: dict[str, Any],
        prefix: str,
        seed: int,
        structured: bool = True,
    ) -> dict[str, list[int]]:
        """Encode one candidate under a factual/null/corrupted prefix."""

        if prefix not in {
            "factual",
            "null",
            "wrong",
            "shuffled",
            "query_masked",
            "query_masked_null",
            "coarse",
        }:
            raise ValueError(f"unknown prefix: {prefix}")
        query_masked = prefix in {"query_masked", "query_masked_null"}
        query_ids = [self.mask_id] if query_masked else list(
            self._encode_text(str(record.get("query", "")))[: self.query_tokens]
        )

        history = list(record.get("history", []))
        if prefix == "wrong":
            history = list(record.get("wrong_history", []))
        elif prefix == "shuffled":
            history = self._shuffled(history, str(record.get("request_id", "")), seed)
        elif prefix in {"null", "query_masked_null"}:
            history = []
        coarse_only = prefix == "coarse"

        # Empty factual history is intentionally byte-identical to the null prefix.
        if not record.get("history") and prefix in {
            "factual",
            "wrong",
            "shuffled",
            "coarse",
        }:
            history = []

        input_ids = [self.cls_id]
        token_types = [0]
        if structured:
            input_ids.append(self.marker_ids["query"])
            token_types.append(0)
        input_ids.extend(query_ids)
        token_types.extend([0] * len(query_ids))
        input_ids.append(self.sep_id)
        token_types.append(0)

        if structured:
            input_ids.append(self.marker_ids["history"])
            token_types.append(0)
        if history:
            for event in history[-self.max_history_events :]:
                event_ids = self._event_ids(event, coarse_only=coarse_only)
                input_ids.extend(event_ids)
                token_types.extend([0] * len(event_ids))
        else:
            input_ids.append(self.marker_ids["null"])
            token_types.append(0)
        input_ids.append(self.sep_id)
        token_types.append(0)

        if structured:
            input_ids.append(self.marker_ids["candidate"])
            token_types.append(1)
        candidate_ids = self._candidate_ids(candidate)
        input_ids.extend(candidate_ids)
        token_types.extend([1] * len(candidate_ids))
        input_ids.append(self.sep_id)
        token_types.append(1)

        if len(input_ids) > self.max_length:
            # Candidate and query budgets are fixed; overflow can only come from history.
            overflow = len(input_ids) - self.max_length
            history_start = 2 + len(query_ids) + 1 if structured else 1 + len(query_ids) + 1
            history_end = len(input_ids) - len(candidate_ids) - (3 if structured else 2)
            removable = max(history_end - history_start - 1, 0)
            cut = min(overflow, removable)
            del input_ids[history_start : history_start + cut]
            del token_types[history_start : history_start + cut]
        input_ids = input_ids[: self.max_length]
        token_types = token_types[: self.max_length]
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids.extend([self.pad_id] * padding)
        token_types.extend([0] * padding)
        attention_mask.extend([0] * padding)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_types,
        }

    def batch_encode(
        self,
        pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
        prefix: str,
        seed: int,
        structured: bool = True,
    ) -> dict[str, torch.Tensor]:
        rows = [
            self.encode(record, candidate, prefix, seed, structured=structured)
            for record, candidate in pairs
        ]
        if not rows:
            raise ValueError("cannot encode an empty candidate batch")
        return {
            key: torch.tensor([row[key] for row in rows], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }
