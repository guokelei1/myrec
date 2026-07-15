"""PPS scorer around the locked official HSTU or matched SASRec core.

The upstream core is imported lazily from ``baselines/hstu``. Callers must run
in the dedicated HSTU environment and put that directory on ``PYTHONPATH``.
This module is baseline code, not a proposed architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from myrec.baselines.representative_sequence_adapter import SequenceRequest


SUPPORTED_SEQUENCE_CORES = ("hstu", "sasrec")


def ensure_fbgemm_sparse_ops() -> dict[str, str | bool]:
    """Ensure the one sparse op not always registered by fbgemm 1.1 is loaded."""

    import fbgemm_gpu

    if hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        return {"manual_load": False, "library": "already_registered"}
    library = Path(fbgemm_gpu.__file__).parent / "fbgemm_gpu_sparse_async_cumsum.so"
    if not library.exists():
        raise FileNotFoundError(f"missing fbgemm sparse cumsum library: {library}")
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        raise RuntimeError("failed to register fbgemm asynchronous_complete_cumsum")
    return {"manual_load": True, "library": str(library)}


def build_official_sequence_core(
    architecture: str,
    *,
    num_item_ids: int,
    embedding_dim: int,
    max_sequence_length: int,
    num_blocks: int,
    num_heads: int,
    dropout_rate: float,
):
    """Instantiate HSTU or SASRec with identical external dimensions."""

    if architecture not in SUPPORTED_SEQUENCE_CORES:
        raise ValueError(
            f"unsupported sequence core={architecture}; expected {SUPPORTED_SEQUENCE_CORES}"
        )
    if num_item_ids <= 3:
        raise ValueError("num_item_ids must include special IDs and at least one item")
    if embedding_dim <= 0 or max_sequence_length <= 0:
        raise ValueError("embedding_dim and max_sequence_length must be positive")
    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")
    ensure_fbgemm_sparse_ops()

    from generative_recommenders.research.modeling.sequential.embedding_modules import (
        LocalEmbeddingModule,
    )
    from generative_recommenders.research.modeling.sequential.hstu import HSTU
    from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
        LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    )
    from generative_recommenders.research.modeling.sequential.output_postprocessors import (
        L2NormEmbeddingPostprocessor,
    )
    from generative_recommenders.research.modeling.sequential.sasrec import SASRec
    from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
        DotProductSimilarity,
    )

    # LocalEmbeddingModule creates num_items + 1 rows. Our vocabulary count is
    # already max_id + 1, so pass max_id here.
    item_embedding = LocalEmbeddingModule(
        num_items=num_item_ids - 1, item_embedding_dim=embedding_dim
    )
    common = {
        "max_sequence_len": max_sequence_length,
        "max_output_len": 0,
        "embedding_dim": embedding_dim,
        "embedding_module": item_embedding,
        "similarity_module": DotProductSimilarity(),
        "input_features_preproc_module": LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=max_sequence_length,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        ),
        "output_postproc_module": L2NormEmbeddingPostprocessor(embedding_dim),
        "verbose": False,
    }
    if architecture == "hstu":
        return HSTU(
            **common,
            num_blocks=num_blocks,
            num_heads=num_heads,
            linear_dim=embedding_dim,
            attention_dim=embedding_dim // num_heads,
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=dropout_rate,
            attn_dropout_rate=dropout_rate,
        )
    return SASRec(
        **common,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ffn_hidden_dim=embedding_dim,
        ffn_activation_fn="relu",
        ffn_dropout_rate=dropout_rate,
        activation_checkpoint=False,
    )


@dataclass(frozen=True)
class SequenceBatch:
    request_ids: tuple[str, ...]
    raw_candidate_item_ids: tuple[tuple[str, ...], ...]
    past_lengths: torch.Tensor
    past_item_ids: torch.Tensor
    past_event_ids: torch.Tensor
    past_timestamps: torch.Tensor
    past_content_features: torch.Tensor
    candidate_item_ids: torch.Tensor
    candidate_content_features: torch.Tensor
    candidate_mask: torch.Tensor

    def to(self, device: str | torch.device) -> "SequenceBatch":
        return SequenceBatch(
            request_ids=self.request_ids,
            raw_candidate_item_ids=self.raw_candidate_item_ids,
            past_lengths=self.past_lengths.to(device),
            past_item_ids=self.past_item_ids.to(device),
            past_event_ids=self.past_event_ids.to(device),
            past_timestamps=self.past_timestamps.to(device),
            past_content_features=self.past_content_features.to(device),
            candidate_item_ids=self.candidate_item_ids.to(device),
            candidate_content_features=self.candidate_content_features.to(device),
            candidate_mask=self.candidate_mask.to(device),
        )


def collate_sequence_requests(
    requests: Sequence[SequenceRequest],
    feature_lookup: Callable[[str], Sequence[float] | torch.Tensor],
    *,
    content_dim: int,
    max_sequence_length: int,
) -> SequenceBatch:
    """Pad shared sequence requests without changing candidate identity/order."""

    if not requests:
        raise ValueError("cannot collate an empty request batch")
    if content_dim <= 0 or max_sequence_length <= 0:
        raise ValueError("content_dim and max_sequence_length must be positive")
    max_candidates = max(len(request.candidates) for request in requests)
    batch_size = len(requests)
    past_item_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
    past_event_ids = torch.zeros_like(past_item_ids)
    past_timestamps = torch.zeros_like(past_item_ids)
    past_content = torch.zeros(
        batch_size, max_sequence_length, content_dim, dtype=torch.float32
    )
    candidate_item_ids = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    candidate_content = torch.zeros(
        batch_size, max_candidates, content_dim, dtype=torch.float32
    )
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    lengths: list[int] = []
    raw_candidate_ids: list[tuple[str, ...]] = []
    for batch_index, request in enumerate(requests):
        sequence_length = len(request.past_item_ids)
        if sequence_length > max_sequence_length:
            raise ValueError(
                f"request_id={request.request_id}: sequence length {sequence_length} "
                f"exceeds max_sequence_length={max_sequence_length}"
            )
        if sequence_length < 1:
            raise ValueError(f"request_id={request.request_id}: empty causal sequence")
        lengths.append(sequence_length)
        past_item_ids[batch_index, :sequence_length] = torch.tensor(
            request.past_item_ids, dtype=torch.long
        )
        past_event_ids[batch_index, :sequence_length] = torch.tensor(
            request.past_event_ids, dtype=torch.long
        )
        past_timestamps[batch_index, :sequence_length] = torch.tensor(
            request.past_timestamps, dtype=torch.long
        )
        for token_index, text in enumerate(request.past_content_texts):
            past_content[batch_index, token_index] = _feature_tensor(
                feature_lookup(text), content_dim
            )
        candidate_count = len(request.candidates)
        candidate_mask[batch_index, :candidate_count] = True
        candidate_item_ids[batch_index, :candidate_count] = torch.tensor(
            [candidate.item_id for candidate in request.candidates], dtype=torch.long
        )
        for candidate_index, candidate in enumerate(request.candidates):
            candidate_content[batch_index, candidate_index] = _feature_tensor(
                feature_lookup(candidate.content_text), content_dim
            )
        raw_candidate_ids.append(
            tuple(candidate.raw_item_id for candidate in request.candidates)
        )
    return SequenceBatch(
        request_ids=tuple(request.request_id for request in requests),
        raw_candidate_item_ids=tuple(raw_candidate_ids),
        past_lengths=torch.tensor(lengths, dtype=torch.long),
        past_item_ids=past_item_ids,
        past_event_ids=past_event_ids,
        past_timestamps=past_timestamps,
        past_content_features=past_content,
        candidate_item_ids=candidate_item_ids,
        candidate_content_features=candidate_content,
        candidate_mask=candidate_mask,
    )


class HSTUPPSRanker(nn.Module):
    """Matched fixed-slate scorer differing only in HSTU versus SASRec core."""

    def __init__(
        self,
        *,
        architecture: str,
        num_item_ids: int,
        num_event_ids: int,
        content_dim: int,
        embedding_dim: int,
        max_sequence_length: int,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if num_event_ids <= 3:
            raise ValueError("num_event_ids must include special IDs and an event")
        self.architecture = architecture
        self.content_dim = content_dim
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.sequence_core = build_official_sequence_core(
            architecture,
            num_item_ids=num_item_ids,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.event_embedding = nn.Embedding(
            num_event_ids, embedding_dim, padding_idx=0
        )
        self.content_projection = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_sequence(self, batch: SequenceBatch) -> torch.Tensor:
        if batch.past_item_ids.size(1) != self.max_sequence_length:
            raise ValueError("batch sequence width does not match model maximum")
        item_embeddings = self.sequence_core.get_item_embeddings(batch.past_item_ids)
        event_embeddings = self.event_embedding(batch.past_event_ids)
        content_embeddings = self.content_projection(batch.past_content_features)
        valid_tokens = (batch.past_item_ids != 0).unsqueeze(-1)
        sequence_embeddings = (
            item_embeddings + event_embeddings + content_embeddings
        ) * valid_tokens
        return self.sequence_core.encode(
            past_lengths=batch.past_lengths,
            past_ids=batch.past_item_ids,
            past_embeddings=sequence_embeddings,
            past_payloads={"timestamps": batch.past_timestamps},
        )

    def encode_candidates(self, batch: SequenceBatch) -> torch.Tensor:
        candidate_embeddings = self.sequence_core.get_item_embeddings(
            batch.candidate_item_ids
        ) + self.content_projection(batch.candidate_content_features)
        return F.normalize(candidate_embeddings, dim=-1)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        user_query_embedding = F.normalize(self.encode_sequence(batch), dim=-1)
        candidate_embeddings = self.encode_candidates(batch)
        scale = self.logit_scale.exp().clamp(max=100.0)
        scores = torch.bmm(
            candidate_embeddings, user_query_embedding.unsqueeze(-1)
        ).squeeze(-1) * scale
        # Keep padded scores finite; the training/scoring caller must apply the
        # explicit mask. Infinite padding can contaminate listwise losses.
        return scores.masked_fill(~batch.candidate_mask, 0.0)


def _feature_tensor(
    value: Sequence[float] | torch.Tensor, expected_dim: int
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() != expected_dim:
        raise ValueError(
            f"content feature has {tensor.numel()} values; expected {expected_dim}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("content feature contains a non-finite value")
    return tensor
