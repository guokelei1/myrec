"""Independent paper-mechanism implementation of the LLM-SRec baseline.

The implementation follows the retrieval, MSE distillation, and uniformity
objectives in *Lost in Sequence: Do Large Language Models Understand Sequential
Recommendation?* (KDD 2025, arXiv:2502.13909v4). It does not copy or import the
unlicensed official repository. The true query added to the user prompt and
fixed-slate candidate scoring are explicit PPS task-interface adaptations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from myrec.baselines.representative_sequence_adapter import (
    SequenceCandidate,
    SequenceRequest,
)


HISTORY_EMBED_TOKEN = "[PPSHistoryEmb]"
USER_OUTPUT_TOKEN = "[PPSUserOut]"
ITEM_OUTPUT_TOKEN = "[PPSItemOut]"


class TwoLayerProjection(nn.Module):
    """Two-layer MLP used for the paper's trainable projection functions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("projection dimensions must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def uniformity_loss(representations: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 3 on normalized representations, excluding self-pairs."""

    if representations.ndim != 2:
        raise ValueError("uniformity input must have shape [batch, dimension]")
    if representations.size(0) < 2:
        # A differentiable zero keeps small final batches valid.
        return representations.sum() * 0.0
    normalized = F.normalize(representations, dim=-1)
    pairwise_squared_distance = torch.pdist(normalized, p=2).square()
    return torch.exp(-2.0 * pairwise_squared_distance).mean()


@dataclass(frozen=True)
class LLMSRecLosses:
    total: torch.Tensor
    retrieval: torch.Tensor
    distillation: torch.Tensor
    uniformity: torch.Tensor


class LLMSRecRetrievalHead(nn.Module):
    """Paper Eq. 1--4 on frozen LLM and frozen CF-SRec representations."""

    def __init__(
        self,
        *,
        llm_dim: int,
        cf_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        retrieval_weight: float = 1.0,
        distillation_weight: float = 1.0,
        uniformity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.user_projection = TwoLayerProjection(
            llm_dim, hidden_dim, projection_dim
        )
        self.item_projection = TwoLayerProjection(
            llm_dim, hidden_dim, projection_dim
        )
        self.cf_user_projection = TwoLayerProjection(
            cf_dim, hidden_dim, projection_dim
        )
        self.retrieval_weight = retrieval_weight
        self.distillation_weight = distillation_weight
        self.uniformity_weight = uniformity_weight

    def score(
        self, llm_user: torch.Tensor, llm_items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if llm_user.ndim != 2 or llm_items.ndim != 3:
            raise ValueError("expected llm_user [B,D] and llm_items [B,C,D]")
        if llm_user.size(0) != llm_items.size(0):
            raise ValueError("user and item batch dimensions differ")
        projected_user = self.user_projection(llm_user)
        projected_items = self.item_projection(llm_items)
        scores = torch.bmm(
            projected_items, projected_user.unsqueeze(-1)
        ).squeeze(-1)
        return scores, projected_user, projected_items

    def losses(
        self,
        *,
        llm_user: torch.Tensor,
        llm_items: torch.Tensor,
        cf_user: torch.Tensor,
        positive_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, LLMSRecLosses]:
        scores, projected_user, _ = self.score(llm_user, llm_items)
        if candidate_mask.shape != scores.shape:
            raise ValueError("candidate mask shape does not match scores")
        if positive_indices.shape != (scores.size(0),):
            raise ValueError("positive_indices must have shape [batch]")
        if not bool(
            candidate_mask.gather(1, positive_indices.reshape(-1, 1)).all()
        ):
            raise ValueError("a positive index points to a padded candidate")
        retrieval = F.cross_entropy(
            scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min),
            positive_indices,
        )
        # The paper aligns unit-normalized LLM and CF-SRec user
        # representations. Keep the unnormalized projections for retrieval,
        # but normalize both representations for matching.
        projected_cf_user = self.cf_user_projection(cf_user.detach())
        normalized_user = F.normalize(projected_user, dim=-1)
        normalized_cf_user = F.normalize(projected_cf_user, dim=-1)
        distillation = F.mse_loss(normalized_user, normalized_cf_user)
        uniformity = uniformity_loss(projected_user) + uniformity_loss(
            projected_cf_user
        )
        total = (
            self.retrieval_weight * retrieval
            + self.distillation_weight * distillation
            + self.uniformity_weight * uniformity
        )
        return scores, LLMSRecLosses(
            total=total,
            retrieval=retrieval,
            distillation=distillation,
            uniformity=uniformity,
        )


def build_user_prompt(request: SequenceRequest) -> str:
    """Build the paper-style chronological user prompt with a PPS query."""

    history_lines = []
    for index, text in enumerate(
        request.past_content_texts[: request.retained_history_count], start=1
    ):
        timestamp = request.past_timestamps[index - 1]
        history_lines.append(
            f"No.{index}; time={timestamp}; {text}; embedding={HISTORY_EMBED_TOKEN}"
        )
    history = "\n".join(history_lines) if history_lines else "No prior interactions."
    return (
        "Rank products for the current search request.\n"
        f"Current query: {request.query}\n"
        "The user history is chronological:\n"
        f"{history}\n"
        f"Generate the query-conditioned user representation: {USER_OUTPUT_TOKEN}"
    )


def build_item_prompt(candidate: SequenceCandidate) -> str:
    return (
        f"Candidate product: {candidate.content_text}; "
        f"collaborative embedding={HISTORY_EMBED_TOKEN}; "
        f"generate the item representation: {ITEM_OUTPUT_TOKEN}"
    )


class FrozenQwenLLMSRecEncoder(nn.Module):
    """Frozen Qwen backbone plus the paper's trainable item-injection projection."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        cf_item_dim: int,
        max_length: int = 1024,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype | str = "auto",
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        added = self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    HISTORY_EMBED_TOKEN,
                    USER_OUTPUT_TOKEN,
                    ITEM_OUTPUT_TOKEN,
                ]
            }
        )
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        )
        if added:
            self.backbone.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.max_length = max_length
        self.llm_dim = int(self.backbone.config.hidden_size)
        self.cf_item_projection = TwoLayerProjection(
            cf_item_dim, max(256, cf_item_dim * 2), self.llm_dim
        )
        input_embeddings = self.backbone.get_input_embeddings().weight.detach()
        self.user_output_embedding = nn.Parameter(
            input_embeddings[self._token_id(USER_OUTPUT_TOKEN)].float().clone()
        )
        self.item_output_embedding = nn.Parameter(
            input_embeddings[self._token_id(ITEM_OUTPUT_TOKEN)].float().clone()
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen LLM dropout must stay disabled while lightweight modules train.
        self.backbone.eval()
        return self

    def encode_users(
        self,
        requests: Sequence[SequenceRequest],
        cf_history_items: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cf_history_items.ndim != 3 or history_mask.ndim != 2:
            raise ValueError("expected CF history [B,H,D] and mask [B,H]")
        if cf_history_items.shape[:2] != history_mask.shape:
            raise ValueError("CF history and history mask shapes differ")
        expected_lengths = torch.tensor(
            [request.retained_history_count for request in requests],
            device=history_mask.device,
        )
        if not torch.equal(history_mask.sum(dim=1), expected_lengths):
            raise ValueError("CF history mask does not match request history lengths")
        projected = self.cf_item_projection(cf_history_items)
        replacement_rows = [
            projected[index, history_mask[index]] for index in range(len(requests))
        ]
        return self._encode_prompts(
            [build_user_prompt(request) for request in requests],
            replacement_rows=replacement_rows,
            output_token=USER_OUTPUT_TOKEN,
            output_embedding=self.user_output_embedding,
        )

    def encode_items(
        self,
        candidates: Sequence[SequenceCandidate],
        cf_items: torch.Tensor,
    ) -> torch.Tensor:
        if cf_items.ndim != 2 or cf_items.size(0) != len(candidates):
            raise ValueError("expected one CF item embedding per candidate")
        projected = self.cf_item_projection(cf_items)
        return self._encode_prompts(
            [build_item_prompt(candidate) for candidate in candidates],
            replacement_rows=[row.unsqueeze(0) for row in projected],
            output_token=ITEM_OUTPUT_TOKEN,
            output_embedding=self.item_output_embedding,
        )

    def _encode_prompts(
        self,
        prompts: Sequence[str],
        *,
        replacement_rows: Sequence[torch.Tensor],
        output_token: str,
        output_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if len(prompts) != len(replacement_rows):
            raise ValueError("prompt and replacement batch sizes differ")
        device = next(self.cf_item_projection.parameters()).device
        tokens = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        inputs_embeds = self.backbone.get_input_embeddings()(tokens["input_ids"])
        history_id = self._token_id(HISTORY_EMBED_TOKEN)
        output_id = self._token_id(output_token)
        output_positions: list[int] = []
        for row_index in range(tokens["input_ids"].size(0)):
            history_positions = torch.nonzero(
                tokens["input_ids"][row_index] == history_id, as_tuple=False
            ).reshape(-1)
            replacements = replacement_rows[row_index]
            if history_positions.numel() != replacements.size(0):
                raise ValueError(
                    "prompt truncation or serialization changed the number of "
                    f"collaborative placeholders in row={row_index}"
                )
            if history_positions.numel():
                inputs_embeds[row_index, history_positions] = replacements.to(
                    inputs_embeds.dtype
                )
            positions = torch.nonzero(
                tokens["input_ids"][row_index] == output_id, as_tuple=False
            ).reshape(-1)
            if positions.numel() != 1:
                raise ValueError(
                    f"expected one {output_token} token in row={row_index}; "
                    "the prompt may have been truncated"
                )
            position = int(positions.item())
            inputs_embeds[row_index, position] = output_embedding.to(
                inputs_embeds.dtype
            )
            output_positions.append(position)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=tokens["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        positions_tensor = torch.tensor(output_positions, device=device)
        row_tensor = torch.arange(len(prompts), device=device)
        return outputs.last_hidden_state[row_tensor, positions_tensor].float()

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"tokenizer did not register special token {token}")
        return int(token_id)
