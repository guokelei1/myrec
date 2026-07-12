"""Executable, dataset-free G1 protocol for C08.

Every outcome-relevant constant is defined in ``FROZEN`` and duplicated in the
human-readable protocol amendment.  This module reads no files and chooses no
configuration from outcomes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reversible_memory import ReversibleCouplingMemory, _unit


Mechanism = Literal["rwpu", "ordinary", "attention", "pooled_ffn"]
MECHANISMS: tuple[Mechanism, ...] = (
    "rwpu",
    "ordinary",
    "attention",
    "pooled_ffn",
)


@dataclass(frozen=True)
class FrozenG1Config:
    seeds: tuple[int, ...] = (20260711, 20260712, 20260713)
    train_requests: int = 4096
    evaluation_requests: int = 1024
    repeat_fraction: float = 0.5
    candidate_count: int = 8
    history_length: int = 8
    evidence_width: int = 16
    topic_count: int = 4
    styles_per_topic: int = 2
    batch_size: int = 64
    optimizer_steps: int = 400
    learning_rate: float = 0.003
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    transformer_heads: int = 4
    transformer_ffn_width: int = 32
    dropout: float = 0.0
    repeat_noninferiority_pp: float = 0.01
    supported_best_control_advantage_pp: float = 0.05
    supported_ordinary_advantage_pp: float = 0.03
    corruption_margin_retention_max: float = 0.25
    permutation_max_error: float = 1e-6
    tie_salt: str = "c08_g1_tie_v1"


FROZEN = FrozenG1Config()

STREAM_OFFSETS: dict[str, int] = {
    "train_structure": 101,
    "train_values": 102,
    "train_schedule": 103,
    "eval_structure": 201,
    "eval_values": 202,
    "wrong_history": 301,
    "shuffle_history": 302,
    "model_initialization": 401,
}


@dataclass
class SyntheticSplit:
    query: Tensor
    history: Tensor
    history_mask: Tensor
    candidates: Tensor
    candidate_ids: Tensor
    target_index: Tensor
    kind: Tensor  # 0 = exact repeat, 1 = supported non-repeat
    query_topic: Tensor
    history_topic: Tensor
    history_style: Tensor
    request_index: Tensor

    def subset(self, index: Tensor) -> "SyntheticSplit":
        return SyntheticSplit(
            query=self.query[index],
            history=self.history[index],
            history_mask=self.history_mask[index],
            candidates=self.candidates[index],
            candidate_ids=self.candidate_ids[index],
            target_index=self.target_index[index],
            kind=self.kind[index],
            query_topic=self.query_topic[index],
            history_topic=self.history_topic[index],
            history_style=self.history_style[index],
            request_index=self.request_index[index],
        )

    def as_model_inputs(self) -> dict[str, Tensor]:
        return {
            "query": self.query,
            "history": self.history,
            "history_mask": self.history_mask,
            "candidates": self.candidates,
        }


def stream_seed(base_seed: int, stream_name: str) -> int:
    if stream_name not in STREAM_OFFSETS:
        raise KeyError(stream_name)
    return base_seed * 1000 + STREAM_OFFSETS[stream_name]


def make_generator(base_seed: int, stream_name: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(stream_seed(base_seed, stream_name))


def _normalize(vector: Tensor) -> Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)


def _candidate_vector(
    topic: int,
    style: int,
    values: torch.Generator,
) -> Tensor:
    vector = torch.zeros(FROZEN.evidence_width)
    topic_amplitude = 0.90 + 0.20 * torch.rand((), generator=values).item()
    style_amplitude = 0.65 + 0.20 * torch.rand((), generator=values).item()
    vector[topic] = topic_amplitude
    vector[FROZEN.topic_count + topic] = float(style) * style_amplitude
    return _normalize(vector)


def _nonrepeat_history_vector(
    topic: int,
    style: int,
    values: torch.Generator,
) -> Tensor:
    vector = torch.zeros(FROZEN.evidence_width)
    topic_amplitude = 0.97 + 0.06 * torch.rand((), generator=values).item()
    style_amplitude = 0.50 + 0.10 * torch.rand((), generator=values).item()
    vector[topic] = topic_amplitude
    vector[FROZEN.topic_count + topic] = float(style) * style_amplitude
    return _normalize(vector)


def _semantic_id(topic: int, style: int) -> int:
    return 2 * topic + (1 if style > 0 else 0)


def generate_split(base_seed: int, split: Literal["train", "eval"]) -> SyntheticSplit:
    """Generate the exact balanced clean split from named RNG streams."""

    if split == "train":
        request_count = FROZEN.train_requests
        structure = make_generator(base_seed, "train_structure")
        values = make_generator(base_seed, "train_values")
    elif split == "eval":
        request_count = FROZEN.evaluation_requests
        structure = make_generator(base_seed, "eval_structure")
        values = make_generator(base_seed, "eval_values")
    else:
        raise ValueError(split)

    repeat_count = int(request_count * FROZEN.repeat_fraction)
    kinds = torch.cat(
        (
            torch.zeros(repeat_count, dtype=torch.long),
            torch.ones(request_count - repeat_count, dtype=torch.long),
        )
    )
    kinds = kinds[torch.randperm(request_count, generator=structure)]

    queries: list[Tensor] = []
    histories: list[Tensor] = []
    candidates_all: list[Tensor] = []
    candidate_ids_all: list[Tensor] = []
    targets: list[int] = []
    query_topics: list[int] = []
    history_topics_all: list[Tensor] = []
    history_styles_all: list[Tensor] = []

    for request in range(request_count):
        query_topic = int(
            torch.randint(FROZEN.topic_count, (), generator=structure).item()
        )
        query = torch.zeros(FROZEN.evidence_width)
        query[query_topic] = 1.0

        candidate_vectors: list[Tensor] = []
        semantic_ids: list[int] = []
        for topic in range(FROZEN.topic_count):
            for style in (-1, 1):
                candidate_vectors.append(_candidate_vector(topic, style, values))
                semantic_ids.append(_semantic_id(topic, style))
        candidate_tensor = torch.stack(candidate_vectors)
        semantic_tensor = torch.tensor(semantic_ids, dtype=torch.long)

        history_vectors: list[Tensor] = []
        history_topics: list[int] = []
        history_styles: list[int] = []
        for topic in range(FROZEN.topic_count):
            for style in (-1, 1):
                history_vectors.append(_nonrepeat_history_vector(topic, style, values))
                history_topics.append(topic)
                history_styles.append(style)

        history_tensor = torch.stack(history_vectors)
        history_topic_tensor = torch.tensor(history_topics, dtype=torch.long)
        history_style_tensor = torch.tensor(history_styles, dtype=torch.long)
        history_permutation = torch.randperm(FROZEN.history_length, generator=structure)
        history_tensor = history_tensor[history_permutation]
        history_topic_tensor = history_topic_tensor[history_permutation]
        history_style_tensor = history_style_tensor[history_permutation]

        if int(kinds[request].item()) == 0:
            target_style = -1 if int(torch.randint(2, (), generator=structure)) == 0 else 1
            target_semantic = _semantic_id(query_topic, target_style)
            repeat_position = int(
                torch.randint(FROZEN.history_length, (), generator=structure).item()
            )
            history_tensor[repeat_position] = candidate_tensor[target_semantic]
            history_topic_tensor[repeat_position] = query_topic
            history_style_tensor[repeat_position] = target_style
        else:
            matching_positions = torch.nonzero(
                history_topic_tensor.eq(query_topic), as_tuple=False
            ).view(-1)
            latest_position = int(matching_positions.max().item())
            target_style = int(history_style_tensor[latest_position].item())
            target_semantic = _semantic_id(query_topic, target_style)

        candidate_permutation = torch.randperm(
            FROZEN.candidate_count, generator=structure
        )
        candidate_tensor = candidate_tensor[candidate_permutation]
        semantic_tensor = semantic_tensor[candidate_permutation]
        target_index = int(torch.nonzero(semantic_tensor.eq(target_semantic))[0, 0])

        queries.append(query)
        histories.append(history_tensor)
        candidates_all.append(candidate_tensor)
        candidate_ids_all.append(semantic_tensor)
        targets.append(target_index)
        query_topics.append(query_topic)
        history_topics_all.append(history_topic_tensor)
        history_styles_all.append(history_style_tensor)

    return SyntheticSplit(
        query=torch.stack(queries),
        history=torch.stack(histories),
        history_mask=torch.ones(request_count, FROZEN.history_length, dtype=torch.bool),
        candidates=torch.stack(candidates_all),
        candidate_ids=torch.stack(candidate_ids_all),
        target_index=torch.tensor(targets, dtype=torch.long),
        kind=kinds,
        query_topic=torch.tensor(query_topics, dtype=torch.long),
        history_topic=torch.stack(history_topics_all),
        history_style=torch.stack(history_styles_all),
        request_index=torch.arange(request_count, dtype=torch.long),
    )


def corrupt_supported(
    clean_supported: SyntheticSplit,
    base_seed: int,
    corruption: Literal["wrong_history", "shuffled_event", "query_mask", "disjoint"],
) -> SyntheticSplit:
    """Create one frozen evaluation-only corruption; labels/candidates stay fixed."""

    output = clean_supported.subset(torch.arange(clean_supported.query.shape[0]))
    request_count = output.query.shape[0]
    if corruption == "wrong_history":
        generator = make_generator(base_seed, "wrong_history")
        offset = int(torch.randint(1, request_count, (), generator=generator).item())
        donor = (torch.arange(request_count) + offset) % request_count
        output.history = clean_supported.history[donor].clone()
        output.history_mask = clean_supported.history_mask[donor].clone()
        output.history_topic = clean_supported.history_topic[donor].clone()
        output.history_style = clean_supported.history_style[donor].clone()
    elif corruption == "shuffled_event":
        generator = make_generator(base_seed, "shuffle_history")
        random_keys = torch.rand(
            request_count, FROZEN.history_length, generator=generator
        )
        permutation = torch.argsort(random_keys, dim=1, stable=True)
        gather_vector = permutation.unsqueeze(-1).expand(-1, -1, FROZEN.evidence_width)
        output.history = torch.gather(clean_supported.history, 1, gather_vector)
        output.history_mask = torch.gather(clean_supported.history_mask, 1, permutation)
        output.history_topic = torch.gather(clean_supported.history_topic, 1, permutation)
        output.history_style = torch.gather(clean_supported.history_style, 1, permutation)
    elif corruption == "query_mask":
        output.query = torch.zeros_like(clean_supported.query)
    elif corruption == "disjoint":
        half = FROZEN.evidence_width // 2
        output.history = torch.cat(
            (clean_supported.history[..., half:], clean_supported.history[..., :half]),
            dim=-1,
        )
    else:
        raise ValueError(corruption)
    return output


def make_batch_schedule(base_seed: int) -> Tensor:
    generator = make_generator(base_seed, "train_schedule")
    batches: list[Tensor] = []
    while len(batches) < FROZEN.optimizer_steps:
        permutation = torch.randperm(FROZEN.train_requests, generator=generator)
        for start in range(0, FROZEN.train_requests, FROZEN.batch_size):
            batches.append(permutation[start : start + FROZEN.batch_size])
            if len(batches) == FROZEN.optimizer_steps:
                break
    return torch.stack(batches)


class G1Ranker(nn.Module):
    """One parameterization with four read equations and identical state dicts."""

    def __init__(self, mechanism: Mechanism) -> None:
        super().__init__()
        if mechanism not in MECHANISMS:
            raise ValueError(mechanism)
        self.mechanism: Mechanism = mechanism
        width = FROZEN.evidence_width
        self.role_embedding = nn.Embedding(3, width)
        self.history_position = nn.Embedding(FROZEN.history_length, width)
        self.lower_block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=FROZEN.transformer_heads,
            dim_feedforward=FROZEN.transformer_ffn_width,
            dropout=FROZEN.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = ReversibleCouplingMemory(
            d_model=width,
            evidence_dim=width,
            read_mode="loop",
        )
        self.upper_block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=FROZEN.transformer_heads,
            dim_feedforward=FROZEN.transformer_ffn_width,
            dropout=FROZEN.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.score_head = nn.Linear(width, 1, bias=False)

    def _encode(self, latent: Tensor, role: int, *, history: bool = False) -> Tensor:
        hidden = latent + self.role_embedding.weight[role]
        if history:
            positions = torch.arange(FROZEN.history_length, device=latent.device)
            hidden = hidden + self.history_position(positions).view(1, -1, latent.shape[-1])
        flat = hidden.reshape(-1, 1, hidden.shape[-1])
        encoded = self.lower_block(flat).squeeze(1)
        return encoded.reshape_as(hidden)

    def _components(
        self,
        history: Tensor,
        query: Tensor,
        candidates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        context = candidates + self.memory.query_condition(query).unsqueeze(1)
        hf = _unit(self.memory.first_axis(history))
        hs = _unit(self.memory.second_axis(history))
        pf = _unit(self.memory.first_axis(context))
        ps = _unit(self.memory.second_axis(context))
        history_strength = self.memory.max_history_strength * torch.tanh(
            self.memory.history_strength(history).squeeze(-1)
        )
        probe_strength = self.memory.max_probe_strength * torch.tanh(
            self.memory.probe_strength(context).squeeze(-1)
        )
        return hf, hs, history_strength, pf, ps, probe_strength, context

    def _ordinary_hidden(
        self,
        history: Tensor,
        query: Tensor,
        candidates: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        hf, hs, h_strength, pf, ps, p_strength, _ = self._components(
            history, query, candidates
        )
        raw = self.memory.ordinary_residual_from_axes(
            hf, hs, h_strength, pf, ps, p_strength, history_mask
        )
        x, y = raw.split(FROZEN.evidence_width, dim=-1)
        x_scale = 1.0 + self.memory.probe_gains[0] * torch.tanh(
            p_strength + self.memory.probe_biases[0]
        )
        y_scale = 1.0 + self.memory.probe_gains[1] * torch.tanh(
            p_strength + self.memory.probe_biases[1]
        )
        raw = torch.cat((x * x_scale.unsqueeze(-1), y * y_scale.unsqueeze(-1)), dim=-1)
        return self.memory.hidden_from_raw(raw)

    def _attention_hidden(
        self,
        history: Tensor,
        query: Tensor,
        candidates: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        hf, hs, h_strength, pf, ps, p_strength, _ = self._components(
            history, query, candidates
        )
        key = (
            self.memory.history_gains[0] * hf
            + self.memory.history_gains[1] * hs
        )
        probe = self.memory.probe_gains[0] * pf + self.memory.probe_gains[1] * ps
        logits = torch.einsum("bce,bhe->bch", probe, key) / math.sqrt(
            FROZEN.evidence_width
        )
        logits = logits * (1.0 + p_strength.abs().unsqueeze(-1))
        logits = logits + h_strength.unsqueeze(1)
        mask = history_mask.unsqueeze(1)
        masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        maximum = masked_logits.max(dim=-1, keepdim=True).values
        weights = torch.exp(masked_logits - maximum) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

        seed_x, seed_y = self.memory.memory_seed.split(FROZEN.evidence_width)
        value_x = hf * (h_strength + self.memory.history_biases[0]).unsqueeze(-1)
        value_y = hs * (h_strength + self.memory.history_biases[1]).unsqueeze(-1)
        overlap_x = torch.einsum("bhe,bce->bch", hf, pf)
        overlap_y = torch.einsum("bhe,bce->bch", hs, ps)
        raw_x = torch.einsum("bch,bhe->bce", weights, value_x)
        raw_y = torch.einsum("bch,bhe->bce", weights, value_y)
        raw_x = raw_x + (weights * overlap_x).sum(-1, keepdim=True) * seed_x
        raw_y = raw_y + (weights * overlap_y).sum(-1, keepdim=True) * seed_y
        raw_x = raw_x * (
            1.0 + p_strength * torch.tanh(self.memory.probe_biases[0])
        ).unsqueeze(-1)
        raw_y = raw_y * (
            1.0 + p_strength * torch.tanh(self.memory.probe_biases[1])
        ).unsqueeze(-1)
        raw = torch.cat((raw_x, raw_y), dim=-1)
        present = history_mask.any(dim=1).view(-1, 1, 1)
        raw = torch.where(present, raw, torch.zeros_like(raw))
        return self.memory.hidden_from_raw(raw)

    def _pooled_ffn_hidden(
        self,
        history: Tensor,
        query: Tensor,
        candidates: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        mask = history_mask.unsqueeze(-1)
        pooled = (history * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        context = candidates + self.memory.query_condition(query).unsqueeze(1)
        h_first = self.memory.first_axis(pooled).unsqueeze(1)
        h_second = self.memory.second_axis(pooled).unsqueeze(1)
        c_first = self.memory.first_axis(context)
        c_second = self.memory.second_axis(context)
        h_strength = self.memory.max_history_strength * torch.tanh(
            self.memory.history_strength(pooled)
        ).unsqueeze(1)
        p_strength = self.memory.max_probe_strength * torch.tanh(
            self.memory.probe_strength(context)
        )
        seed_x, seed_y = self.memory.memory_seed.split(FROZEN.evidence_width)
        raw_x = torch.tanh(
            self.memory.history_gains[0] * h_first
            + self.memory.probe_gains[0] * c_second
            + seed_x
            + self.memory.history_biases[0]
            + p_strength * self.memory.probe_biases[0]
        )
        raw_y = torch.tanh(
            self.memory.history_gains[1] * h_second
            + self.memory.probe_gains[1] * c_first
            + seed_y
            + self.memory.history_biases[1]
            + p_strength * self.memory.probe_biases[1]
        )
        raw = torch.cat((raw_x * h_strength, raw_y * h_strength), dim=-1)
        present = history_mask.any(dim=1).view(-1, 1, 1)
        raw = torch.where(present, raw, torch.zeros_like(raw))
        return self.memory.hidden_from_raw(raw)

    def _memory_hidden(
        self,
        history: Tensor,
        query: Tensor,
        candidates: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        if self.mechanism == "rwpu":
            hidden, _ = self.memory(history, query, candidates, history_mask)
            return hidden
        if self.mechanism == "ordinary":
            return self._ordinary_hidden(history, query, candidates, history_mask)
        if self.mechanism == "attention":
            return self._attention_hidden(history, query, candidates, history_mask)
        if self.mechanism == "pooled_ffn":
            return self._pooled_ffn_hidden(history, query, candidates, history_mask)
        raise AssertionError(self.mechanism)

    def _score(self, query: Tensor, candidates: Tensor, memory_hidden: Tensor) -> Tensor:
        count = candidates.shape[1]
        query_tokens = query.unsqueeze(1).expand(-1, count, -1)
        pair = torch.stack((query_tokens, candidates + memory_hidden), dim=2)
        flat = pair.reshape(-1, 2, FROZEN.evidence_width)
        output = self.upper_block(flat)[:, 1]
        output = output.reshape(-1, count, FROZEN.evidence_width)
        return self.score_head(output).squeeze(-1)

    def query_only(self, query: Tensor, candidates: Tensor) -> Tensor:
        query_encoded = self._encode(query, role=0)
        candidate_encoded = self._encode(candidates, role=2)
        return self._score(
            query_encoded, candidate_encoded, torch.zeros_like(candidate_encoded)
        )

    def forward(
        self,
        query: Tensor,
        history: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        query_encoded = self._encode(query, role=0)
        history_encoded = self._encode(history, role=1, history=True)
        candidate_encoded = self._encode(candidates, role=2)
        history_encoded = history_encoded * history_mask.unsqueeze(-1)
        memory_hidden = self._memory_hidden(
            history_encoded, query_encoded, candidate_encoded, history_mask
        )
        return self._score(query_encoded, candidate_encoded, memory_hidden)


def initialized_models(base_seed: int) -> dict[Mechanism, G1Ranker]:
    torch.manual_seed(stream_seed(base_seed, "model_initialization"))
    reference = G1Ranker("rwpu")
    reference_state = deepcopy(reference.state_dict())
    models: dict[Mechanism, G1Ranker] = {}
    for mechanism in MECHANISMS:
        model = G1Ranker(mechanism)
        model.load_state_dict(reference_state, strict=True)
        models[mechanism] = model
    return models


def tensor_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def train_model(
    model: G1Ranker,
    train: SyntheticSplit,
    schedule: Tensor,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN.learning_rate,
        betas=(FROZEN.adam_beta1, FROZEN.adam_beta2),
        eps=FROZEN.adam_epsilon,
        weight_decay=FROZEN.weight_decay,
    )
    trace: list[float] = []
    gradient_finite = True
    model.train()
    for batch_index in schedule:
        batch = train.subset(batch_index)
        optimizer.zero_grad(set_to_none=True)
        scores = model(**batch.as_model_inputs())
        loss = F.cross_entropy(scores, batch.target_index)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                gradient_finite = False
        if not gradient_finite:
            raise RuntimeError("non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN.gradient_clip_norm)
        optimizer.step()
        trace.append(float(loss.detach()))
    return {
        "steps": len(trace),
        "loss_trace": trace,
        "loss_first": trace[0],
        "loss_final": trace[-1],
        "all_gradients_finite": gradient_finite,
    }


def _tie_priority(base_seed: int, request_index: int, candidate_id: int) -> int:
    text = (
        f"{base_seed}:eval:{request_index}:{candidate_id}:{FROZEN.tie_salt}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest(), byteorder="big", signed=False)


def top1_with_ties(
    scores: Tensor,
    candidate_ids: Tensor,
    request_index: Tensor,
    base_seed: int,
) -> Tensor:
    winners: list[int] = []
    for row in range(scores.shape[0]):
        priority = sorted(
            range(scores.shape[1]),
            key=lambda column: _tie_priority(
                base_seed,
                int(request_index[row]),
                int(candidate_ids[row, column]),
            ),
        )
        ordered = torch.tensor(priority, dtype=torch.long)
        rank = torch.argsort(scores[row, ordered], descending=True, stable=True)
        winners.append(int(ordered[rank[0]]))
    return torch.tensor(winners, dtype=torch.long)


def score_metrics(scores: Tensor, split: SyntheticSplit, base_seed: int) -> dict[str, float]:
    winners = top1_with_ties(
        scores, split.candidate_ids, split.request_index, base_seed
    )
    accuracy = winners.eq(split.target_index).float().mean().item()
    target = scores.gather(1, split.target_index.unsqueeze(1)).squeeze(1)
    negative = scores.clone()
    negative.scatter_(1, split.target_index.unsqueeze(1), float("-inf"))
    margin = (target - negative.max(dim=1).values).mean().item()
    return {"accuracy": accuracy, "mean_target_margin": margin}


def item_recurrence_accuracy(split: SyntheticSplit, base_seed: int) -> float:
    equal = split.candidates.unsqueeze(2).eq(split.history.unsqueeze(1)).all(dim=-1)
    scores = equal.any(dim=2).to(torch.float32)
    winners = top1_with_ties(
        scores, split.candidate_ids, split.request_index, base_seed
    )
    return winners.eq(split.target_index).float().mean().item()


def evaluate_model(
    model: G1Ranker,
    split: SyntheticSplit,
    base_seed: int,
) -> tuple[Tensor, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        scores = model(**split.as_model_inputs())
    if not torch.isfinite(scores).all():
        raise RuntimeError("non-finite evaluation score")
    return scores, score_metrics(scores, split, base_seed)


def config_dict() -> dict[str, object]:
    result = asdict(FROZEN)
    result["seeds"] = list(FROZEN.seeds)
    result["mechanisms"] = list(MECHANISMS)
    result["stream_offsets"] = dict(STREAM_OFFSETS)
    return result
