"""One-shot executable for C09's frozen G1 CPU synthetic falsifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "G1_EXECUTION_LOCK.json"
ARTIFACT_DIR = ROOT / "g1_artifacts"
START_MARKER = ARTIFACT_DIR / "STARTED.json"
RAW_RESULT = ARTIFACT_DIR / "g1_raw.json"

SEEDS = (20260711, 20260712, 20260713)
METHODS = (
    "base_raw",
    "base_matched",
    "q_only",
    "c_only",
    "view_mean",
    "poe",
    "global_scalar",
    "diagonal",
    "ordinary_attention",
    "constant_cma",
    "cma",
)
MATCHED_METHODS = tuple(method for method in METHODS if method != "base_raw")
Mode = Literal[
    "base_raw",
    "base_matched",
    "q_only",
    "c_only",
    "view_mean",
    "poe",
    "global_scalar",
    "diagonal",
    "ordinary_attention",
    "constant_cma",
    "cma",
]

D_MODEL = 8
N_HEAD = 2
FF_DIM = 32
HISTORY_LENGTH = 6
CANDIDATE_COUNT = 8
TRAIN_REQUESTS = 2048
VALID_REQUESTS = 512
BATCH_SIZE = 64
TRAIN_STEPS = 200


@dataclass(frozen=True)
class SyntheticSplit:
    query: Tensor
    history: Tensor
    candidates: Tensor
    utility: Tensor
    target: Tensor


def configure_cpu_determinism() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('G1 requires explicit CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It may already be frozen at one by an importing structural test.
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True, warn_only=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_execution_lock() -> str:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    entries = lock["locked_files"]
    for entry in entries:
        payload = (ROOT / entry["path"]).read_bytes()
        if len(payload) != entry["bytes"]:
            raise RuntimeError(f"locked byte count changed: {entry['path']}")
        if sha256_bytes(payload) != entry["sha256"]:
            raise RuntimeError(f"locked SHA-256 changed: {entry['path']}")
    manifest = "".join(
        f"{entry['sha256']}  {entry['path']}\n"
        for entry in sorted(entries, key=lambda item: item["path"].encode("utf-8"))
    ).encode("utf-8")
    combined = sha256_bytes(manifest)
    if combined != lock["combined_manifest"]["sha256"]:
        raise RuntimeError("execution-lock combined manifest mismatch")

    pre_lock_path = ROOT / "PRE_OUTCOME_LOCK.json"
    pre_lock = json.loads(pre_lock_path.read_text(encoding="utf-8"))
    pre_entries = pre_lock["normative_files"]
    for entry in pre_entries:
        payload = (ROOT / entry["path"]).read_bytes()
        if len(payload) != entry["bytes"]:
            raise RuntimeError(f"transitive pre-lock byte count changed: {entry['path']}")
        if sha256_bytes(payload) != entry["sha256"]:
            raise RuntimeError(f"transitive pre-lock SHA-256 changed: {entry['path']}")
    pre_manifest = "".join(
        f"{entry['sha256']}  {entry['path']}\n"
        for entry in sorted(
            pre_entries,
            key=lambda item: item["path"].encode("utf-8"),
        )
    ).encode("utf-8")
    pre_combined = sha256_bytes(pre_manifest)
    if pre_combined != pre_lock["combined_manifest"]["sha256"]:
        raise RuntimeError("transitive PRE_OUTCOME_LOCK manifest mismatch")
    if pre_combined != lock["transitive_original_manifest_sha256"]:
        raise RuntimeError("G1 lock names the wrong transitive original manifest")
    return combined


def generate_split(seed: int, request_count: int) -> SyntheticSplit:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    query = torch.randn(request_count, D_MODEL, generator=generator)
    history = torch.randn(
        request_count,
        HISTORY_LENGTH,
        D_MODEL,
        generator=generator,
    )
    candidates = torch.randn(
        request_count,
        CANDIDATE_COUNT,
        D_MODEL,
        generator=generator,
    )
    history_mean = history.mean(dim=1)
    alternating = torch.tensor(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    base = (
        (query[:, None, :] * alternating * candidates).sum(dim=-1)
        / math.sqrt(D_MODEL)
    )
    query_history = torch.tanh(
        (query * history_mean).sum(dim=-1) / math.sqrt(D_MODEL)
    )
    candidate_history = torch.tanh(
        (candidates * history_mean[:, None, :]).sum(dim=-1) / math.sqrt(D_MODEL)
    )
    utility = 0.35 * base + 4.0 * query_history[:, None] * candidate_history
    target = utility.argmax(dim=-1)
    return SyntheticSplit(query, history, candidates, utility, target)


def sattolo_permutation(length: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.arange(length)
    for index in range(length - 1, 0, -1):
        other = int(torch.randint(0, index, (1,), generator=generator).item())
        saved = int(permutation[index].item())
        permutation[index] = permutation[other]
        permutation[other] = saved
    if length > 1 and bool((permutation == torch.arange(length)).any()):
        raise RuntimeError("Sattolo implementation produced a fixed point")
    return permutation


def positive_meet(left: Tensor, right: Tensor) -> Tensor:
    left_pos = torch.relu(left)
    right_pos = torch.relu(right)
    denominator = left_pos + right_pos
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    result = left_pos * right_pos / safe
    return torch.where(denominator > 0, result, torch.zeros_like(result))


class G1SharedTransformer(nn.Module):
    def __init__(self, mode: Mode) -> None:
        super().__init__()
        if mode not in METHODS:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.input_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.history_position = nn.Parameter(torch.zeros(HISTORY_LENGTH, D_MODEL))
        self.history_encoder = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mediator_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEAD,
            dropout=0.0,
            batch_first=True,
        )
        self.rank_token = nn.Parameter(torch.zeros(D_MODEL))
        self.null_mediator = nn.Parameter(torch.zeros(D_MODEL))
        self.role_embedding = nn.Parameter(torch.zeros(4, D_MODEL))
        self.rank_encoder = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rank_norm = nn.LayerNorm(D_MODEL)
        self.score_head = nn.Linear(D_MODEL, 1, bias=False)
        self.fusion_value = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.fusion_output = nn.Linear(D_MODEL, 1, bias=False)
        self.rho = nn.Parameter(torch.zeros(()))
        self.tau_raw = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.rank_token, std=0.02)
        nn.init.normal_(self.null_mediator, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def encode(self, query: Tensor, history: Tensor, candidates: Tensor) -> Dict[str, Tensor]:
        query_state = self.input_projection(query)
        history_state = self.input_projection(history) + self.history_position[None]
        history_state = self.history_encoder(history_state)
        candidate_state = self.input_projection(candidates)

        q_mediator, _ = self.mediator_attention(
            query_state[:, None, :],
            history_state,
            history_state,
            need_weights=False,
        )
        batch, count, _ = candidate_state.shape
        expanded_history = history_state[:, None].expand(-1, count, -1, -1)
        expanded_history = expanded_history.reshape(
            batch * count,
            HISTORY_LENGTH,
            D_MODEL,
        )
        c_mediator, _ = self.mediator_attention(
            candidate_state.reshape(batch * count, 1, D_MODEL),
            expanded_history,
            expanded_history,
            need_weights=False,
        )
        c_mediator = c_mediator.reshape(batch, count, D_MODEL)
        return {
            "query": query_state,
            "candidates": candidate_state,
            "q_mediator": q_mediator.squeeze(1),
            "c_mediator": c_mediator,
        }

    def rank(
        self,
        query: Tensor,
        candidates: Tensor,
        mediator: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, count, _ = candidates.shape
        rank = self.rank_token[None, None].expand(batch, count, -1)
        q = query[:, None].expand(-1, count, -1)
        sequence = torch.stack((rank, q, candidates, mediator), dim=2)
        sequence = sequence + self.role_embedding[None, None]
        encoded = self.rank_encoder(sequence.reshape(batch * count, 4, D_MODEL))
        hidden = self.rank_norm(encoded[:, 0]).reshape(batch, count, D_MODEL)
        return self.score_head(hidden).squeeze(-1), hidden

    @staticmethod
    def _zero_diagonal(strength: Tensor) -> Tensor:
        count = strength.shape[-1]
        diagonal = torch.eye(count, dtype=torch.bool, device=strength.device)[None]
        return strength.masked_fill(diagonal, 0.0)

    def _aggregate(self, strength: Tensor, values: Tensor) -> Tensor:
        strength = self._zero_diagonal(strength)
        attention = strength / (1.0 + strength.sum(dim=-1, keepdim=True))
        return F.softplus(self.rho) * (attention * values).sum(dim=-1)

    def _pair_values(self, base_hidden: Tensor) -> Tensor:
        contrast = base_hidden.unsqueeze(2) - base_hidden.unsqueeze(1)
        return self.fusion_output(self.fusion_value(contrast)).squeeze(-1)

    def fuse(
        self,
        base: Tensor,
        q_scores: Tensor,
        c_scores: Tensor,
        base_hidden: Tensor,
        evidence_available: Tensor,
        corruption: str = "clean",
    ) -> tuple[Tensor, Tensor]:
        rq = q_scores - base
        rc = c_scores - base
        if corruption == "query_factor_flip":
            rq = -rq
        elif corruption == "candidate_factor_flip":
            rc = -rc
        elif corruption == "all_pair_disagreement":
            rc = -rq
        elif corruption != "clean":
            raise ValueError(f"unknown fusion corruption: {corruption}")

        mq = rq.unsqueeze(-1) - rq.unsqueeze(-2)
        mc = rc.unsqueeze(-1) - rc.unsqueeze(-2)
        temperature = F.softplus(self.tau_raw) + 1e-4
        scale = F.softplus(self.rho)
        pair_values = self._pair_values(base_hidden)

        if self.mode == "cma":
            strength = positive_meet(temperature * mq, temperature * mc)
            correction = self._aggregate(strength, pair_values)
        elif self.mode == "q_only":
            correction = self._aggregate(torch.relu(temperature * mq), pair_values)
        elif self.mode == "c_only":
            correction = self._aggregate(torch.relu(temperature * mc), pair_values)
        elif self.mode == "ordinary_attention":
            energy = torch.clamp(temperature * (mq + mc) / 2.0, -12.0, 12.0)
            correction = self._aggregate(torch.exp(energy), pair_values)
        elif self.mode == "constant_cma":
            strength = positive_meet(temperature * mq, temperature * mc)
            constant = self.fusion_output(
                self.fusion_value(self.rank_token)
            ).reshape(1, 1, 1)
            constant_values = constant.expand_as(strength)
            correction = self._aggregate(strength, constant_values)
        elif self.mode == "base_raw":
            correction = torch.zeros_like(base)
        else:
            base_margin = base.unsqueeze(-1) - base.unsqueeze(-2)
            carrier_strength = torch.exp(
                torch.clamp(temperature * base_margin, -12.0, 12.0)
            )
            carrier = self._aggregate(carrier_strength, pair_values)
            rbar = (rq + rc) / 2.0
            if self.mode == "base_matched":
                correction = carrier
            elif self.mode == "view_mean":
                alpha = torch.sigmoid(self.tau_raw)
                correction = carrier + scale * (alpha * rq + (1.0 - alpha) * rc)
            elif self.mode == "poe":
                correction = carrier + scale * temperature * (rq + rc)
            elif self.mode == "global_scalar":
                evidence = torch.sqrt(rbar.square().mean(dim=-1, keepdim=True) + 1e-12)
                gate = torch.sigmoid(temperature * evidence)
                correction = carrier + scale * gate * rbar
            elif self.mode == "diagonal":
                gate = torch.sigmoid(temperature * rbar.abs())
                correction = carrier + scale * gate * rbar
            else:
                raise AssertionError(self.mode)

        available = evidence_available.to(dtype=torch.bool).unsqueeze(-1)
        final = torch.where(available, base + correction, base)
        effective_correction = torch.where(
            available,
            correction,
            torch.zeros_like(correction),
        )
        return final, effective_correction

    def forward(
        self,
        query: Tensor,
        history: Tensor,
        candidates: Tensor,
        history_present: Tensor | None = None,
        query_present: Tensor | None = None,
        corruption: str = "clean",
    ) -> Dict[str, Tensor]:
        states = self.encode(query, history, candidates)
        batch, count, _ = states["candidates"].shape
        null = self.null_mediator[None, None].expand(batch, count, -1)
        q_mediator = states["q_mediator"][:, None].expand(-1, count, -1)
        base, base_hidden = self.rank(states["query"], states["candidates"], null)
        q_scores, _ = self.rank(
            states["query"],
            states["candidates"],
            q_mediator,
        )
        c_scores, _ = self.rank(
            states["query"],
            states["candidates"],
            states["c_mediator"],
        )
        if history_present is None:
            history_present = torch.ones(batch, dtype=torch.bool)
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool)
        available = history_present & query_present
        final, correction = self.fuse(
            base,
            q_scores,
            c_scores,
            base_hidden,
            available,
            corruption=corruption,
        )
        return {
            "scores": final,
            "base": base,
            "q_scores": q_scores,
            "c_scores": c_scores,
            "correction": correction,
        }


def make_batch_schedule(seed: int) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 200000)
    schedule: list[Tensor] = []
    while len(schedule) < TRAIN_STEPS:
        permutation = torch.randperm(TRAIN_REQUESTS, generator=generator)
        for start in range(0, TRAIN_REQUESTS, BATCH_SIZE):
            schedule.append(permutation[start : start + BATCH_SIZE])
            if len(schedule) == TRAIN_STEPS:
                break
    return schedule


def gradient_groups(model: G1SharedTransformer) -> Dict[str, float]:
    groups: Mapping[str, Iterable[nn.Parameter]] = {
        "shared_input": model.input_projection.parameters(),
        "history_transformer": tuple(model.history_encoder.parameters())
        + (model.history_position,),
        "mediator_mha": model.mediator_attention.parameters(),
        "rank_transformer_head": tuple(model.rank_encoder.parameters())
        + tuple(model.rank_norm.parameters())
        + tuple(model.score_head.parameters())
        + (model.rank_token, model.null_mediator, model.role_embedding),
        "fusion_value_output": tuple(model.fusion_value.parameters())
        + tuple(model.fusion_output.parameters()),
        "fusion_scale": (model.rho,),
        "fusion_temperature": (model.tau_raw,),
    }
    result: Dict[str, float] = {}
    for name, parameters in groups.items():
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().double().square().sum())
        result[name] = math.sqrt(squared)
    return result


def training_loss(output: Mapping[str, Tensor], target: Tensor) -> Tensor:
    return (
        F.cross_entropy(output["scores"], target)
        + F.cross_entropy(output["base"], target)
        + 0.5 * F.cross_entropy(output["q_scores"], target)
        + 0.5 * F.cross_entropy(output["c_scores"], target)
    )


def train_method(
    mode: Mode,
    seed: int,
    train: SyntheticSplit,
    schedule: Sequence[Tensor],
) -> tuple[G1SharedTransformer, Dict[str, object]]:
    torch.manual_seed(seed + 1000003)
    model = G1SharedTransformer(mode)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    first_gradients: Dict[str, float] | None = None
    losses: list[float] = []
    for step, indices in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train.query[indices],
            train.history[indices],
            train.candidates[indices],
        )
        loss = training_loss(output, train.target[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss for {mode} at step {step}")
        loss.backward()
        if step == 0:
            first_gradients = gradient_groups(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert first_gradients is not None
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return model, {
        "total_trainable_parameters": total_parameters,
        "first_backward_group_grad_norms": first_gradients,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "all_losses_finite": all(math.isfinite(value) for value in losses),
    }


def pairwise_accuracy(scores: Tensor, utility: Tensor) -> float:
    count = scores.shape[-1]
    left, right = torch.triu_indices(count, count, offset=1)
    score_difference = scores[:, left] - scores[:, right]
    utility_difference = utility[:, left] - utility[:, right]
    retained = utility_difference.abs() > 1e-8
    predicted_tie = score_difference.abs() <= 1e-8
    correct = torch.sign(score_difference) == torch.sign(utility_difference)
    credit = torch.where(
        predicted_tie,
        torch.full_like(score_difference, 0.5),
        correct.to(score_difference.dtype),
    )
    return float(credit[retained].double().mean())


def evaluate_scores(scores: Tensor, split: SyntheticSplit) -> Dict[str, object]:
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("non-finite validation score")
    return {
        "pairwise_accuracy": pairwise_accuracy(scores, split.utility),
        "top1_accuracy": float(
            (scores.argmax(dim=-1) == split.target).double().mean()
        ),
        "scores_finite": True,
    }


@torch.no_grad()
def evaluate_method(
    model: G1SharedTransformer,
    split: SyntheticSplit,
    seed: int,
) -> Dict[str, object]:
    model.eval()
    clean_output = model(split.query, split.history, split.candidates)
    result: Dict[str, object] = {
        "clean": evaluate_scores(clean_output["scores"], split),
    }
    if model.mode == "cma":
        corruptions: Dict[str, object] = {}
        for corruption in (
            "query_factor_flip",
            "candidate_factor_flip",
            "all_pair_disagreement",
        ):
            output = model(
                split.query,
                split.history,
                split.candidates,
                corruption=corruption,
            )
            corruptions[corruption] = {
                **evaluate_scores(output["scores"], split),
                "max_abs_correction": float(output["correction"].abs().max()),
                "bit_exact_base": bool(torch.equal(output["scores"], output["base"])),
            }
        permutation = sattolo_permutation(len(split.query), seed + 300000)
        shuffled = model(
            split.query,
            split.history[permutation],
            split.candidates,
        )
        corruptions["shuffled_history"] = {
            **evaluate_scores(shuffled["scores"], split),
            "fixed_points": int((permutation == torch.arange(len(permutation))).sum()),
        }
        no_history = model(
            split.query,
            split.history,
            split.candidates,
            history_present=torch.zeros(len(split.query), dtype=torch.bool),
        )
        query_masked = model(
            torch.zeros_like(split.query),
            split.history,
            split.candidates,
            query_present=torch.zeros(len(split.query), dtype=torch.bool),
        )
        corruptions["no_history"] = {
            "score_mismatch_count": int((no_history["scores"] != no_history["base"]).sum()),
            "bit_exact_base": bool(torch.equal(no_history["scores"], no_history["base"])),
        }
        corruptions["query_masked"] = {
            "score_mismatch_count": int(
                (query_masked["scores"] != query_masked["base"]).sum()
            ),
            "bit_exact_base": bool(
                torch.equal(query_masked["scores"], query_masked["base"])
            ),
        }
        result["corruptions"] = corruptions
    return result


def decide(seed_results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    per_seed: list[Dict[str, object]] = []
    for seed_result in seed_results:
        methods = seed_result["methods"]
        accuracy = {
            name: float(methods[name]["evaluation"]["clean"]["pairwise_accuracy"])
            for name in METHODS
        }
        simple_best = max(
            accuracy[name]
            for name in (
                "base_raw",
                "base_matched",
                "q_only",
                "c_only",
                "global_scalar",
                "diagonal",
            )
        )
        clean_surplus = accuracy["cma"] - accuracy["base_raw"]
        cma_corruptions = methods["cma"]["evaluation"]["corruptions"]
        retentions: Dict[str, float | None] = {}
        for name in (
            "query_factor_flip",
            "candidate_factor_flip",
            "shuffled_history",
        ):
            corrupted_accuracy = float(cma_corruptions[name]["pairwise_accuracy"])
            retentions[name] = (
                (corrupted_accuracy - accuracy["base_raw"]) / clean_surplus
                if clean_surplus > 0
                else None
            )
        per_seed.append(
            {
                "seed": seed_result["seed"],
                "accuracies": accuracy,
                "simple_best": simple_best,
                "cma_simple_surplus": accuracy["cma"] - simple_best,
                "cma_attention_surplus": accuracy["cma"]
                - accuracy["ordinary_attention"],
                "cma_constant_surplus": accuracy["cma"] - accuracy["constant_cma"],
                "clean_surplus_over_base_raw": clean_surplus,
                "corruption_retentions": retentions,
                "all_disagreement_exact": bool(
                    cma_corruptions["all_pair_disagreement"]["bit_exact_base"]
                ),
                "no_history_mismatch_count": cma_corruptions["no_history"][
                    "score_mismatch_count"
                ],
                "query_masked_mismatch_count": cma_corruptions["query_masked"][
                    "score_mismatch_count"
                ],
            }
        )

    simple_passes = sum(row["cma_simple_surplus"] >= 0.05 for row in per_seed)
    mean_simple_surplus = sum(row["cma_simple_surplus"] for row in per_seed) / 3.0
    attention_passes = sum(row["cma_attention_surplus"] >= 0.02 for row in per_seed)
    criterion_1 = simple_passes >= 2 and mean_simple_surplus >= 0.05
    criterion_2 = attention_passes >= 2
    criterion_3 = all(row["all_disagreement_exact"] for row in per_seed)
    criterion_4 = all(
        retention is not None and retention <= 0.25
        for row in per_seed
        for retention in row["corruption_retentions"].values()
    )
    criterion_5 = all(row["cma_constant_surplus"] >= 0.01 for row in per_seed)
    criterion_6 = all(
        row["no_history_mismatch_count"] == 0
        and row["query_masked_mismatch_count"] == 0
        for row in per_seed
    )

    integrity_rows: list[bool] = []
    for seed_result in seed_results:
        matched_counts = {
            seed_result["methods"][name]["training"]["total_trainable_parameters"]
            for name in MATCHED_METHODS
        }
        count_ok = len(matched_counts) == 1
        gradient_ok = True
        finite_ok = True
        for name in MATCHED_METHODS:
            training = seed_result["methods"][name]["training"]
            finite_ok &= bool(training["all_losses_finite"])
            for value in training["first_backward_group_grad_norms"].values():
                gradient_ok &= math.isfinite(float(value)) and float(value) > 0.0
        integrity_rows.append(count_ok and gradient_ok and finite_ok)
    criterion_7 = all(integrity_rows)

    criteria = {
        "c1_cma_vs_simple": criterion_1,
        "c2_cma_vs_ordinary_attention": criterion_2,
        "c3_all_disagreement_exact": criterion_3,
        "c4_corruption_retention": criterion_4,
        "c5_constant_value_gap": criterion_5,
        "c6_exact_fallbacks": criterion_6,
        "c7_integrity_and_active_matching": criterion_7,
    }
    passed = all(criteria.values())
    return {
        "per_seed": per_seed,
        "criteria": criteria,
        "simple_surplus_pass_seed_count": simple_passes,
        "mean_cma_simple_surplus": mean_simple_surplus,
        "ordinary_attention_pass_seed_count": attention_passes,
        "terminal_decision": (
            "PASS_G1_REQUEST_D0_REVIEW" if passed else "STOP_C09_G1_FAILED"
        ),
    }


def run() -> Dict[str, object]:
    configure_cpu_determinism()
    lock_hash = verify_execution_lock()
    if START_MARKER.exists() or RAW_RESULT.exists():
        raise RuntimeError("G1 one-shot marker/result already exists; rerun forbidden")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    START_MARKER.write_text(
        json.dumps(
            {
                "state": "started_once",
                "execution_lock_manifest_sha256": lock_hash,
                "seeds": list(SEEDS),
                "device": "cpu",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    seed_results: list[Dict[str, object]] = []
    for seed in SEEDS:
        train = generate_split(seed, TRAIN_REQUESTS)
        valid = generate_split(seed + 100000, VALID_REQUESTS)
        schedule = make_batch_schedule(seed)
        method_results: Dict[str, object] = {}
        for mode in METHODS:
            model, training = train_method(mode, seed, train, schedule)
            evaluation = evaluate_method(model, valid, seed)
            method_results[mode] = {
                "training": training,
                "evaluation": evaluation,
            }
        seed_results.append(
            {
                "seed": seed,
                "generator_audit": {
                    "train_requests": len(train.query),
                    "validation_requests": len(valid.query),
                    "train_utility_finite": bool(torch.isfinite(train.utility).all()),
                    "validation_utility_finite": bool(torch.isfinite(valid.utility).all()),
                },
                "methods": method_results,
            }
        )

    decision = decide(seed_results)
    result: Dict[str, object] = {
        "schema": "c09.g1.raw.v1",
        "execution_lock_manifest_sha256": lock_hash,
        "command_contract": 'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python systems/09_cross_view_agreement_transformer/g1_runner.py',
        "repository_data_or_labels_used": False,
        "gpu_used": False,
        "dev_or_test_used": False,
        "seeds": list(SEEDS),
        "seed_results": seed_results,
        "decision": decision,
    }
    temporary = RAW_RESULT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RAW_RESULT)
    return result


def main() -> None:
    result = run()
    summary = {
        "terminal_decision": result["decision"]["terminal_decision"],
        "criteria": result["decision"]["criteria"],
        "mean_cma_simple_surplus": result["decision"]["mean_cma_simple_surplus"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
