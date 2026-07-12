"""Locked CPU-only C07 G1 semantic synthetic probe.

There is deliberately no repository-data input.  The execution lock is
verified before any semantic RNG or model is constructed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


C = 5
H = 8
D = 16
TAU = 0.5
KAPPA = 1.0
TRAIN_SIZE = 4096
HELDOUT_SIZE = 4096
UPDATES = 512
BATCH_PER_WORLD = 16
EVAL_BATCH = 256
SEEDS = (20260711, 20260712, 20260713)
METHODS = (
    "PDSK",
    "CENTER0",
    "GATED_CENTER",
    "TARGET_NULL",
    "DIFF_ATTN",
    "BASE_FFN",
    "ITEM_ONLY",
)
CORRUPTIONS = ("wrong_history", "shuffled_event", "query_masked")
AUTHORIZED_OUTPUT = Path("artifacts/runs/20260711_c07_pdsk_g1_cpu/result.json")
LOCKED_MANIFEST = "66308db14f00e20de860a2060d147329fb93fa07b806951c227a5499746c2edd"


@dataclass(frozen=True)
class KernelOutput:
    weights: Tensor
    balances: Tensor


class PairwiseSignedKernel(nn.Module):
    """Self-contained executable copy of the locked PDSK equation."""

    def __init__(self, threshold: float, null_mass: float) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("null_mass", torch.tensor(float(null_mass)))

    def forward(self, scores: Tensor, history_mask: Tensor) -> KernelOutput:
        batch, candidates, _ = scores.shape
        margins = scores[:, :, None, :] - scores[:, None, :, :]
        eye = torch.eye(candidates, dtype=torch.bool, device=scores.device)
        valid = ~eye[None, :, :, None] & history_mask[:, None, None, :]
        threshold = self.threshold.to(scores.dtype)
        shrunk = F.relu(margins - threshold) - F.relu(-margins - threshold)
        shrunk = torch.where(valid, shrunk, torch.zeros_like(shrunk))
        balances = shrunk.sum(dim=2) / max(candidates - 1, 1)
        balances = balances * history_mask[:, None, :].to(scores.dtype)
        weights = balances / (
            self.null_mass.to(scores.dtype)
            + balances.abs().sum(dim=(1, 2), keepdim=True)
        )
        return KernelOutput(weights, balances)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_old_normative_lock(root: Path) -> dict:
    pre_lock_path = root / "PRE_OUTCOME_LOCK.json"
    pre_lock = json.loads(pre_lock_path.read_text(encoding="utf-8"))
    if pre_lock.get("combined_manifest_sha256") != LOCKED_MANIFEST:
        raise RuntimeError("old lock declares the wrong combined digest")
    for key in ("semantic_probe_run", "repository_data_read", "labels_or_qrels_read", "gpu_used"):
        if pre_lock.get(key) is not False:
            raise RuntimeError(f"old lock declaration {key!r} is not false")
    repo_root = root.parents[1]
    lines: list[str] = []
    for rel, expected in pre_lock["normative_files"].items():
        path = root / rel
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"old normative file hash mismatch: {rel}")
        project_relative = path.relative_to(repo_root)
        lines.append(f"{actual}  {project_relative}\n")
    combined = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if combined != LOCKED_MANIFEST:
        raise RuntimeError("old normative combined manifest mismatch")
    return pre_lock


def verify_execution_lock(lock_path: Path, output_path: Path) -> dict:
    """Verify the second-level lock without constructing semantic state."""

    lock_path = lock_path.resolve()
    root = lock_path.parent
    verify_old_normative_lock(root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen_before_g1_semantic_outcome":
        raise RuntimeError("execution lock status is not pre-outcome")
    declarations = lock.get("declarations", {})
    required_false = (
        "semantic_g1_outcome_observed",
        "repository_data_read",
        "labels_or_qrels_read",
        "gpu_used",
    )
    for key in required_false:
        if declarations.get(key) is not False:
            raise RuntimeError(f"execution lock declaration {key!r} is not false")
    if lock.get("old_lock_combined_sha256") != LOCKED_MANIFEST:
        raise RuntimeError("old C07 normative lock digest mismatch")

    entries = lock.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("execution lock file list missing")
    canonical_lines: list[str] = []
    for entry in entries:
        rel = entry["path"]
        expected = entry["sha256"]
        expected_size = entry["bytes"]
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"locked file missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"locked file hash mismatch: {rel}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"locked file size mismatch: {rel}")
        canonical_lines.append(f"{rel}\t{expected_size}\t{expected}\n")
    combined = hashlib.sha256("".join(canonical_lines).encode("utf-8")).hexdigest()
    if combined != lock.get("combined_manifest_sha256"):
        raise RuntimeError("execution combined manifest mismatch")

    repo_root = root.parents[1]
    expected_output = (repo_root / AUTHORIZED_OUTPUT).resolve()
    if output_path.resolve() != expected_output:
        raise RuntimeError(f"output must be exactly {expected_output}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    return lock


def cpu_generator(seed: int, stream_id: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1000 + stream_id)
    return generator


@dataclass(frozen=True)
class World:
    query: Tensor
    candidates: Tensor
    history_key: Tensor
    history_value: Tensor
    history_mask: Tensor
    exact_match: Tensor
    target: Tensor
    foil: Tensor
    subthreshold_mask: Tensor

    def __len__(self) -> int:
        return int(self.query.shape[0])

    def take(self, index: Tensor) -> "World":
        return World(
            query=self.query[index],
            candidates=self.candidates[index],
            history_key=self.history_key[index],
            history_value=self.history_value[index],
            history_mask=self.history_mask[index],
            exact_match=self.exact_match[index],
            target=self.target[index],
            foil=self.foil[index],
            subthreshold_mask=self.subthreshold_mask[index],
        )


def concatenate_worlds(parts: Sequence[World]) -> World:
    return World(*(
        torch.cat([getattr(part, field) for part in parts], dim=0)
        for field in World.__dataclass_fields__
    ))


def _base_draws(n: int, generator: torch.Generator) -> tuple[Tensor, ...]:
    target = torch.randint(0, C, (n,), generator=generator)
    foil_raw = torch.randint(0, C - 1, (n,), generator=generator)
    foil = foil_raw + (foil_raw >= target).long()
    candidate_noise = torch.randn((n, C, D - 6), generator=generator) * 0.05
    query_noise = torch.randn((n, D - 6), generator=generator) * 0.05
    history_key_noise = torch.randn((n, H, D - 6), generator=generator) * 0.05
    history_value_noise = torch.randn((n, H, D - 6), generator=generator) * 0.05
    common_shift = torch.rand((n, H), generator=generator) * 2.0 - 1.0

    candidates = torch.zeros((n, C, D), dtype=torch.float32)
    candidates[:, :, :C] = torch.eye(C, dtype=torch.float32)[None, :, :]
    candidates[:, :, 5] = 1.0
    candidates[:, :, 6:] = candidate_noise
    query = torch.zeros((n, D), dtype=torch.float32)
    query[:, 5] = 1.0
    query[:, 6:] = query_noise
    history_key = torch.zeros((n, H, D), dtype=torch.float32)
    history_key[:, :, 5] = common_shift
    history_key[:, :, 6:] = history_key_noise
    history_value = torch.zeros((n, H, D), dtype=torch.float32)
    history_value[:, :, 6:] = history_value_noise
    history_mask = torch.ones((n, H), dtype=torch.bool)
    exact_match = torch.zeros((n, C, H), dtype=torch.float32)
    subthreshold = torch.zeros((n, H), dtype=torch.bool)
    return (
        target,
        foil,
        candidates,
        query,
        history_key,
        history_value,
        history_mask,
        exact_match,
        subthreshold,
    )


def oracle_ranges(world: World) -> Tensor:
    scores = torch.einsum(
        "bd,bcd,bhd->bch",
        world.query[:, :6],
        world.candidates[:, :, :6],
        world.history_key[:, :, :6],
    )
    scores = scores + world.exact_match
    return scores.amax(dim=1) - scores.amin(dim=1)


def _assert_world_contract(world: World, name: str, required_active: Tensor | None = None) -> None:
    ranges = oracle_ranges(world)
    if required_active is not None and not torch.all(ranges[required_active] > TAU):
        raise RuntimeError(f"{name}: required oracle-active event failed")
    if world.subthreshold_mask.any() and not torch.all(
        ranges[world.subthreshold_mask] <= TAU
    ):
        raise RuntimeError(f"{name}: sub-threshold slice became active")


def make_world(n: int, generator: torch.Generator, kind: str) -> World:
    (
        target,
        foil,
        candidates,
        query,
        history_key,
        history_value,
        history_mask,
        exact_match,
        subthreshold,
    ) = _base_draws(n, generator)
    row = torch.arange(n)

    if kind == "R":
        query[row, target] = 0.55
        query[row, foil] = 0.55
        recurrence = torch.randint(0, H, (n,), generator=generator)
        semantic_key = torch.rand((n, H, C), generator=generator) * 0.60 - 0.30
        semantic_value = torch.randn((n, H, C), generator=generator) * 0.10
        history_key[:, :, :C] = semantic_key
        history_value[:, :, :C] = semantic_value
        history_key[row, recurrence, target] = 1.25
        history_value[row, recurrence, target] = 1.0
        exact_match[row, target, recurrence] = 1.0
        subthreshold[:] = True
        subthreshold[row, recurrence] = False
        world = World(
            query,
            candidates,
            history_key,
            history_value,
            history_mask,
            exact_match,
            target,
            foil,
            subthreshold,
        )
        active_mask = torch.zeros((n, H), dtype=torch.bool)
        active_mask[row, recurrence] = True
        _assert_world_contract(world, kind, active_mask)
        return world

    if kind in {"S", "U_BASE"}:
        query[row, target] = 0.80
        query[row, foil] = 0.80
        event_order = torch.rand((n, H), generator=generator).argsort(dim=1)
        supports = event_order[:, :2]
        contradiction = event_order[:, 2]
        sub_positions = event_order[:, 3:]
        semantic_key = torch.rand((n, H, C), generator=generator) * 0.50 - 0.25
        semantic_value = torch.randn((n, H, C), generator=generator) * 0.10
        history_key[:, :, :C] = semantic_key
        history_value[:, :, :C] = semantic_value
        for column in range(2):
            position = supports[:, column]
            history_key[row, position, target] = 1.25
            history_value[row, position, target] = 1.0
        history_key[row, contradiction, foil] = 0.85
        history_value[row, contradiction, foil] = 0.80
        subthreshold[row[:, None], sub_positions] = True
        world = World(
            query,
            candidates,
            history_key,
            history_value,
            history_mask,
            exact_match,
            target,
            foil,
            subthreshold,
        )
        active_mask = torch.zeros((n, H), dtype=torch.bool)
        active_mask[row[:, None], supports] = True
        active_mask[row, contradiction] = True
        _assert_world_contract(world, kind, active_mask)
        return world

    if kind == "N":
        query[row, target] = 1.0
        query[row, foil] = 0.20
        history_mask.zero_()
        b = torch.arange(n)[:, None, None]
        j = torch.arange(H)[None, :, None]
        r = torch.arange(D)[None, None, :]
        history_key = torch.where((b + j + r) % 2 == 0, 1000.0, -1000.0).float()
        history_value = -history_key
        exact_match.fill_(1000.0)
        world = World(
            query,
            candidates,
            history_key,
            history_value,
            history_mask,
            exact_match,
            target,
            foil,
            subthreshold,
        )
        if not all(torch.isfinite(tensor).all() for tensor in (history_key, history_value, exact_match)):
            raise RuntimeError("N: non-finite canary")
        return world

    raise ValueError(f"unknown world kind {kind}")


def make_u_corruptions(base: World, generator: torch.Generator) -> dict[str, World]:
    n = len(base)
    starts = torch.randint(1, n, (n,), generator=generator)
    donors = torch.empty(n, dtype=torch.long)
    for index in range(n):
        for step in range(n - 1):
            donor = int((index + int(starts[index]) + step) % n)
            donor_target = int(base.target[donor])
            if donor != index and donor_target not in {
                int(base.target[index]),
                int(base.foil[index]),
            }:
                donors[index] = donor
                break
        else:
            raise RuntimeError("failed to find wrong-history donor")
    wrong = World(
        query=base.query.clone(),
        candidates=base.candidates.clone(),
        history_key=base.history_key[donors].clone(),
        history_value=base.history_value[donors].clone(),
        history_mask=base.history_mask[donors].clone(),
        exact_match=base.exact_match[donors].clone(),
        target=base.target.clone(),
        foil=base.foil.clone(),
        subthreshold_mask=base.subthreshold_mask[donors].clone(),
    )

    value_permutation = torch.tensor([(j + 3) % H for j in range(H)])
    shuffled = World(
        query=base.query.clone(),
        candidates=base.candidates.clone(),
        history_key=base.history_key.clone(),
        history_value=base.history_value[:, value_permutation].clone(),
        history_mask=base.history_mask.clone(),
        exact_match=base.exact_match.clone(),
        target=base.target.clone(),
        foil=base.foil.clone(),
        subthreshold_mask=base.subthreshold_mask.clone(),
    )

    masked_query = base.query.clone()
    for index in range(n):
        original_set = {int(base.target[index]), int(base.foil[index])}
        shift = next(
            candidate_shift
            for candidate_shift in range(1, C)
            if {
                (int(base.target[index]) + candidate_shift) % C,
                (int(base.foil[index]) + candidate_shift) % C,
            }.isdisjoint(original_set)
        )
        original = base.query[index, :C].clone()
        destinations = (torch.arange(C) + shift) % C
        masked_query[index, destinations] = original
    query_masked = World(
        query=masked_query,
        candidates=base.candidates.clone(),
        history_key=base.history_key.clone(),
        history_value=base.history_value.clone(),
        history_mask=base.history_mask.clone(),
        exact_match=base.exact_match.clone(),
        target=base.target.clone(),
        foil=base.foil.clone(),
        subthreshold_mask=base.subthreshold_mask.clone(),
    )
    worlds = {
        "wrong_history": wrong,
        "shuffled_event": shuffled,
        "query_masked": query_masked,
    }
    for name, world in worlds.items():
        _assert_world_contract(world, name)
        if not torch.allclose(world.query.norm(dim=1), base.query.norm(dim=1), atol=1e-6, rtol=0):
            if name == "query_masked":
                raise RuntimeError("query-masked norm was not preserved")
    return worlds


def world_sha256(world: World) -> str:
    digest = hashlib.sha256()
    for field in World.__dataclass_fields__:
        tensor = getattr(world, field).detach().cpu().contiguous()
        digest.update(field.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RankerOutput:
    logits: Tensor
    base_logits: Tensor
    evidence_logits: Tensor
    signed_weights: Tensor


class SyntheticRanker(nn.Module):
    def __init__(self, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.type_embedding = nn.Parameter(torch.empty(3, D))
        self.query_projection = nn.Linear(D, D, bias=False)
        self.candidate_projection = nn.Linear(D, D, bias=False)
        self.history_projection = nn.Linear(D, D, bias=False)
        self.value_projection = nn.Linear(D, D, bias=False)
        self.output_projection = nn.Linear(D, D, bias=False)
        self.final_norm = nn.LayerNorm(D)
        self.score_head = nn.Linear(D, 1, bias=False)
        self.theta = nn.Parameter(torch.empty(D))
        self.pdsk = PairwiseSignedKernel(TAU, KAPPA)
        self.center0 = PairwiseSignedKernel(0.0, KAPPA)
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

    @staticmethod
    def flow_mask(device: torch.device) -> Tensor:
        length = 1 + H + C
        mask = torch.zeros((length, length), dtype=torch.bool, device=device)
        history = slice(1, 1 + H)
        candidates = slice(1 + H, length)
        mask[0, 1:] = True
        mask[history, candidates] = True
        mask[candidates, history] = True
        return mask

    @staticmethod
    def _global_l1(balance: Tensor) -> Tensor:
        return balance / (KAPPA + balance.abs().sum(dim=(1, 2), keepdim=True))

    @staticmethod
    def _candidate_softmax(scores: Tensor, history_mask: Tensor) -> Tensor:
        probabilities = torch.softmax(scores, dim=1)
        return probabilities * history_mask[:, None, :].to(scores.dtype)

    def forward(self, world: World) -> RankerOutput:
        batch = len(world)
        typed_query = world.query[:, None, :] + self.type_embedding[0]
        typed_history = world.history_key + self.type_embedding[1]
        typed_candidates = world.candidates + self.type_embedding[2]
        sequence = torch.cat((typed_query, typed_history, typed_candidates), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool),
                ~world.history_mask,
                torch.zeros((batch, C), dtype=torch.bool),
            ),
            dim=1,
        )
        contextual = self.encoder(
            sequence,
            mask=self.flow_mask(sequence.device),
            src_key_padding_mask=padding_mask,
        )
        query_state = contextual[:, 0]
        history_state = contextual[:, 1 : 1 + H]
        candidate_state = contextual[:, 1 + H :]
        q = torch.tanh(self.query_projection(query_state))
        c = self.candidate_projection(candidate_state)
        h = self.history_projection(history_state)
        semantic = torch.einsum("bcd,bhd,bd->bch", c, h, q) / 4.0
        qc_support = F.softplus(torch.einsum("bcd,bd->bc", c, q) / 4.0)
        valid = world.history_mask[:, None, :]
        exact = torch.where(valid, world.exact_match, torch.zeros_like(world.exact_match))
        scores = semantic + exact * qc_support[:, :, None]
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        values = self.value_projection(world.history_value)

        if self.method in {"PDSK", "CENTER0", "DIFF_ATTN"}:
            capacity = F.gelu(candidate_state) * self.theta
        elif self.method == "ITEM_ONLY":
            capacity = self.theta * (
                F.gelu(candidate_state)
                + 0.1 * F.gelu(self.history_projection(candidate_state))
            )
        elif self.method == "BASE_FFN":
            local = (
                c
                + self.history_projection(candidate_state)
                + self.value_projection(candidate_state)
                + q[:, None, :]
            )
            capacity = self.theta * self.output_projection(F.gelu(local))
        else:
            capacity = torch.zeros_like(candidate_state)

        weights = torch.zeros_like(scores)
        if self.method == "PDSK":
            weights = self.pdsk(scores, world.history_mask).weights
            history_delta = torch.einsum("bch,bhd->bcd", weights, values)
        elif self.method == "CENTER0":
            weights = self.center0(scores, world.history_mask).weights
            history_delta = torch.einsum("bch,bhd->bcd", weights, values)
        elif self.method == "GATED_CENTER":
            ranges = scores.amax(dim=1) - scores.amin(dim=1)
            ranges = ranges * world.history_mask.to(scores.dtype)
            request_active = (ranges.amax(dim=1) > TAU).to(scores.dtype)
            summary = c.mean(dim=1) * q
            amplitude = request_active * torch.sigmoid(
                (summary[:, :8] * self.theta[:8]).sum(dim=1) / math.sqrt(8.0)
            )
            temperature = 0.10 + F.softplus(
                (summary[:, 8:] * self.theta[8:]).sum(dim=1) / math.sqrt(8.0)
            )
            centered = self._candidate_softmax(
                scores / temperature[:, None, None], world.history_mask
            ) - (1.0 / C) * world.history_mask[:, None, :].to(scores.dtype)
            balance = amplitude[:, None, None] * centered
            weights = self._global_l1(balance)
            history_delta = torch.einsum("bch,bhd->bcd", weights, values)
        elif self.method == "TARGET_NULL":
            masked_scores = scores.masked_fill(~world.history_mask[:, None, :], -1e9)
            null_logits = torch.zeros((batch, C, 1), dtype=scores.dtype)
            attention = torch.softmax(torch.cat((masked_scores, null_logits), dim=2), dim=2)
            history_attention = attention[:, :, :H]
            null_attention = attention[:, :, H:]
            history_delta = torch.einsum("bch,bhd->bcd", history_attention, values)
            history_delta = history_delta + null_attention * self.theta
            present = world.history_mask.any(dim=1)[:, None, None]
            history_delta = torch.where(present, history_delta, torch.zeros_like(history_delta))
            weights = history_attention * present.to(scores.dtype)
        elif self.method == "DIFF_ATTN":
            first = torch.einsum(
                "bcd,bhd,bd->bch", c[:, :, :8], h[:, :, :8], q[:, :8]
            ) / math.sqrt(8.0)
            first = first + exact * qc_support[:, :, None]
            second = torch.einsum(
                "bcd,bhd,bd->bch", c[:, :, 8:], h[:, :, 8:], q[:, 8:]
            ) / math.sqrt(8.0)
            balance = self._candidate_softmax(first, world.history_mask) - self._candidate_softmax(
                second, world.history_mask
            )
            weights = self._global_l1(balance)
            history_delta = torch.einsum("bch,bhd->bcd", weights, values)
        elif self.method == "BASE_FFN":
            history_delta = torch.zeros_like(candidate_state)
        elif self.method == "ITEM_ONLY":
            balance = exact * qc_support[:, :, None]
            weights = self._global_l1(balance)
            history_delta = torch.einsum("bch,bhd->bcd", weights, values)
        else:  # pragma: no cover
            raise AssertionError(self.method)

        projected_delta = self.output_projection(history_delta)
        base_input = candidate_state + capacity
        base_logits = self.score_head(self.final_norm(base_input)).squeeze(-1)
        logits = self.score_head(self.final_norm(base_input + projected_delta)).squeeze(-1)
        return RankerOutput(logits, base_logits, scores, weights)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


class PermutationIndexStream:
    def __init__(self, size: int, generator: torch.Generator) -> None:
        self.size = size
        self.generator = generator
        self.buffer = torch.empty(0, dtype=torch.long)

    def take(self, count: int) -> Tensor:
        while self.buffer.numel() < count:
            self.buffer = torch.cat((self.buffer, torch.randperm(self.size, generator=self.generator)))
        result = self.buffer[:count]
        self.buffer = self.buffer[count:]
        return result


@dataclass(frozen=True)
class TrainingSchedule:
    r: Tensor
    s: Tensor
    n: Tensor
    u: tuple[dict[str, Tensor], ...]
    within: Tensor


def build_schedule(seed: int) -> TrainingSchedule:
    def two_epochs(stream_id: int) -> Tensor:
        generator = cpu_generator(seed, stream_id)
        return torch.cat(
            (torch.randperm(TRAIN_SIZE, generator=generator), torch.randperm(TRAIN_SIZE, generator=generator))
        ).reshape(UPDATES, BATCH_PER_WORLD)

    u_streams = {
        name: PermutationIndexStream(TRAIN_SIZE, cpu_generator(seed, 304 + index))
        for index, name in enumerate(CORRUPTIONS)
    }
    u_schedule: list[dict[str, Tensor]] = []
    rotations = ((6, 5, 5), (5, 6, 5), (5, 5, 6))
    for update in range(UPDATES):
        counts = rotations[update % 3]
        u_schedule.append(
            {
                name: u_streams[name].take(count)
                for name, count in zip(CORRUPTIONS, counts, strict=True)
            }
        )
    within_generator = cpu_generator(seed, 307)
    within = torch.stack([torch.randperm(64, generator=within_generator) for _ in range(UPDATES)])
    return TrainingSchedule(
        r=two_epochs(301),
        s=two_epochs(302),
        n=two_epochs(303),
        u=tuple(u_schedule),
        within=within,
    )


def batch_for_update(
    update: int,
    schedule: TrainingSchedule,
    r: World,
    s: World,
    u: dict[str, World],
    n: World,
) -> World:
    u_parts = [u[name].take(schedule.u[update][name]) for name in CORRUPTIONS]
    batch = concatenate_worlds(
        [
            r.take(schedule.r[update]),
            s.take(schedule.s[update]),
            *u_parts,
            n.take(schedule.n[update]),
        ]
    )
    if len(batch) != 64:
        raise RuntimeError("training batch is not 64")
    return batch.take(schedule.within[update])


def stable_order(logits: Tensor) -> Tensor:
    return torch.argsort(logits, dim=1, descending=True, stable=True)


def train_method(
    method: str,
    seed: int,
    worlds: dict[str, object],
    schedule: TrainingSchedule,
) -> tuple[SyntheticRanker, dict]:
    torch.manual_seed(seed * 1000 + 401)
    model = SyntheticRanker(method)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    last_loss = math.nan
    maximum_gradient_norm = 0.0
    for update in range(UPDATES):
        batch = batch_for_update(
            update,
            schedule,
            worlds["train_R"],
            worlds["train_S"],
            worlds["train_U"],
            worlds["train_N"],
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if not torch.isfinite(output.logits).all():
            raise RuntimeError(f"{method} seed {seed}: non-finite training logits")
        loss = F.cross_entropy(output.logits, batch.target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{method} seed {seed}: non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"{method} seed {seed}: non-finite gradient norm")
        maximum_gradient_norm = max(maximum_gradient_norm, float(norm))
        optimizer.step()
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"{method} seed {seed}: non-finite parameter")
        last_loss = float(loss)
    return model, {
        "final_loss": last_loss,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "updates": UPDATES,
    }


@torch.no_grad()
def evaluate_world(model: SyntheticRanker, world: World) -> dict:
    model.eval()
    correct = 0
    margin_sum = 0.0
    absolute_change_sum = 0.0
    flip_count = 0
    max_logit_mismatch = 0.0
    order_mismatch_requests = 0
    rank_mismatch_requests = 0
    finite = True
    for start in range(0, len(world), EVAL_BATCH):
        batch = world.take(torch.arange(start, min(start + EVAL_BATCH, len(world))))
        output = model(batch)
        finite = finite and bool(torch.isfinite(output.logits).all())
        order = stable_order(output.logits)
        base_order = stable_order(output.base_logits)
        correct += int((order[:, 0] == batch.target).sum())
        row = torch.arange(len(batch))
        target_logit = output.logits[row, batch.target]
        masked = output.logits.clone()
        masked[row, batch.target] = -torch.inf
        margin_sum += float((target_logit - masked.amax(dim=1)).sum())
        difference = output.logits - output.base_logits
        absolute_change_sum += float(difference.abs().sum())
        flip_count += int((order[:, 0] != base_order[:, 0]).sum())
        max_logit_mismatch = max(max_logit_mismatch, float(difference.abs().amax()))
        request_mismatch = (order != base_order).any(dim=1)
        order_mismatch_requests += int(request_mismatch.sum())
        output_rank = torch.argsort(order, dim=1, stable=True)
        base_rank = torch.argsort(base_order, dim=1, stable=True)
        rank_mismatch_requests += int((output_rank != base_rank).any(dim=1).sum())
    return {
        "requests": len(world),
        "top1_accuracy": correct / len(world),
        "target_margin_mean": margin_sum / len(world),
        "history_logit_change_mae": absolute_change_sum / (len(world) * C),
        "history_top1_flip_rate": flip_count / len(world),
        "max_logit_mismatch": max_logit_mismatch,
        "score_order_mismatch_requests": order_mismatch_requests,
        "rank_mismatch_requests": rank_mismatch_requests,
        "finite": finite,
    }


def pdsk_audits(model: SyntheticRanker, s_world: World) -> dict:
    model.eval()
    active = 0
    active_denominator = len(s_world) * H * C * (C - 1)
    conservation = 0.0
    common_mode_error = 0.0
    shifts = torch.tensor(
        [((-1.0) ** j) * 0.37 * (j + 1) for j in range(H)], dtype=torch.float64
    )[None, None, :]
    double_kernel = PairwiseSignedKernel(TAU, KAPPA).double()
    with torch.no_grad():
        for start in range(0, len(s_world), EVAL_BATCH):
            batch = s_world.take(torch.arange(start, min(start + EVAL_BATCH, len(s_world))))
            output = model(batch)
            scores = output.evidence_logits
            margins = scores[:, :, None, :] - scores[:, None, :, :]
            off_diagonal = ~torch.eye(C, dtype=torch.bool)[None, :, :, None]
            active += int(((margins.abs() > TAU) & off_diagonal).sum())
            scores64 = scores.double()
            original = double_kernel(scores64, batch.history_mask).weights
            shifted = double_kernel(scores64 + shifts, batch.history_mask).weights
            conservation = max(conservation, float(original.sum(dim=1).abs().amax()))
            common_mode_error = max(common_mode_error, float((original - shifted).abs().amax()))

    first64 = s_world.take(torch.arange(64))
    model.zero_grad(set_to_none=True)
    output = model(first64)
    gradient = torch.autograd.grad(
        F.cross_entropy(output.logits, first64.target), output.evidence_logits, retain_graph=False
    )[0]
    gradient_fraction = float((gradient.abs() > 1e-12).sum()) / gradient.numel()

    first256 = s_world.take(torch.arange(256))
    candidate_permutation = torch.tensor([2, 4, 1, 0, 3])
    history_permutation = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3])
    permuted = World(
        query=first256.query,
        candidates=first256.candidates[:, candidate_permutation],
        history_key=first256.history_key[:, history_permutation],
        history_value=first256.history_value[:, history_permutation],
        history_mask=first256.history_mask[:, history_permutation],
        exact_match=first256.exact_match[:, candidate_permutation][:, :, history_permutation],
        target=first256.target,
        foil=first256.foil,
        subthreshold_mask=first256.subthreshold_mask[:, history_permutation],
    )
    with torch.no_grad():
        reference = model(first256).logits
        changed = model(permuted).logits
    permutation_error = float((changed - reference[:, candidate_permutation]).abs().amax())
    return {
        "active_pair_fraction": active / active_denominator,
        "nonzero_evidence_gradient_fraction": gradient_fraction,
        "candidate_conservation_error": conservation,
        "common_mode_error": common_mode_error,
        "permutation_error": permutation_error,
    }


def generate_seed_worlds(seed: int) -> tuple[dict[str, object], dict[str, str]]:
    train_r = make_world(TRAIN_SIZE, cpu_generator(seed, 101), "R")
    train_s = make_world(TRAIN_SIZE, cpu_generator(seed, 102), "S")
    train_u_base_generator = cpu_generator(seed, 103)
    train_u_base = make_world(TRAIN_SIZE, train_u_base_generator, "U_BASE")
    train_u = make_u_corruptions(train_u_base, train_u_base_generator)
    train_n = make_world(TRAIN_SIZE, cpu_generator(seed, 104), "N")

    heldout_r = make_world(HELDOUT_SIZE, cpu_generator(seed, 201), "R")
    heldout_s = make_world(HELDOUT_SIZE, cpu_generator(seed, 202), "S")
    heldout_u_base_generator = cpu_generator(seed, 203)
    heldout_u_base = make_world(HELDOUT_SIZE, heldout_u_base_generator, "U_BASE")
    heldout_u = make_u_corruptions(heldout_u_base, heldout_u_base_generator)
    heldout_n = make_world(HELDOUT_SIZE, cpu_generator(seed, 204), "N")
    worlds: dict[str, object] = {
        "train_R": train_r,
        "train_S": train_s,
        "train_U": train_u,
        "train_N": train_n,
        "heldout_R": heldout_r,
        "heldout_S": heldout_s,
        "heldout_U": heldout_u,
        "heldout_N": heldout_n,
    }
    hashes = {
        "heldout_R": world_sha256(heldout_r),
        "heldout_S": world_sha256(heldout_s),
        "heldout_U_wrong_history": world_sha256(heldout_u["wrong_history"]),
        "heldout_U_shuffled_event": world_sha256(heldout_u["shuffled_event"]),
        "heldout_U_query_masked": world_sha256(heldout_u["query_masked"]),
        "heldout_N": world_sha256(heldout_n),
    }
    return worlds, hashes


def check_initial_equalization(seed: int) -> dict:
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        torch.manual_seed(seed * 1000 + 401)
        model = SyntheticRanker(method)
        counts[method] = parameter_count(model)
        hashes[method] = state_hash(model)
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"parameter counts differ: {counts}")
    if len(set(hashes.values())) != 1:
        raise RuntimeError("initial state hashes differ")
    return {
        "parameter_count": next(iter(counts.values())),
        "initial_state_sha256": next(iter(hashes.values())),
        "all_methods_equal": True,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def adjudicate(per_seed: dict[str, dict]) -> tuple[dict, dict]:
    seeds = [str(seed) for seed in SEEDS]
    checks: dict[str, bool] = {}

    checks["R_accuracy_every_seed_strictly_above_0.99"] = all(
        per_seed[seed]["methods"]["PDSK"]["R"]["top1_accuracy"] > 0.99 for seed in seeds
    )
    checks["R_margin_not_more_than_0.01_below_item_only"] = all(
        per_seed[seed]["methods"]["PDSK"]["R"]["target_margin_mean"]
        > per_seed[seed]["methods"]["ITEM_ONLY"]["R"]["target_margin_mean"] - 0.01
        for seed in seeds
    )
    rule1 = all(checks[key] for key in list(checks))

    checks["S_accuracy_every_seed_strictly_above_0.75"] = all(
        per_seed[seed]["methods"]["PDSK"]["S"]["top1_accuracy"] > 0.75 for seed in seeds
    )
    control_methods = ("CENTER0", "GATED_CENTER", "TARGET_NULL", "DIFF_ATTN")
    pdsk_s_mean = mean(
        per_seed[seed]["methods"]["PDSK"]["S"]["top1_accuracy"] for seed in seeds
    )
    control_means = {
        method: mean(
            per_seed[seed]["methods"][method]["S"]["top1_accuracy"] for seed in seeds
        )
        for method in control_methods
    }
    best_control = max(control_means, key=control_means.get)
    checks["S_mean_advantage_over_best_control_strictly_above_0.05"] = (
        pdsk_s_mean - control_means[best_control] > 0.05
    )
    checks["S_each_seed_advantage_over_gated_center_positive"] = all(
        per_seed[seed]["methods"]["PDSK"]["S"]["top1_accuracy"]
        > per_seed[seed]["methods"]["GATED_CENTER"]["S"]["top1_accuracy"]
        for seed in seeds
    )
    rule2 = all(
        checks[key]
        for key in (
            "S_accuracy_every_seed_strictly_above_0.75",
            "S_mean_advantage_over_best_control_strictly_above_0.05",
            "S_each_seed_advantage_over_gated_center_positive",
        )
    )

    for corruption in CORRUPTIONS:
        checks[f"U_{corruption}_flip_every_seed_strictly_below_0.01"] = all(
            per_seed[seed]["methods"]["PDSK"][f"U_{corruption}"][
                "history_top1_flip_rate"
            ]
            < 0.01
            for seed in seeds
        )
        checks[f"U_{corruption}_mae_every_seed_strictly_below_0.01"] = all(
            per_seed[seed]["methods"]["PDSK"][f"U_{corruption}"][
                "history_logit_change_mae"
            ]
            < 0.01
            for seed in seeds
        )
        checks[f"S_minus_U_{corruption}_every_seed_strictly_above_0.20"] = all(
            per_seed[seed]["methods"]["PDSK"]["S"]["top1_accuracy"]
            - per_seed[seed]["methods"]["PDSK"][f"U_{corruption}"]["top1_accuracy"]
            > 0.20
            for seed in seeds
        )
    rule3 = all(
        value for key, value in checks.items() if key.startswith("U_") or key.startswith("S_minus_U_")
    )

    checks["N_exact_logit_and_rank_fallback"] = all(
        per_seed[seed]["methods"]["PDSK"]["N"]["max_logit_mismatch"] == 0.0
        and per_seed[seed]["methods"]["PDSK"]["N"]["score_order_mismatch_requests"] == 0
        and per_seed[seed]["methods"]["PDSK"]["N"]["rank_mismatch_requests"] == 0
        for seed in seeds
    )
    rule4 = checks["N_exact_logit_and_rank_fallback"]

    checks["S_active_pair_fraction_strictly_inside_bounds"] = all(
        0.05 < per_seed[seed]["pdsk_audits"]["active_pair_fraction"] < 0.80
        for seed in seeds
    )
    checks["S_gradient_fraction_strictly_inside_bounds"] = all(
        0.05 < per_seed[seed]["pdsk_audits"]["nonzero_evidence_gradient_fraction"] < 0.80
        for seed in seeds
    )
    checks["all_reported_outputs_finite"] = all(
        metrics["finite"]
        for seed in seeds
        for method in per_seed[seed]["methods"].values()
        for name, metrics in method.items()
        if name != "training"
    )
    rule5 = all(
        checks[key]
        for key in (
            "S_active_pair_fraction_strictly_inside_bounds",
            "S_gradient_fraction_strictly_inside_bounds",
            "all_reported_outputs_finite",
        )
    )

    checks["conservation_every_seed_strictly_below_1e-8"] = all(
        per_seed[seed]["pdsk_audits"]["candidate_conservation_error"] < 1e-8
        for seed in seeds
    )
    checks["common_mode_every_seed_strictly_below_1e-8"] = all(
        per_seed[seed]["pdsk_audits"]["common_mode_error"] < 1e-8 for seed in seeds
    )
    checks["permutation_every_seed_strictly_below_1e-6"] = all(
        per_seed[seed]["pdsk_audits"]["permutation_error"] < 1e-6 for seed in seeds
    )
    rule6 = all(
        checks[key]
        for key in (
            "conservation_every_seed_strictly_below_1e-8",
            "common_mode_every_seed_strictly_below_1e-8",
            "permutation_every_seed_strictly_below_1e-6",
        )
    )
    rules = {
        "rule_1_R_preservation": rule1,
        "rule_2_S_positive_action_and_novelty": rule2,
        "rule_3_corruption_specificity": rule3,
        "rule_4_N_fallback": rule4,
        "rule_5_optimization_viability": rule5,
        "rule_6_post_training_algebra": rule6,
    }
    first_failure = next((name for name, passed in rules.items() if not passed), None)
    decision = {
        "passed": first_failure is None,
        "first_failed_rule": first_failure,
        "action": (
            "eligible_only_for_a_new_separately_locked_train_internal_smoke_design"
            if first_failure is None
            else "stop_c07_without_real_data_gpu_dev_or_test"
        ),
        "pdsk_S_mean_top1": pdsk_s_mean,
        "control_S_mean_top1": control_means,
        "best_S_control": best_control,
        "pdsk_minus_best_S_control": pdsk_s_mean - control_means[best_control],
    }
    return {"checks": checks, "rules": rules}, decision


def run_probe(lock: dict) -> dict:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        print(f"seed={seed} generating fixed worlds", flush=True)
        worlds, generated_hashes = generate_seed_worlds(seed)
        schedule = build_schedule(seed)
        equalization = check_initial_equalization(seed)
        seed_result: dict = {
            "generated_heldout_sha256": generated_hashes,
            "equalization": equalization,
            "methods": {},
        }
        pdsk_model: SyntheticRanker | None = None
        for method in METHODS:
            print(f"seed={seed} method={method} training 512 updates", flush=True)
            model, training = train_method(method, seed, worlds, schedule)
            method_metrics: dict[str, dict] = {
                "R": evaluate_world(model, worlds["heldout_R"]),
                "N": evaluate_world(model, worlds["heldout_N"]),
            }
            if method != "ITEM_ONLY":
                method_metrics["S"] = evaluate_world(model, worlds["heldout_S"])
                for corruption in CORRUPTIONS:
                    method_metrics[f"U_{corruption}"] = evaluate_world(
                        model, worlds["heldout_U"][corruption]
                    )
            seed_result["methods"][method] = {
                "training": training,
                **method_metrics,
            }
            if method == "PDSK":
                pdsk_model = model
        if pdsk_model is None:  # pragma: no cover
            raise AssertionError("PDSK model missing")
        seed_result["pdsk_audits"] = pdsk_audits(pdsk_model, worlds["heldout_S"])
        per_seed[str(seed)] = seed_result
    adjudication, decision = adjudicate(per_seed)
    return {
        "schema_version": 1,
        "candidate": "C07_pairwise_deadzone_signed_kernel_transformer",
        "probe": "G1_semantic_synthetic_cpu",
        "execution_lock_combined_sha256": lock["combined_manifest_sha256"],
        "old_normative_lock_combined_sha256": LOCKED_MANIFEST,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "repository_data_read": False,
            "labels_or_qrels_read": False,
            "dev_or_test_evaluator_called": False,
            "gpu_used": False,
        },
        "fixed_constants": {
            "seeds": list(SEEDS),
            "candidates": C,
            "history_events": H,
            "width": D,
            "tau": TAU,
            "kappa": KAPPA,
            "train_per_world_seed": TRAIN_SIZE,
            "heldout_per_world_seed": HELDOUT_SIZE,
            "updates": UPDATES,
            "batch_size": 64,
        },
        "per_seed": per_seed,
        "adjudication": adjudication,
        "decision": decision,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lock = verify_execution_lock(args.execution_lock, args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    result = run_probe(lock)
    atomic_write_json(args.output, result)
    print(
        f"decision_passed={result['decision']['passed']} "
        f"first_failed_rule={result['decision']['first_failed_rule']}",
        flush=True,
    )
    print(f"raw_result={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
