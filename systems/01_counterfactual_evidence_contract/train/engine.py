"""Training, calibration, and train-internal falsification for C01."""

from __future__ import annotations

import copy
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from model.cect import (
    CECTModel,
    CECTOutput,
    counterfactual_margin_loss,
    counterfactual_upper_quantile,
    masked_zscore,
    model_signature,
    multi_positive_listwise_loss,
)
from train.data import PackedSplit, move_batch, request_batches, sha256_file


TWIN_CONDITIONS = ("wrong", "shuffled", "query_masked", "coarse")


def model_from_config(config: dict[str, Any], mode: str) -> CECTModel:
    model = config["model"]
    return CECTModel(
        frozen_text_dim=int(model["frozen_text_dim"]),
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        dim_feedforward=int(model["dim_feedforward"]),
        dropout=float(model["dropout"]),
        max_history=int(model["max_history"]),
        category_buckets=int(config["data"]["category_buckets"]),
        beta=float(model["beta"]),
        gate_temperature=float(model["gate_temperature"]),
        mode=mode,
    )


def _autocast(device: str, config: dict[str, Any]):
    if not device.startswith("cuda"):
        return nullcontext()
    dtype_name = str(config["training"]["amp_dtype"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported AMP dtype: {dtype_name}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _assert_finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite {label}")


def _optimizer(
    model: CECTModel, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer has no trainable parameters")
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )


def _run_epoch(
    *,
    model: CECTModel,
    split: PackedSplit,
    indices: Iterable[int],
    config: dict[str, Any],
    device: str,
    optimizer: torch.optim.Optimizer,
    epoch_seed: int,
    use_counterfactual_loss: bool,
    progress_label: str,
    donor_bounds: tuple[int, int],
) -> dict[str, Any]:
    model.train()
    training = config["training"]
    model_config = config["model"]
    total_loss = 0.0
    total_rank_loss = 0.0
    total_cf_loss = 0.0
    steps = 0
    started = time.monotonic()
    batches = request_batches(
        indices,
        int(training["requests_per_batch"]),
        shuffle=True,
        seed=epoch_seed,
    )
    for raw_batch in batches:
        wrong = split.default_wrong_indices(
            raw_batch, donor_start=donor_bounds[0], donor_end=donor_bounds[1]
        )
        batch = move_batch(
            split.build_batch(
                raw_batch, all_candidates=False, wrong_indices=wrong
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config):
            true_output = model(batch, condition="true")
            rank_loss = multi_positive_listwise_loss(
                true_output.scores, batch["labels"], batch["candidate_mask"]
            )
            if use_counterfactual_loss:
                twins = [
                    model(batch, condition=condition)
                    for condition in TWIN_CONDITIONS
                ]
                cf_loss = counterfactual_margin_loss(
                    true_output,
                    twins,
                    batch["labels"],
                    batch["candidate_mask"],
                    margin=float(model_config["cf_margin"]),
                    temperature=float(model_config["lse_temperature"]),
                )
            else:
                cf_loss = rank_loss * 0.0
            loss = rank_loss + float(model_config["cf_loss_weight"]) * cf_loss
        _assert_finite(loss, "training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(training["gradient_clip_norm"]),
        )
        _assert_finite(torch.as_tensor(gradient_norm), "gradient norm")
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_rank_loss += float(rank_loss.detach().cpu())
        total_cf_loss += float(cf_loss.detach().cpu())
        steps += 1
        if steps % 250 == 0:
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "label": progress_label,
                        "mean_loss": total_loss / steps,
                        "steps": steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if steps == 0:
        raise ValueError("training epoch produced zero optimizer steps")
    return {
        "cf_loss": total_cf_loss / steps,
        "duration_seconds": time.monotonic() - started,
        "loss": total_loss / steps,
        "optimizer_steps": steps,
        "rank_loss": total_rank_loss / steps,
    }


@torch.no_grad()
def calibrate_certificate(
    model: CECTModel,
    split: PackedSplit,
    indices: Iterable[int],
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Freeze Q_cf from counterfactual energies on the train-only slice."""

    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        condition: [] for condition in TWIN_CONDITIONS
    }
    batch_size = int(config["training"]["requests_per_batch"])
    for raw_batch in request_batches(
        indices, batch_size, shuffle=False, seed=int(config["seed"])
    ):
        wrong = split.default_wrong_indices(
            raw_batch,
            donor_start=int(config["data"]["calibration_start"]),
            donor_end=int(config["data"]["calibration_end"]),
        )
        batch = move_batch(
            split.build_batch(
                raw_batch, all_candidates=False, wrong_indices=wrong
            ),
            device,
        )
        with _autocast(device, config):
            true_output = model(batch, condition="true")
            eligible_candidate = (
                batch["labels"].bool()
                & batch["candidate_mask"].bool()
                & ~(true_output.exact_scores > 0)
            )
            for condition in TWIN_CONDITIONS:
                output = model(batch, condition=condition)
                selected = output.energies[
                    eligible_candidate[:, :, None] & output.nonexact_mask
                ]
                if selected.numel():
                    collected[condition].append(
                        selected.detach().float().cpu().numpy()
                    )
    arrays = {
        condition: np.concatenate(parts)
        for condition, parts in collected.items()
        if parts
    }
    if set(arrays) != set(TWIN_CONDITIONS):
        raise ValueError(f"empty calibration condition(s): {set(TWIN_CONDITIONS) - set(arrays)}")
    pooled = np.concatenate([arrays[condition] for condition in TWIN_CONDITIONS])
    threshold_tensor = counterfactual_upper_quantile(
        torch.from_numpy(pooled), float(config["model"]["alpha_cf"])
    )
    threshold = float(threshold_tensor.item())
    model.set_certificate_threshold(threshold)
    false_admission = {
        condition: float(np.mean(values > threshold))
        for condition, values in arrays.items()
    }
    return {
        "alpha_cf": float(config["model"]["alpha_cf"]),
        "false_admission_rate": false_admission,
        "num_counterfactual_event_energies": int(len(pooled)),
        "per_condition_count": {
            condition: int(len(values)) for condition, values in arrays.items()
        },
        "threshold": threshold,
    }


def train_models(
    config: dict[str, Any], split: PackedSplit, device: str
) -> tuple[CECTModel, CECTModel, dict[str, Any]]:
    """Train CECT and its parameter-matched plain-attention control."""

    seed = int(config["seed"])
    torch.manual_seed(seed)
    contract = model_from_config(config, "contract").to(device)
    initial_state = copy.deepcopy(contract.state_dict())
    torch.manual_seed(seed)
    plain = model_from_config(config, "plain").to(device)
    plain.load_state_dict(initial_state, strict=True)
    for (left_name, left), (right_name, right) in zip(
        contract.named_parameters(), plain.named_parameters()
    ):
        if left_name != right_name or not torch.equal(left, right):
            raise ValueError("matched control did not receive identical initialization")
    if contract.parameter_count() != plain.parameter_count():
        raise ValueError("contract/plain parameter counts differ")

    fit = range(int(config["data"]["fit_start"]), int(config["data"]["fit_end"]))
    calibration = range(
        int(config["data"]["calibration_start"]),
        int(config["data"]["calibration_end"]),
    )
    training = config["training"]
    log: dict[str, Any] = {
        "contract": [],
        "plain": [],
        "parameter_count": contract.parameter_count(),
    }
    started = time.monotonic()

    contract_optimizer = _optimizer(
        contract,
        float(training["learning_rate_stage1"]),
        float(training["weight_decay"]),
    )
    for epoch in range(int(training["stage1_epochs"])):
        log["contract"].append(
            _run_epoch(
                model=contract,
                split=split,
                indices=fit,
                config=config,
                device=device,
                optimizer=contract_optimizer,
                epoch_seed=seed + epoch,
                use_counterfactual_loss=True,
                progress_label=f"contract_stage1_epoch{epoch + 1}",
                donor_bounds=(
                    int(config["data"]["fit_start"]),
                    int(config["data"]["fit_end"]),
                ),
            )
        )
    log["calibration"] = calibrate_certificate(
        contract, split, calibration, config, device
    )
    contract.freeze_certificate_path()
    contract_optimizer = _optimizer(
        contract,
        float(training["learning_rate_stage2"]),
        float(training["weight_decay"]),
    )
    for epoch in range(int(training["stage2_epochs"])):
        log["contract"].append(
            _run_epoch(
                model=contract,
                split=split,
                indices=fit,
                config=config,
                device=device,
                optimizer=contract_optimizer,
                epoch_seed=seed + int(training["stage1_epochs"]) + epoch,
                use_counterfactual_loss=False,
                progress_label=f"contract_stage2_epoch{epoch + 1}",
                donor_bounds=(
                    int(config["data"]["fit_start"]),
                    int(config["data"]["fit_end"]),
                ),
            )
        )

    plain_optimizer = _optimizer(
        plain,
        float(training["learning_rate_stage1"]),
        float(training["weight_decay"]),
    )
    for epoch in range(int(training["stage1_epochs"])):
        log["plain"].append(
            _run_epoch(
                model=plain,
                split=split,
                indices=fit,
                config=config,
                device=device,
                optimizer=plain_optimizer,
                epoch_seed=seed + epoch,
                use_counterfactual_loss=False,
                progress_label=f"plain_stage1_epoch{epoch + 1}",
                donor_bounds=(
                    int(config["data"]["fit_start"]),
                    int(config["data"]["fit_end"]),
                ),
            )
        )
    plain_optimizer = _optimizer(
        plain,
        float(training["learning_rate_stage2"]),
        float(training["weight_decay"]),
    )
    for epoch in range(int(training["stage2_epochs"])):
        log["plain"].append(
            _run_epoch(
                model=plain,
                split=split,
                indices=fit,
                config=config,
                device=device,
                optimizer=plain_optimizer,
                epoch_seed=seed + int(training["stage1_epochs"]) + epoch,
                use_counterfactual_loss=False,
                progress_label=f"plain_stage2_epoch{epoch + 1}",
                donor_bounds=(
                    int(config["data"]["fit_start"]),
                    int(config["data"]["fit_end"]),
                ),
            )
        )
    log["duration_seconds"] = time.monotonic() - started
    log["gpu_hours"] = log["duration_seconds"] / 3600.0
    if log["gpu_hours"] > float(training["max_gpu_hours"]):
        raise RuntimeError(
            f"GPU-hour budget exceeded: {log['gpu_hours']:.4f} > {training['max_gpu_hours']}"
        )
    contract.eval()
    plain.eval()
    return contract, plain, log


def save_checkpoint(
    path: str | Path,
    model: CECTModel,
    config_path: str | Path,
    candidate_hash: str,
    training_log: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    torch.save(
        {
            "candidate_hash": candidate_hash,
            "config_sha256": sha256_file(config_path),
            "mode": model.mode,
            "model_signature": model_signature(model),
            "state_dict": model.state_dict(),
            "training_log": training_log,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    config: dict[str, Any],
    candidate_hash: str,
    device: str,
) -> tuple[CECTModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("candidate_hash") != candidate_hash:
        raise ValueError("checkpoint candidate hash differs from proposal lock")
    model = model_from_config(config, str(checkpoint["mode"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _compose_evidence_scores(
    base_scores: torch.Tensor,
    evidence_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    history_present: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    minimum = evidence_scores.masked_fill(~candidate_mask, torch.inf).amin(dim=-1)
    maximum = evidence_scores.masked_fill(~candidate_mask, -torch.inf).amax(dim=-1)
    available = history_present & ((maximum - minimum) > 1e-8)
    mixed = beta * masked_zscore(base_scores, candidate_mask) + (1.0 - beta) * masked_zscore(
        evidence_scores, candidate_mask
    )
    return torch.where(available[:, None], mixed, base_scores).masked_fill(
        ~candidate_mask, 0.0
    )


def _request_ndcg(
    request_id: str,
    item_ids: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    # Imported here so model-only unit tests do not need repository src on path.
    from myrec.eval.metrics import ScoredCandidate, request_metrics

    positives = {
        item_id for item_id, label in zip(item_ids, labels) if float(label) > 0.0
    }
    row = request_metrics(
        request_id=request_id,
        scored_candidates=[
            ScoredCandidate(item_id=item_id, score=float(score))
            for item_id, score in zip(item_ids, scores)
        ],
        clicked_item_ids=positives,
        purchased_item_ids=set(),
    )
    return float(row["ndcg@10"])


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, samples: int, seed: int
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap requires non-empty equal vectors")
    delta = left - right
    generator = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    chunk = 100
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = generator.integers(0, delta.size, size=(stop - start, delta.size))
        boot[start:stop] = delta[indices].mean(axis=1)
    return {
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "mean_delta": float(delta.mean()),
        "num_requests": int(delta.size),
    }


@dataclass
class EventStats:
    energy_sum: float = 0.0
    energy_square_sum: float = 0.0
    event_count: int = 0
    admitted_count: int = 0
    mass_sum: float = 0.0
    candidate_count: int = 0

    def update(
        self,
        output: CECTOutput,
        row_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> None:
        surface = row_mask[:, None, None] & candidate_mask[:, :, None]
        events = surface & output.nonexact_mask
        values = output.energies[events].detach().float()
        self.energy_sum += float(values.sum().cpu())
        self.energy_square_sum += float(values.square().sum().cpu())
        self.event_count += int(events.sum().item())
        self.admitted_count += int((output.hard_admission & events).sum().item())
        candidates = row_mask[:, None] & candidate_mask & output.nonexact_mask.any(dim=-1)
        self.mass_sum += float(
            (output.gates.sum(dim=-1) * candidates.to(output.gates.dtype)).sum().detach().cpu()
        )
        self.candidate_count += int(candidates.sum().item())

    def summary(self) -> dict[str, float | int]:
        if self.event_count == 0 or self.candidate_count == 0:
            raise ValueError("event diagnostic surface is empty")
        mean = self.energy_sum / self.event_count
        variance = max(self.energy_square_sum / self.event_count - mean * mean, 0.0)
        return {
            "admission_rate": self.admitted_count / self.event_count,
            "energy_mean": mean,
            "energy_std": math.sqrt(variance),
            "event_count": self.event_count,
            "mean_admitted_mass": self.mass_sum / self.candidate_count,
            "candidate_count": self.candidate_count,
        }


@torch.no_grad()
def run_internal_falsifier(
    contract: CECTModel,
    plain: CECTModel,
    split: PackedSplit,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Apply all six frozen train/internal gates without reading dev labels."""

    contract.eval()
    plain.eval()
    start = int(config["data"]["internal_start"])
    end = int(config["data"]["internal_end"])
    variants = ("base", "item_only", "contract", *TWIN_CONDITIONS, "plain")
    ndcg: dict[str, list[float]] = {variant: [] for variant in variants}
    repeat_flags: list[bool] = []
    nonrepeat_flags: list[bool] = []
    no_history_flags: list[bool] = []
    all_history_stats = {
        "true": EventStats(),
        **{condition: EventStats() for condition in TWIN_CONDITIONS},
    }
    nonrepeat_stats = {
        "true": EventStats(),
        **{condition: EventStats() for condition in TWIN_CONDITIONS},
    }
    no_history_max_abs = 0.0
    no_history_rank_mismatches = 0
    beta = float(config["model"]["beta"])
    internal_batch_size = min(int(config["training"]["requests_per_batch"]), 24)

    for raw_batch in request_batches(
        range(start, end), internal_batch_size, shuffle=False, seed=int(config["seed"])
    ):
        wrong = split.default_wrong_indices(
            raw_batch, donor_start=start, donor_end=end
        )
        batch = move_batch(
            split.build_batch(
                raw_batch, all_candidates=True, wrong_indices=wrong
            ),
            device,
        )
        true_output = contract(batch, condition="true")
        twin_outputs = {
            condition: contract(batch, condition=condition)
            for condition in TWIN_CONDITIONS
        }
        plain_output = plain(batch, condition="true")
        history_present = batch["history_mask"].any(dim=-1)
        repeat = (true_output.exact_scores > 0).any(dim=-1)
        nonrepeat = history_present & ~repeat
        no_history = ~history_present
        item_only_scores = _compose_evidence_scores(
            batch["base_scores"],
            true_output.exact_scores,
            batch["candidate_mask"].bool(),
            history_present,
            beta,
        )
        tensor_scores = {
            "base": batch["base_scores"],
            "item_only": item_only_scores,
            "contract": true_output.scores,
            "plain": plain_output.scores,
            **{
                condition: output.scores
                for condition, output in twin_outputs.items()
            },
        }
        for output in (true_output, plain_output, *twin_outputs.values()):
            _assert_finite(output.scores, "internal score")

        if no_history.any():
            difference = (
                true_output.scores[no_history] - batch["base_scores"][no_history]
            ).abs()
            no_history_max_abs = max(
                no_history_max_abs, float(difference.max().cpu())
            )
            for row in torch.nonzero(no_history, as_tuple=False).flatten().tolist():
                width = int(batch["candidate_mask"][row].sum().item())
                if not torch.equal(
                    true_output.scores[row, :width], batch["base_scores"][row, :width]
                ):
                    no_history_rank_mismatches += 1

        for label, output in (("true", true_output), *twin_outputs.items()):
            all_history_stats[label].update(
                output, history_present, batch["candidate_mask"].bool()
            )
            nonrepeat_stats[label].update(
                output, nonrepeat, batch["candidate_mask"].bool()
            )

        repeat_flags.extend(bool(value) for value in repeat.cpu().tolist())
        nonrepeat_flags.extend(bool(value) for value in nonrepeat.cpu().tolist())
        no_history_flags.extend(bool(value) for value in no_history.cpu().tolist())
        for row, request_id in enumerate(batch["request_ids"]):
            width = int(batch["candidate_mask"][row].sum().item())
            labels = batch["labels"][row, :width].detach().cpu().numpy()
            item_ids = batch["candidate_item_id_rows"][row]
            for variant in variants:
                scores = tensor_scores[variant][row, :width].detach().float().cpu().numpy()
                ndcg[variant].append(
                    _request_ndcg(request_id, item_ids, scores, labels)
                )

    metrics = {key: np.asarray(values, dtype=np.float64) for key, values in ndcg.items()}
    repeat_mask = np.asarray(repeat_flags, dtype=bool)
    nonrepeat_mask = np.asarray(nonrepeat_flags, dtype=bool)
    no_history_mask = np.asarray(no_history_flags, dtype=bool)
    bootstrap_samples = int(config["internal_gate"]["bootstrap_samples"])
    seed = int(config["seed"])
    repeat_comparison = paired_bootstrap(
        metrics["contract"][repeat_mask],
        metrics["item_only"][repeat_mask],
        bootstrap_samples,
        seed,
    )
    nonrepeat_comparison = paired_bootstrap(
        metrics["contract"][nonrepeat_mask],
        metrics["base"][nonrepeat_mask],
        bootstrap_samples,
        seed,
    )
    plain_comparison = paired_bootstrap(
        metrics["contract"][nonrepeat_mask],
        metrics["plain"][nonrepeat_mask],
        bootstrap_samples,
        seed,
    )
    true_gain = float(nonrepeat_comparison["mean_delta"])
    true_nonrepeat_stats = nonrepeat_stats["true"].summary()
    corruption: dict[str, Any] = {}
    for condition in TWIN_CONDITIONS:
        twin_gain = float(
            (metrics[condition][nonrepeat_mask] - metrics["base"][nonrepeat_mask]).mean()
        )
        stats = nonrepeat_stats[condition].summary()
        admission_drop = (
            (float(true_nonrepeat_stats["admission_rate"]) - float(stats["admission_rate"]))
            / float(true_nonrepeat_stats["admission_rate"])
            if float(true_nonrepeat_stats["admission_rate"]) > 0
            else None
        )
        corruption[condition] = {
            "admission_drop_relative": admission_drop,
            "ndcg_gain_over_base": twin_gain,
            "recovery_fraction": twin_gain / true_gain if true_gain > 0 else None,
            "stats": stats,
        }

    all_stats = {
        label: accumulator.summary()
        for label, accumulator in all_history_stats.items()
    }
    pooled_twin_energy = float(
        np.mean([all_stats[condition]["energy_mean"] for condition in TWIN_CONDITIONS])
    )
    shuffle_mass_drop = (
        (float(all_stats["true"]["mean_admitted_mass"]) - float(all_stats["shuffled"]["mean_admitted_mass"]))
        / float(all_stats["true"]["mean_admitted_mass"])
        if float(all_stats["true"]["mean_admitted_mass"]) > 0
        else None
    )

    thresholds = config["internal_gate"]
    gate_items = {
        "protected_recurrence": (
            float(repeat_comparison["mean_delta"]) >= float(thresholds["repeat_delta_min"])
            and float(repeat_comparison["ci95_lower"]) >= float(thresholds["repeat_ci_lower_min"])
        ),
        "nonrepeat_transfer": (
            float(nonrepeat_comparison["mean_delta"]) >= float(thresholds["nonrepeat_delta_min"])
            and float(nonrepeat_comparison["ci95_lower"]) > 0.0
        ),
        "counterfactual_rejection": (
            true_gain > 0.0
            and all(
                item["recovery_fraction"] is not None
                and float(item["recovery_fraction"]) <= float(thresholds["corruption_max_recovery"])
                and item["admission_drop_relative"] is not None
                and float(item["admission_drop_relative"])
                >= float(thresholds["corruption_min_admission_drop_relative"])
                for item in corruption.values()
            )
        ),
        "no_history_contract": (
            no_history_max_abs == 0.0
            and no_history_rank_mismatches == 0
            and float((metrics["contract"][no_history_mask] - metrics["base"][no_history_mask]).sum()) == 0.0
        ),
        "noncollapse_and_order": (
            float(all_stats["true"]["admission_rate"]) >= float(thresholds["admission_rate_min"])
            and float(all_stats["true"]["admission_rate"]) <= float(thresholds["admission_rate_max"])
            and float(all_stats["true"]["energy_std"]) >= float(thresholds["certificate_std_min"])
            and float(all_stats["true"]["energy_mean"]) - pooled_twin_energy
            >= float(thresholds["true_twin_energy_gap_min"])
            and shuffle_mass_drop is not None
            and shuffle_mass_drop >= float(thresholds["shuffle_mass_drop_relative_min"])
        ),
        "matched_plain_control": (
            contract.parameter_count() == plain.parameter_count()
            and float(plain_comparison["mean_delta"]) >= float(thresholds["plain_delta_min"])
            and float(plain_comparison["ci95_lower"]) > 0.0
        ),
    }
    return {
        "all_passed": all(gate_items.values()),
        "corruption": corruption,
        "gate_items": gate_items,
        "matched_parameter_count": {
            "contract": contract.parameter_count(),
            "equal": contract.parameter_count() == plain.parameter_count(),
            "plain": plain.parameter_count(),
        },
        "no_history": {
            "max_absolute_score_difference": no_history_max_abs,
            "ndcg_delta_sum": float(
                (metrics["contract"][no_history_mask] - metrics["base"][no_history_mask]).sum()
            ),
            "rank_mismatches": no_history_rank_mismatches,
        },
        "noncollapse": {
            "all_history_present_stats": all_stats,
            "pooled_twin_energy_mean": pooled_twin_energy,
            "shuffle_mass_drop_relative": shuffle_mass_drop,
            "true_minus_pooled_twin_energy": float(all_stats["true"]["energy_mean"])
            - pooled_twin_energy,
        },
        "nonrepeat_vs_base": nonrepeat_comparison,
        "plain_control": plain_comparison,
        "protected_recurrence": repeat_comparison,
        "request_counts": {
            "internal": int(len(metrics["contract"])),
            "no_history": int(no_history_mask.sum()),
            "nonrepeat_history_present": int(nonrepeat_mask.sum()),
            "repeat_present": int(repeat_mask.sum()),
        },
        "subset_mean_ndcg10": {
            variant: {
                "all": float(values.mean()),
                "nonrepeat_history_present": float(values[nonrepeat_mask].mean()),
                "repeat_present": float(values[repeat_mask].mean()),
            }
            for variant, values in metrics.items()
        },
    }
