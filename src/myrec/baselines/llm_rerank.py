"""LLM top-k reranking baselines for PPS Batch 2."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_llm_rerank_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    variant: str,
    history_len: int,
    base_run_id: str,
    subset_request_ids_path: str | Path,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    cache_dir: str | Path = "models/huggingface/llm",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    top_k: int = 20,
    max_new_tokens: int = 64,
    generation_batch_size: int = 8,
) -> dict[str, Any]:
    """Run an LLM reranker on a fixed request subset and base-score fallback elsewhere."""

    if variant not in {"b8a", "b8b"}:
        raise ValueError(f"unknown B8 variant: {variant}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    standardized_dir = Path(standardized_dir)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subset_request_ids_path = Path(subset_request_ids_path)
    subset_ids = _read_request_ids(subset_request_ids_path)
    base_scores = _load_base_scores(runs_dir / base_run_id / "scores.jsonl")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        dtype=torch_dtype,
    ).to(device)
    model.eval()

    scores_path = run_dir / "scores.jsonl"
    trace_path = run_dir / "llm_trace.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    method_id = f"{variant}_llm_rerank_h{history_len}"
    rows = 0
    reranked_requests = 0
    parse_failures = 0
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    started = time.perf_counter()
    subset_records: list[dict[str, Any]] = []

    with scores_path.open("w", encoding="utf-8") as score_handle, trace_path.open("w", encoding="utf-8") as trace_handle:
        for record in iter_jsonl(standardized_dir / f"records_{split}.jsonl"):
            request_id = str(record["request_id"])
            request_base_scores = base_scores[request_id]
            if request_id not in subset_ids:
                for candidate in record["candidates"]:
                    item_id = str(candidate["item_id"])
                    _write_score(score_handle, method_id, request_id, item_id, request_base_scores[item_id])
                    rows += 1
                continue

            subset_records.append(record)
            reranked_requests += 1

        prepared = []
        memory_outputs: list[str | None] = [None] * len(subset_records)
        if variant == "b8b":
            memory_prompts = []
            for record in subset_records:
                history = list(record.get("history", []))[-history_len:]
                memory_prompts.append(_memory_prompt(record, history))
            memory_outputs, memory_in_tokens, memory_out_tokens = _generate_text_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=memory_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                batch_size=generation_batch_size,
            )
            prompt_tokens += sum(memory_in_tokens)
            completion_tokens += sum(memory_out_tokens)
            llm_calls += len(memory_prompts)

        rerank_prompts = []
        for index, record in enumerate(subset_records):
            request_base_scores = base_scores[str(record["request_id"])]
            top_candidates = _top_candidates(record, request_base_scores, top_k)
            history = list(record.get("history", []))[-history_len:]
            prepared.append(
                {
                    "history": history,
                    "record": record,
                    "top_candidates": top_candidates,
                }
            )
            rerank_prompts.append(
                _rerank_prompt(
                    record,
                    history,
                    top_candidates,
                    variant=variant,
                    memory=memory_outputs[index],
                )
            )
        rerank_outputs, rerank_in_tokens, rerank_out_tokens = _generate_text_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=rerank_prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size,
        )
        prompt_tokens += sum(rerank_in_tokens)
        completion_tokens += sum(rerank_out_tokens)
        llm_calls += len(rerank_prompts)

        for prepared_row, output in zip(prepared, rerank_outputs):
            record = prepared_row["record"]
            request_id = str(record["request_id"])
            request_base_scores = base_scores[request_id]
            top_candidates = prepared_row["top_candidates"]
            order = _parse_order(output, len(top_candidates))
            parse_failed = order is None
            if parse_failed:
                parse_failures += 1
                order = list(range(1, len(top_candidates) + 1))
            reranked_item_ids = [top_candidates[index - 1]["item_id"] for index in order]
            rerank_scores = _scores_with_reranked_topk(record, request_base_scores, reranked_item_ids)
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                _write_score(score_handle, method_id, request_id, item_id, rerank_scores[item_id])
                rows += 1
            trace_handle.write(
                json.dumps(
                    {
                        "completion": output,
                        "history_len": history_len,
                        "parse_failed": parse_failed,
                        "request_id": request_id,
                        "variant": variant,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    elapsed = time.perf_counter() - started
    metadata = {
        "base_run_id": base_run_id,
        "base_scores_path": str(runs_dir / base_run_id / "scores.jsonl"),
        "base_scores_sha256": sha256_file(runs_dir / base_run_id / "scores.jsonl"),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "config_path": str(config_path) if config_path else None,
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "device": device,
        "dtype": dtype,
        "fallback": "non-subset requests and parse failures use base-run order/scores",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_len": history_len,
        "input_fields_used": [
            "query",
            "history.title",
            "history.cat",
            "history.event",
            "candidates.title",
            "candidates.brand",
            "candidates.seller",
            "candidates.cat",
            "base_run_scores_for_top20_truncation_and_fallback",
        ],
        "llm_calls": llm_calls,
        "generation_batch_size": generation_batch_size,
        "max_new_tokens": max_new_tokens,
        "method_id": method_id,
        "model_name": model_name,
        "parse_failure_rate": parse_failures / reranked_requests if reranked_requests else 0.0,
        "parse_failures": parse_failures,
        "qrels_read": False,
        "request_count": len(base_scores),
        "reranked_requests": reranked_requests,
        "run_id": run_id,
        "score_definition": "LLM reranks base-run top-20 on a fixed dev subset; all other candidates preserve base fallback scores",
        "score_rows": rows,
        "split": split,
        "standardized_dir": str(standardized_dir),
        "subset_request_ids_path": str(subset_request_ids_path),
        "subset_request_ids_sha256": sha256_file(subset_request_ids_path),
        "timing": {
            "requests_per_second": reranked_requests / elapsed if elapsed else None,
            "seconds_total": elapsed,
        },
        "token_usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "top_k": top_k,
        "tuning": {
            "class": "zero-shot",
            "dev_eval_budget": 3,
            "history_length_trial": history_len,
        },
        "variant": variant,
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def ensure_subset_request_ids(
    standardized_dir: str | Path,
    split: str,
    output_path: str | Path,
    sample_size: int = 2000,
    seed: int = 20260708,
) -> dict[str, Any]:
    import random

    output_path = Path(output_path)
    if not output_path.exists():
        request_ids = [str(record["request_id"]) for record in iter_jsonl(Path(standardized_dir) / f"records_{split}.jsonl")]
        rng = random.Random(seed)
        sampled = sorted(rng.sample(request_ids, min(sample_size, len(request_ids))))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for request_id in sampled:
                handle.write(request_id + "\n")
    request_ids = _read_request_ids(output_path)
    return {
        "path": str(output_path),
        "requests": len(request_ids),
        "seed": seed,
        "sha256": sha256_file(output_path),
    }


def _generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[int], list[int]]:
    outputs = []
    prompt_tokens = []
    completion_tokens = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch_prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_width = int(inputs.input_ids.shape[1])
        for row_index in range(len(batch_prompts)):
            completion_ids = output_ids[row_index][input_width:]
            outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
            prompt_tokens.append(int(inputs.attention_mask[row_index].sum().item()))
            completion_tokens.append(int(completion_ids.shape[0]))
    return outputs, prompt_tokens, completion_tokens


def _memory_prompt(record: dict[str, Any], history: list[dict[str, Any]]) -> str:
    history_text = _history_text(history)
    return (
        "你是电商搜索个性化记忆抽取器。根据用户最近行为，提炼最多5条购物偏好。"
        "只输出短句，不要解释。\n"
        f"当前查询：{record.get('query', '')}\n"
        f"历史：\n{history_text}"
    )


def _rerank_prompt(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    candidates: list[dict[str, str]],
    variant: str,
    memory: str | None,
) -> str:
    candidate_text = "\n".join(f"{idx}. {row['text']}" for idx, row in enumerate(candidates, start=1))
    if variant == "b8b":
        context = f"偏好记忆：{memory or '无'}"
    else:
        context = f"用户历史：\n{_history_text(history)}"
    return (
        "你是电商搜索重排器。请根据查询、用户偏好和候选商品，把候选按用户最可能点击的顺序排序。"
        "只输出一个JSON数组，元素是候选编号，例如 [3,1,2]，不要输出解释。\n"
        f"查询：{record.get('query', '')}\n"
        f"{context}\n"
        f"候选：\n{candidate_text}"
    )


def _history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "无"
    rows = []
    for event in history:
        cat = "/".join(str(part) for part in event.get("cat", []) if part)
        rows.append(f"- {event.get('event', 'click')}: {event.get('title', '')} {cat}")
    return "\n".join(rows)


def _parse_order(text: str, count: int) -> list[int] | None:
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    result = []
    seen = set()
    for number in numbers:
        if 1 <= number <= count and number not in seen:
            result.append(number)
            seen.add(number)
    if not result:
        return None
    result.extend(number for number in range(1, count + 1) if number not in seen)
    return result


def _scores_with_reranked_topk(
    record: dict[str, Any],
    base_scores: dict[str, float],
    reranked_item_ids: list[str],
) -> dict[str, float]:
    result = {str(candidate["item_id"]): float(base_scores[str(candidate["item_id"])]) for candidate in record["candidates"]}
    reranked_set = set(reranked_item_ids)
    rest_scores = [score for item_id, score in result.items() if item_id not in reranked_set]
    floor = max(rest_scores) if rest_scores else max(result.values()) if result else 0.0
    for rank, item_id in enumerate(reranked_item_ids, start=1):
        result[item_id] = floor + 100.0 + (len(reranked_item_ids) - rank)
    return result


def _top_candidates(record: dict[str, Any], base_scores: dict[str, float], top_k: int) -> list[dict[str, str]]:
    candidates = sorted(
        record["candidates"],
        key=lambda candidate: (-base_scores[str(candidate["item_id"])], str(candidate["item_id"])),
    )[:top_k]
    return [
        {"item_id": str(candidate["item_id"]), "text": document_text(candidate)}
        for candidate in candidates
    ]


def _load_base_scores(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        scores.setdefault(request_id, {})[str(row["candidate_item_id"])] = float(row["score"])
    return scores


def _read_request_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _write_score(handle: Any, method_id: str, request_id: str, item_id: str, score: float) -> None:
    if not math.isfinite(score):
        raise ValueError(f"non-finite score for {request_id} {item_id}: {score}")
    handle.write(
        json.dumps(
            {
                "candidate_item_id": item_id,
                "method_id": method_id,
                "request_id": request_id,
                "score": score,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def _copy_config(config_path: str | Path | None, run_dir: Path) -> None:
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
