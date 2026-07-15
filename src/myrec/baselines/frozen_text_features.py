"""Frozen multilingual content features shared by HSTU and LLM-SRec.

Feature collection reads only visible standardized records and serializes only
query/item/context fields accepted by the representative sequence adapter. It
never reads qrels or candidate labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


def collect_visible_content_texts(record_paths: Iterable[str | Path]) -> list[str]:
    """Return unique adapter texts in deterministic hash order."""

    by_hash: dict[str, str] = {}
    for path in record_paths:
        for record in iter_jsonl(path):
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"request_id={record.get('request_id')}: empty query")
            texts = [f"query: {query}"]
            texts.extend(serialize_item_content(row) for row in record.get("history", []))
            texts.extend(
                serialize_item_content(row) for row in record.get("candidates", [])
            )
            for text in texts:
                digest = sha256_text(text)
                prior = by_hash.setdefault(digest, text)
                if prior != text:
                    raise RuntimeError("SHA-256 collision while collecting content text")
    if not by_hash:
        raise ValueError("no visible content text was collected")
    return [by_hash[digest] for digest in sorted(by_hash)]


def materialize_frozen_text_features(
    record_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    model_name_or_path: str,
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    device: str = "cuda:0",
    batch_size: int = 64,
    max_length: int = 128,
    dtype: str = "bfloat16",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Encode all unique visible texts with a frozen CrossEncoder backbone."""

    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported dtype={dtype}")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in record_paths]
    texts = collect_visible_content_texts(paths)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import CrossEncoder

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model = CrossEncoder(
        model_name_or_path,
        cache_folder=str(cache_folder),
        device=device,
        trust_remote_code=True,
        local_files_only=local_files_only,
        model_kwargs={"dtype": torch.float32},
    )
    classifier_model = model[0].model
    backbone = classifier_model.base_model
    tokenizer = model.tokenizer
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    hidden_size = int(classifier_model.config.hidden_size)
    vectors_path = output_dir / "vectors.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), hidden_size),
    )
    autocast_enabled = device.startswith("cuda") and dtype != "float32"
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch_dtype, enabled=autocast_enabled
            ):
                hidden = backbone(**tokens, return_dict=True).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            vectors[start : start + len(batch)] = hidden.cpu().numpy().astype(
                np.float16
            )
    vectors.flush()
    hashes = [sha256_text(text) for text in texts]
    index = {
        "schema_version": 1,
        "hash_to_row": {digest: row for row, digest in enumerate(hashes)},
    }
    write_json(output_dir / "index.json", index)
    metadata = {
        "schema_version": 1,
        "feature_contract": "frozen_transformer_cls_l2_v1",
        "model_name_or_path": model_name_or_path,
        "cache_folder": str(cache_folder),
        "local_files_only": local_files_only,
        "device": device,
        "inference_dtype": dtype,
        "storage_dtype": "float16",
        "hidden_size": hidden_size,
        "max_length": max_length,
        "batch_size": batch_size,
        "text_count": len(texts),
        "qrels_read": False,
        "record_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "vectors_sha256": sha256_file(vectors_path),
        "index_sha256": sha256_file(output_dir / "index.json"),
        "package_versions": {
            "numpy": np.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


class FrozenTextFeatureStore:
    """Read-only hash-addressed feature store usable from the HSTU environment."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        with (self.root / "index.json").open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        with (self.root / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.hash_to_row = {
            str(key): int(value) for key, value in index["hash_to_row"].items()
        }
        self.vectors = np.load(self.root / "vectors.npy", mmap_mode="r")
        if self.vectors.ndim != 2:
            raise ValueError("frozen text vectors must be a matrix")
        if self.vectors.shape[0] != len(self.hash_to_row):
            raise ValueError("frozen text vector/index row counts differ")

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def __call__(self, text: str) -> np.ndarray:
        digest = sha256_text(text)
        try:
            row = self.hash_to_row[digest]
        except KeyError as exc:
            raise KeyError(f"text is absent from frozen feature store: {digest}") from exc
        return np.asarray(self.vectors[row], dtype=np.float32)
