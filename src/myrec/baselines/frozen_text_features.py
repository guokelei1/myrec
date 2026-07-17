"""Frozen multilingual content features shared by HSTU and LLM-SRec.

Feature collection reads only visible standardized records and serializes only
query/item/context fields accepted by the representative sequence adapter. It
never reads qrels or candidate labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from myrec.baselines.representative_sequence_adapter import serialize_item_content
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


ENCODER_FINGERPRINT_SCHEMA_VERSION = 1
STORE_FINGERPRINT_SCHEMA_VERSION = 1


def serialize_item_semantic_content(row: Mapping[str, Any]) -> str:
    """Serialize canonical item semantics without event/query/user context."""

    return serialize_item_content(
        {key: row[key] for key in ("title", "brand", "cat") if key in row}
    )


def finalize_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical self-authenticating fingerprint object."""

    if "sha256" in payload:
        raise ValueError("fingerprint payload must not already contain sha256")
    frozen = json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    frozen["sha256"] = sha256_text(
        json.dumps(
            frozen,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return frozen


def build_store_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable identity of one concrete vector/index store."""

    encoder = metadata.get("encoder_fingerprint")
    if not isinstance(encoder, dict) or not encoder.get("sha256"):
        raise ValueError("store metadata lacks an encoder fingerprint")
    record_hashes = sorted(
        str(row["sha256"])
        for row in metadata.get("record_files", [])
        if isinstance(row, dict) and row.get("sha256")
    )
    return finalize_fingerprint(
        {
            "schema_version": STORE_FINGERPRINT_SCHEMA_VERSION,
            "feature_contract": metadata.get("feature_contract"),
            "visible_text_contract": metadata.get("visible_text_contract"),
            "encoder_fingerprint_sha256": encoder["sha256"],
            "index_sha256": metadata.get("index_sha256"),
            "vectors_sha256": metadata.get("vectors_sha256"),
            "record_sha256s": record_hashes,
            "text_count": metadata.get("text_count"),
            "hidden_size": metadata.get("hidden_size"),
            "storage_dtype": metadata.get("storage_dtype"),
            "base_store_fingerprint_sha256": metadata.get(
                "base_store_fingerprint_sha256"
            ),
            "base_store_metadata_sha256": metadata.get(
                "base_store_metadata_sha256"
            ),
            "reused_text_rows": metadata.get("reused_text_rows"),
            "new_text_rows": metadata.get("new_text_rows"),
            "store_ancestry": metadata.get("store_ancestry", []),
        }
    )


def build_local_encoder_fingerprint(
    *,
    model_name_or_path: str,
    cache_folder: str | Path,
    max_length: int,
    inference_dtype: str,
    batch_size: int,
    device_identity: Mapping[str, Any],
    package_versions: Mapping[str, str],
    local_files_only: bool,
) -> dict[str, Any]:
    """Fingerprint the exact local encoder snapshot and encoding recipe."""

    snapshot = _resolve_local_snapshot(
        model_name_or_path,
        cache_folder=cache_folder,
        local_files_only=local_files_only,
    )
    artifact_rows = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    if not artifact_rows:
        raise ValueError(f"encoder snapshot contains no files: {snapshot}")
    revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    return finalize_fingerprint(
        {
            "schema_version": ENCODER_FINGERPRINT_SCHEMA_VERSION,
            "model_name_or_path": model_name_or_path,
            "resolved_revision": revision,
            "artifact_files": artifact_rows,
            "encoding_recipe": {
                "backbone": "sequence_classification_base_model",
                "pooling": "last_hidden_state_token_0_cls",
                "l2_normalize": True,
                "max_length": int(max_length),
                "batch_size": int(batch_size),
                "requested_inference_dtype": inference_dtype,
                "effective_compute_dtype": (
                    inference_dtype
                    if device_identity.get("autocast_enabled")
                    else "float32"
                ),
                "model_load_dtype": "float32",
                "tokenizer_padding": True,
                "tokenizer_truncation": True,
                "device_identity": dict(device_identity),
                "package_versions": dict(package_versions),
            },
        }
    )


def collect_visible_content_texts(record_paths: Iterable[str | Path]) -> list[str]:
    """Return unique adapter texts in deterministic hash order."""

    by_hash: dict[str, str] = {}
    for path in record_paths:
        for record in iter_jsonl(path):
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"request_id={record.get('request_id')}: empty query")
            texts = [f"query: {query}"]
            history = record.get("history", [])
            texts.extend(serialize_item_content(row) for row in history)
            # W0 uses canonical item-only vectors to choose/encode semantic
            # replacements while preserving contextual history vectors for its
            # supervised ranking view.
            texts.extend(serialize_item_semantic_content(row) for row in history)
            texts.extend(
                serialize_item_semantic_content(row)
                for row in record.get("candidates", [])
            )
            for text in texts:
                digest = sha256_text(text)
                prior = by_hash.setdefault(digest, text)
                if prior != text:
                    raise RuntimeError("SHA-256 collision while collecting content text")
    if not by_hash:
        raise ValueError("no visible content text was collected")
    return [by_hash[digest] for digest in sorted(by_hash)]


def copy_base_feature_rows_bitwise(
    base_store: "FrozenTextFeatureStore",
    target_hashes: Sequence[str],
    target_vectors: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> int:
    """Copy every shared float16 row and verify the target bytes immediately."""

    if chunk_size <= 0:
        raise ValueError("feature-row copy chunk_size must be positive")
    if target_vectors.ndim != 2 or target_vectors.shape[0] != len(target_hashes):
        raise ValueError("target feature vector/hash shapes differ")
    if target_vectors.shape[1] != base_store.dimension:
        raise ValueError("target/base feature dimensions differ")
    if target_vectors.dtype != base_store.vectors.dtype:
        raise ValueError("target/base feature storage dtypes differ")
    copied = 0
    for start in range(0, len(target_hashes), chunk_size):
        target_rows = []
        source_rows = []
        for target_row in range(start, min(start + chunk_size, len(target_hashes))):
            source_row = base_store.hash_to_row.get(str(target_hashes[target_row]))
            if source_row is not None:
                target_rows.append(target_row)
                source_rows.append(source_row)
        if not target_rows:
            continue
        target_index = np.asarray(target_rows)
        source = np.asarray(
            base_store.vectors[np.asarray(source_rows)], dtype=target_vectors.dtype
        )
        target_vectors[target_index] = source
        if not np.array_equal(np.asarray(target_vectors[target_index]), source):
            raise RuntimeError("base frozen feature rows were not copied bitwise")
        copied += len(target_rows)
    return copied


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
    base_store: str | Path | None = None,
) -> dict[str, Any]:
    """Encode visible texts, bitwise-reusing a verified base store when supplied."""

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
    hashes = [sha256_text(text) for text in texts]
    record_files = [
        {"path": str(path), "sha256": sha256_file(path)} for path in paths
    ]

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import CrossEncoder

    device_object = torch.device(device)
    autocast_enabled = device_object.type == "cuda" and dtype != "float32"
    device_identity: dict[str, Any] = {
        "device_type": device_object.type,
        "autocast_enabled": autocast_enabled,
    }
    if device_object.type == "cuda" and torch.cuda.is_available():
        device_index = device_object.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_identity.update(
            {
                "device_name": torch.cuda.get_device_name(device_index),
                "compute_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
        )
    package_versions = {
        "numpy": np.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    encoder_fingerprint = build_local_encoder_fingerprint(
        model_name_or_path=model_name_or_path,
        cache_folder=cache_folder,
        max_length=max_length,
        inference_dtype=dtype,
        batch_size=batch_size,
        device_identity=device_identity,
        package_versions=package_versions,
        local_files_only=local_files_only,
    )

    base: FrozenTextFeatureStore | None = None
    base_metadata_sha256: str | None = None
    ancestry: list[dict[str, Any]] = []
    if base_store is not None:
        base = FrozenTextFeatureStore(base_store, require_fingerprints=True)
        if base.metadata.get("qrels_read") is not False:
            raise ValueError("base frozen feature store crossed the qrels boundary")
        if base.metadata.get("feature_contract") != "frozen_transformer_cls_l2_v1":
            raise ValueError("base frozen feature contract differs")
        if (
            base.metadata.get("visible_text_contract")
            != "query_context_and_canonical_item_semantics_v2"
        ):
            raise ValueError("base frozen visible-text contract differs")
        if base.metadata.get("encoder_fingerprint") != encoder_fingerprint:
            raise ValueError("base frozen encoder/effective-compute identity differs")
        if base.vectors.dtype != np.dtype("float16"):
            raise ValueError("base frozen feature storage dtype is not float16")
        missing_hashes = set(base.hash_to_row) - set(hashes)
        if missing_hashes:
            raise ValueError(
                "requested feature store is not a text superset of the base store"
            )
        current_record_hashes = {row["sha256"] for row in record_files}
        base_record_hashes = {
            str(row.get("sha256"))
            for row in base.metadata.get("record_files", [])
            if isinstance(row, dict) and row.get("sha256")
        }
        if not base_record_hashes <= current_record_hashes:
            raise ValueError(
                "requested feature store records are not a superset of base records"
            )
        base_metadata_sha256 = sha256_file(base.root / "metadata.json")
        ancestry = [
            {
                "store_fingerprint_sha256": base.store_fingerprint_sha256,
                "metadata_sha256": base_metadata_sha256,
                "relation": "direct_bitwise_row_reuse",
                "text_count": int(base.metadata["text_count"]),
            }
        ]
        seen_ancestors = {base.store_fingerprint_sha256}
        for entry in base.metadata.get("store_ancestry", []):
            fingerprint = str(entry.get("store_fingerprint_sha256", ""))
            if fingerprint and fingerprint not in seen_ancestors:
                ancestry.append(json.loads(json.dumps(entry, sort_keys=True)))
                seen_ancestors.add(fingerprint)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    new_rows = [
        row for row, digest in enumerate(hashes) if base is None or digest not in base.hash_to_row
    ]
    backbone = None
    tokenizer = None
    if new_rows:
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
        if base is not None and base.dimension != hidden_size:
            raise ValueError("base frozen feature dimension differs from encoder")
    elif base is not None:
        hidden_size = base.dimension
    else:
        raise AssertionError("a non-empty text collection unexpectedly has no rows")
    vectors_path = output_dir / "vectors.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), hidden_size),
    )
    reused_rows = len(texts) - len(new_rows)
    if base is not None and reused_rows:
        copied_rows = copy_base_feature_rows_bitwise(base, hashes, vectors)
        if copied_rows != reused_rows:
            raise AssertionError("base frozen feature reuse count differs")
    if new_rows:
        assert backbone is not None and tokenizer is not None
        with torch.inference_mode():
            for start in range(0, len(new_rows), batch_size):
                target_rows = new_rows[start : start + batch_size]
                batch = [texts[row] for row in target_rows]
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.autocast(
                    device_type=device_object.type,
                    dtype=torch_dtype,
                    enabled=autocast_enabled,
                ):
                    hidden = backbone(
                        **tokens, return_dict=True
                    ).last_hidden_state[:, 0]
                    hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
                vectors[np.asarray(target_rows)] = hidden.cpu().numpy().astype(
                    np.float16
                )
    vectors.flush()
    index = {
        "schema_version": 1,
        "hash_to_row": {digest: row for row, digest in enumerate(hashes)},
    }
    write_json(output_dir / "index.json", index)
    metadata = {
        "schema_version": 1,
        "feature_contract": "frozen_transformer_cls_l2_v1",
        "visible_text_contract": "query_context_and_canonical_item_semantics_v2",
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
        "base_store_path": str(base.root) if base is not None else None,
        "base_store_fingerprint_sha256": (
            base.store_fingerprint_sha256 if base is not None else None
        ),
        "base_store_metadata_sha256": base_metadata_sha256,
        "reused_text_rows": reused_rows,
        "new_text_rows": len(new_rows),
        "store_ancestry": ancestry,
        "qrels_read": False,
        "materializer_implementation_sha256": sha256_file(Path(__file__)),
        "encoder_fingerprint": encoder_fingerprint,
        "record_files": record_files,
        "vectors_sha256": sha256_file(vectors_path),
        "index_sha256": sha256_file(output_dir / "index.json"),
        "package_versions": package_versions,
    }
    metadata["store_fingerprint"] = build_store_fingerprint(metadata)
    write_json(output_dir / "metadata.json", metadata)
    return metadata


class FrozenTextFeatureStore:
    """Read-only hash-addressed feature store usable from the HSTU environment."""

    def __init__(
        self, root: str | Path, *, require_fingerprints: bool = False
    ) -> None:
        self.root = Path(root)
        index_path = self.root / "index.json"
        metadata_path = self.root / "metadata.json"
        vectors_path = self.root / "vectors.npy"
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if require_fingerprints:
            for label, declared in (
                ("index", self.metadata.get("index_sha256")),
                ("vectors", self.metadata.get("vectors_sha256")),
            ):
                if not isinstance(declared, str) or len(declared) != 64:
                    raise ValueError(
                        f"frozen text {label} lacks a required declared SHA-256"
                    )
        _verify_declared_file_hash(
            index_path, self.metadata.get("index_sha256"), label="index"
        )
        _verify_declared_file_hash(
            vectors_path, self.metadata.get("vectors_sha256"), label="vectors"
        )
        self.hash_to_row = {
            str(key): int(value) for key, value in index["hash_to_row"].items()
        }
        self.vectors = np.load(vectors_path, mmap_mode="r")
        if self.vectors.ndim != 2:
            raise ValueError("frozen text vectors must be a matrix")
        if self.vectors.shape[0] != len(self.hash_to_row):
            raise ValueError("frozen text vector/index row counts differ")
        if sorted(self.hash_to_row.values()) != list(range(len(self.hash_to_row))):
            raise ValueError("frozen text index rows are not a complete permutation")
        text_count = self.metadata.get("text_count")
        if text_count is not None and int(text_count) != self.vectors.shape[0]:
            raise ValueError("frozen text metadata text_count differs from vectors")
        hidden_size = self.metadata.get("hidden_size")
        if hidden_size is not None and int(hidden_size) != self.vectors.shape[1]:
            raise ValueError("frozen text metadata hidden_size differs from vectors")
        storage_dtype = self.metadata.get("storage_dtype")
        if storage_dtype is not None and np.dtype(storage_dtype) != self.vectors.dtype:
            raise ValueError("frozen text metadata storage dtype differs from vectors")
        encoder = self.metadata.get("encoder_fingerprint")
        store = self.metadata.get("store_fingerprint")
        if require_fingerprints and (not isinstance(encoder, dict) or not isinstance(store, dict)):
            raise ValueError("frozen text store lacks required fingerprints")
        if isinstance(encoder, dict):
            _verify_fingerprint(encoder, label="encoder")
        if isinstance(store, dict):
            _verify_fingerprint(store, label="store")
            if store != build_store_fingerprint(self.metadata):
                raise ValueError("frozen text store fingerprint payload differs")
        _validate_store_ancestry(self.metadata)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def encoder_fingerprint_sha256(self) -> str | None:
        value = self.metadata.get("encoder_fingerprint", {})
        return str(value["sha256"]) if isinstance(value, dict) and value.get("sha256") else None

    @property
    def store_fingerprint_sha256(self) -> str | None:
        value = self.metadata.get("store_fingerprint", {})
        return str(value["sha256"]) if isinstance(value, dict) and value.get("sha256") else None

    def __call__(self, text: str) -> np.ndarray:
        digest = sha256_text(text)
        try:
            row = self.hash_to_row[digest]
        except KeyError as exc:
            raise KeyError(f"text is absent from frozen feature store: {digest}") from exc
        return np.asarray(self.vectors[row], dtype=np.float32)


def _resolve_local_snapshot(
    model_name_or_path: str,
    *,
    cache_folder: str | Path,
    local_files_only: bool,
) -> Path:
    local = Path(model_name_or_path).expanduser()
    if local.exists():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name_or_path,
            cache_dir=str(cache_folder),
            local_files_only=local_files_only,
        )
    ).resolve()


def _verify_declared_file_hash(path: Path, declared: Any, *, label: str) -> None:
    if declared is not None and sha256_file(path) != str(declared):
        raise ValueError(f"frozen text {label} hash differs from metadata")


def _verify_fingerprint(value: dict[str, Any], *, label: str) -> None:
    declared = value.get("sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"frozen text {label} fingerprint lacks SHA-256")
    payload = {key: row for key, row in value.items() if key != "sha256"}
    if finalize_fingerprint(payload)["sha256"] != declared:
        raise ValueError(f"frozen text {label} fingerprint is not self-consistent")


def _validate_store_ancestry(metadata: Mapping[str, Any]) -> None:
    ancestry = metadata.get("store_ancestry", [])
    if not isinstance(ancestry, list):
        raise ValueError("frozen text store ancestry must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(ancestry):
        if not isinstance(entry, Mapping):
            raise ValueError(f"frozen text store ancestor {index} is not an object")
        fingerprint = entry.get("store_fingerprint_sha256")
        metadata_sha = entry.get("metadata_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"frozen text store ancestor {index} lacks a fingerprint")
        if not isinstance(metadata_sha, str) or len(metadata_sha) != 64:
            raise ValueError(f"frozen text store ancestor {index} lacks a metadata SHA")
        if fingerprint in seen:
            raise ValueError("frozen text store ancestry contains a duplicate fingerprint")
        seen.add(fingerprint)
        if not str(entry.get("relation", "")).strip():
            raise ValueError(f"frozen text store ancestor {index} lacks a relation")
        if int(entry.get("text_count", -1)) <= 0:
            raise ValueError(f"frozen text store ancestor {index} has invalid text_count")

    base_fingerprint = metadata.get("base_store_fingerprint_sha256")
    base_metadata_sha = metadata.get("base_store_metadata_sha256")
    if base_fingerprint is None and base_metadata_sha is None:
        if ancestry:
            raise ValueError("frozen text store has ancestry without a direct base")
    else:
        if not isinstance(base_fingerprint, str) or len(base_fingerprint) != 64:
            raise ValueError("frozen text direct base fingerprint is invalid")
        if not isinstance(base_metadata_sha, str) or len(base_metadata_sha) != 64:
            raise ValueError("frozen text direct base metadata SHA is invalid")
        if not ancestry:
            raise ValueError("frozen text derived store has no ancestry")
        direct = ancestry[0]
        if (
            direct.get("store_fingerprint_sha256") != base_fingerprint
            or direct.get("metadata_sha256") != base_metadata_sha
            or direct.get("relation") != "direct_bitwise_row_reuse"
        ):
            raise ValueError("frozen text direct base differs from ancestry")

    reused = metadata.get("reused_text_rows")
    new = metadata.get("new_text_rows")
    if reused is not None or new is not None:
        if not isinstance(reused, int) or reused < 0:
            raise ValueError("frozen text reused_text_rows is invalid")
        if not isinstance(new, int) or new < 0:
            raise ValueError("frozen text new_text_rows is invalid")
        if reused + new != int(metadata.get("text_count", -1)):
            raise ValueError("frozen text reused/new row counts differ from text_count")
        if base_fingerprint is None and reused != 0:
            raise ValueError("frozen text root store claims reused rows")
        if base_fingerprint is not None and reused != int(ancestry[0]["text_count"]):
            raise ValueError("frozen text reused rows differ from direct base text_count")
