"""Frozen SASRec teacher representations for the independent LLM-SRec baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from myrec.baselines.frozen_text_features import FrozenTextFeatureStore
from myrec.baselines.hstu_pps_adapter import (
    HSTUPPSRanker,
    collate_sequence_requests,
)
from myrec.baselines.representative_sequence_adapter import (
    SequenceRequest,
    TrainVocabulary,
    build_sequence_request,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


def teacher_item_key(raw_item_id: str, content_text: str) -> str:
    return sha256_text(f"{raw_item_id}\0{content_text}")


class FrozenSequenceTeacherStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        with (self.root / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        with (self.root / "index.json").open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.item_to_row = {str(k): int(v) for k, v in index["item_to_row"].items()}
        self.request_to_row = {
            str(k): int(v) for k, v in index["train_request_to_row"].items()
        }
        self.item_vectors = np.load(self.root / "item_vectors.npy", mmap_mode="r")
        self.train_user_vectors = np.load(
            self.root / "train_user_vectors.npy", mmap_mode="r"
        )

    @property
    def dimension(self) -> int:
        return int(self.item_vectors.shape[1])

    def item(self, raw_item_id: str, content_text: str) -> np.ndarray:
        row = self.item_to_row[teacher_item_key(raw_item_id, content_text)]
        return np.asarray(self.item_vectors[row], dtype=np.float32)

    def train_user(self, request_id: str) -> np.ndarray:
        return np.asarray(
            self.train_user_vectors[self.request_to_row[str(request_id)]],
            dtype=np.float32,
        )


def materialize_sequence_teacher_features(
    standardized_dir: str | Path,
    feature_store_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    dev_assignment_paths: Sequence[str | Path],
    evaluation_split: str = "dev",
    device: str = "cuda:0",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Materialize frozen teacher items and train user vectors without dev qrels."""

    standardized_dir = Path(standardized_dir)
    if evaluation_split not in {"dev", "internal", "confirmation"}:
        raise ValueError(
            "teacher materialization supports dev, internal, or confirmation only"
        )
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if checkpoint["architecture"] != "sasrec" or checkpoint["input_mode"] != "full":
        raise ValueError("LLM-SRec teacher must be a FULL SASRec checkpoint")
    with (checkpoint_dir / "vocabulary.json").open("r", encoding="utf-8") as handle:
        vocabulary = TrainVocabulary.from_dict(json.load(handle))
    feature_store = FrozenTextFeatureStore(feature_store_dir)
    model = HSTUPPSRanker(**checkpoint["model_config"]).to(device)
    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    history_budget = int(checkpoint["model_config"]["max_sequence_length"]) - 1

    train_requests = [
        build_sequence_request(row, vocabulary, history_budget=history_budget)
        for row in iter_jsonl(standardized_dir / "records_train.jsonl")
    ]
    evaluation_records_path = standardized_dir / f"records_{evaluation_split}.jsonl"
    if not evaluation_records_path.exists():
        raise FileNotFoundError(
            f"missing standardized records for split={evaluation_split}: "
            f"{evaluation_records_path}"
        )
    evaluation_visible = {
        str(row["request_id"]): row
        for row in iter_jsonl(evaluation_records_path)
    }
    dev_requests: list[SequenceRequest] = []
    assignment_hashes = []
    for assignment_path in map(Path, dev_assignment_paths):
        assignment_hashes.append(
            {"path": str(assignment_path), "sha256": sha256_file(assignment_path)}
        )
        for assignment in iter_jsonl(assignment_path):
            request_id = str(assignment["request_id"])
            record = dict(
                evaluation_visible[request_id],
                history=assignment.get("history", []),
            )
            dev_requests.append(
                build_sequence_request(record, vocabulary, history_budget=history_budget)
            )

    items: dict[str, tuple[int, str]] = {}
    for request in [*train_requests, *dev_requests]:
        for raw_id, item_id, text in zip(
            request.past_raw_item_ids,
            request.past_item_ids[: request.retained_history_count],
            request.past_content_texts[: request.retained_history_count],
        ):
            items.setdefault(teacher_item_key(raw_id, text), (item_id, text))
        for candidate in request.candidates:
            items.setdefault(
                teacher_item_key(candidate.raw_item_id, candidate.content_text),
                (candidate.item_id, candidate.content_text),
            )
    item_keys = sorted(items)
    item_vectors = np.lib.format.open_memmap(
        output_dir / "item_vectors.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(item_keys), int(checkpoint["model_config"]["embedding_dim"])),
    )
    with torch.inference_mode():
        for start in range(0, len(item_keys), batch_size):
            keys = item_keys[start : start + batch_size]
            ids = torch.tensor([items[key][0] for key in keys], device=device)
            content = torch.stack(
                [torch.from_numpy(feature_store(items[key][1])) for key in keys]
            ).to(device)
            values = model.sequence_core.get_item_embeddings(ids)
            values = F.normalize(values + model.content_projection(content), dim=-1)
            item_vectors[start : start + len(keys)] = values.cpu().numpy().astype(np.float16)
    item_vectors.flush()

    train_user_vectors = np.lib.format.open_memmap(
        output_dir / "train_user_vectors.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(train_requests), int(checkpoint["model_config"]["embedding_dim"])),
    )
    with torch.inference_mode():
        for start in range(0, len(train_requests), batch_size):
            requests = train_requests[start : start + batch_size]
            batch = collate_sequence_requests(
                requests,
                feature_store,
                content_dim=feature_store.dimension,
                max_sequence_length=checkpoint["model_config"]["max_sequence_length"],
            ).to(device)
            values = F.normalize(model.encode_sequence(batch), dim=-1)
            train_user_vectors[start : start + len(requests)] = (
                values.cpu().numpy().astype(np.float16)
            )
    train_user_vectors.flush()
    index = {
        "schema_version": 1,
        "item_to_row": {key: row for row, key in enumerate(item_keys)},
        "train_request_to_row": {
            request.request_id: row for row, request in enumerate(train_requests)
        },
    }
    write_json(output_dir / "index.json", index)
    metadata = {
        "schema_version": 1,
        "feature_contract": "frozen_sasrec_pps_teacher_v1",
        "dimension": int(checkpoint["model_config"]["embedding_dim"]),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_weights_sha256": checkpoint["weights_sha256"],
        "item_count": len(item_keys),
        "train_request_count": len(train_requests),
        "qrels_read": False,
        "dev_qrels_read": False,
        "records_train_sha256": sha256_file(standardized_dir / "records_train.jsonl"),
        "evaluation_split": evaluation_split,
        "records_evaluation_sha256": sha256_file(evaluation_records_path),
        "dev_assignments": assignment_hashes,
        "item_vectors_sha256": sha256_file(output_dir / "item_vectors.npy"),
        "train_user_vectors_sha256": sha256_file(
            output_dir / "train_user_vectors.npy"
        ),
        "index_sha256": sha256_file(output_dir / "index.json"),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
