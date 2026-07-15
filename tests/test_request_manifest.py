from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.request_manifest import materialize_request_manifest


def test_request_manifest_hashes_visible_identity(tmp_path: Path) -> None:
    records = tmp_path / "records_dev.jsonl"
    records.write_text(json.dumps({
        "request_id": "r1", "query": "red shoes",
        "candidates": [{"item_id": "a"}, {"item_id": "b"}],
    }) + "\n", encoding="utf-8")
    result = materialize_request_manifest(
        [("dev", records)], tmp_path / "manifest.json", dataset_version="v1"
    )
    assert result["entries"][0]["request_id"] == "r1"
    assert result["entries"][0]["candidate_item_ids_sha256"]
