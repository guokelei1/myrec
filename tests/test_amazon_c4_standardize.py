from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.amazon_c4_standardize import build_standardized_amazon_c4
from myrec.data.contracts import audit_standardized_file


def _jsonl(path: Path, rows: list[dict], *, gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8") if gz else path.open(
        "w", encoding="utf-8"
    )
    with handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_current_amazon_materializer_is_causal_and_label_isolated(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "qid", "query", "item_id", "user_id", "ori_rating", "ori_review"
        ])
        writer.writeheader()
        for qid, split in enumerate(("train", "dev", "test"), start=1):
            writer.writerow({
                "qid": qid, "query": f"query {split}", "item_id": f"p{qid}",
                "user_id": f"u{qid}", "ori_rating": 5, "ori_review": "unused",
            })

    sampled = tmp_path / "sampled.jsonl"
    _jsonl(sampled, [
        {"item_id": f"p{i}", "category": "Cat", "metadata": f"query product {i}"}
        for i in range(1, 4)
    ] + [
        {"item_id": f"n{i}", "category": "Cat", "metadata": f"query negative {i}"}
        for i in range(1, 4)
    ])
    history_root = tmp_path / "history"
    for qid, split in enumerate(("train", "dev", "test"), start=1):
        _jsonl(history_root / "Cat" / f"{split}.jsonl", [{
            "user_id": f"u{qid}", "query": qid, "pos_product": f"p{qid}",
            "pos_product_category": "Cat",
            "grouped_purchase_history": {"Cat": [[f"h{qid}", qid * 10, True]]},
        }])
    metadata = tmp_path / "meta"
    _jsonl(metadata / "meta_Cat.jsonl.gz", [
        {"parent_asin": item, "title": f"title {item}", "store": "s",
         "main_category": "Cat", "categories": ["Cat"]}
        for item in [*(f"p{i}" for i in range(1, 4)),
                     *(f"n{i}" for i in range(1, 4)),
                     *(f"h{i}" for i in range(1, 4))]
    ], gz=True)

    output = tmp_path / "out"
    manifest = build_standardized_amazon_c4(
        c4_csv_path=csv_path, history_root=history_root,
        sampled_metadata_path=sampled, reviews_metadata_dir=metadata,
        fts_index_path=tmp_path / "items.sqlite", output_dir=output,
        report_path=tmp_path / "report.json", dataset_version="fixture_v1",
        max_history_len=1, bm25_top_k=3, retrieval_workers=1, metadata_workers=1,
    )
    assert manifest["overall_status"] == "passed"
    assert manifest["source_audit"]["target_in_history_rows"] == 0
    assert audit_standardized_file(output / "records_dev.jsonl", "dev")[
        "strict_nonrepeat_requests"
    ] == 1
    train_qrel = json.loads((output / "qrels_train.jsonl").read_text().splitlines()[0])
    assert train_qrel["purchased"] == ["p1"]
    dev = json.loads((output / "records_dev.jsonl").read_text().splitlines()[0])
    assert max(event["ts"] for event in dev["history"]) < dev["ts"]
    assert all("clicked" not in candidate for candidate in dev["candidates"])
