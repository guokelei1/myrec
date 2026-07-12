from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from myrec.data.amazon_c4_standardize import (
    AmazonC4Request,
    _write_standardized_files,
    build_sampled_metadata_fts,
    build_standardized_amazon_c4,
    retrieve_bm25_candidates,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture(root: Path) -> dict[str, Path]:
    c4_csv = root / "test.csv"
    with c4_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "query", "item_id", "user_id", "ori_rating", "ori_review"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"qid": 1, "query": "red running shoes", "item_id": "p1", "user_id": "u1"},
                {"qid": 2, "query": "quiet coffee grinder", "item_id": "p2", "user_id": "u2"},
                {"qid": 3, "query": "warm winter gloves", "item_id": "p3", "user_id": "u3"},
            ]
        )
    sampled = root / "sampled.jsonl"
    _write_jsonl(
        sampled,
        [
            {"item_id": "p1", "category": "Fashion", "metadata": "red running shoes light"},
            {"item_id": "p2", "category": "Home", "metadata": "quiet coffee grinder steel"},
            {"item_id": "p3", "category": "Fashion", "metadata": "warm winter gloves wool"},
            {"item_id": "n1", "category": "Fashion", "metadata": "red shoes casual"},
            {"item_id": "n2", "category": "Home", "metadata": "coffee grinder electric"},
            {"item_id": "n3", "category": "Fashion", "metadata": "winter gloves black"},
        ],
    )
    history_root = root / "history"
    split_rows = {
        "train": {
            "user_id": "u1",
            "query": 1,
            "pos_product": "p1",
            "pos_product_category": "Fashion",
            "grouped_purchase_history": {
                "Home": [["h2", 20, False]],
                "Fashion": [["h1", 10, True], ["h3", 30, True]],
            },
        },
        "dev": {
            "user_id": "u2",
            "query": 2,
            "pos_product": "p2",
            "pos_product_category": "Home",
            "grouped_purchase_history": {"Home": [["h2", 20, True]]},
        },
        "test": {
            "user_id": "u3",
            "query": 3,
            "pos_product": "p3",
            "pos_product_category": "Fashion",
            "grouped_purchase_history": {"Fashion": [["h3", 30, True]]},
        },
    }
    for split, row in split_rows.items():
        _write_jsonl(history_root / row["pos_product_category"] / f"{split}.jsonl", [row])
    # The loader requires a file for each split, but empty category files are valid.
    _write_jsonl(history_root / "Fashion" / "dev.jsonl", [])
    _write_jsonl(history_root / "Home" / "train.jsonl", [])
    _write_jsonl(history_root / "Home" / "test.jsonl", [])

    metadata_dir = root / "reviews_meta"
    meta_rows = [
        {
            "parent_asin": item_id,
            "title": f"title {item_id}",
            "store": "store",
            "main_category": category,
            "categories": [category],
        }
        for item_id, category in [
            ("p1", "Fashion"),
            ("p2", "Home"),
            ("p3", "Fashion"),
            ("n1", "Fashion"),
            ("n2", "Home"),
            ("n3", "Fashion"),
            ("h1", "Fashion"),
            ("h2", "Home"),
            ("h3", "Fashion"),
        ]
    ]
    # Put h2 in the wrong archive to exercise the official cross-category
    # fallback used for upstream `Unknown` mappings.
    fashion_rows = [row for row in meta_rows if row["main_category"] == "Fashion"]
    home_rows = [row for row in meta_rows if row["main_category"] == "Home" and row["parent_asin"] != "h2"]
    fashion_rows.append(next(row for row in meta_rows if row["parent_asin"] == "h2"))
    _write_gzip_jsonl(metadata_dir / "meta_Fashion.jsonl.gz", fashion_rows)
    _write_gzip_jsonl(metadata_dir / "meta_Home.jsonl.gz", home_rows)
    return {
        "c4_csv": c4_csv,
        "sampled": sampled,
        "history_root": history_root,
        "metadata_dir": metadata_dir,
    }


def test_fts_bm25_is_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = tmp_path / "items.sqlite"
    manifest = build_sampled_metadata_fts(paths["sampled"], index)
    assert manifest["rows"] == 6
    import sqlite3

    connection = sqlite3.connect(index)
    try:
        first = retrieve_bm25_candidates(connection, "red running shoes", top_k=3)
        second = retrieve_bm25_candidates(connection, "red running shoes", top_k=3)
    finally:
        connection.close()
    assert first == second
    assert first[0]["item_id"] == "p1"


def test_standardizer_isolates_labels_and_sorts_history(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "standardized"
    report = tmp_path / "report.json"
    manifest = build_standardized_amazon_c4(
        c4_csv_path=paths["c4_csv"],
        history_root=paths["history_root"],
        sampled_metadata_path=paths["sampled"],
        reviews_metadata_dir=paths["metadata_dir"],
        fts_index_path=tmp_path / "items.sqlite",
        output_dir=output,
        report_path=report,
        max_history_len=2,
        bm25_top_k=3,
        metadata_workers=2,
    )
    assert manifest["overall_status"] == "passed"
    assert manifest["metadata_scan"]["cross_category_fallback"]["triggered"] is True
    assert manifest["metadata_scan"]["cross_category_fallback"]["matched_items"] == 1
    train = json.loads((output / "records_train.jsonl").read_text().splitlines()[0])
    train_blind = json.loads((output / "records_train_blind.jsonl").read_text().splitlines()[0])
    dev = json.loads((output / "records_dev.jsonl").read_text().splitlines()[0])
    test = json.loads((output / "records_test.jsonl").read_text().splitlines()[0])
    assert [event["item_id"] for event in train["history"]] == ["h2", "h3"]
    assert train["ts"] == 31
    assert sum(candidate["purchased"] for candidate in train["candidates"]) == 1
    assert all(
        "clicked" not in candidate and "purchased" not in candidate
        for candidate in train_blind["candidates"]
    )
    assert all("clicked" not in candidate and "purchased" not in candidate for candidate in dev["candidates"])
    assert all("clicked" not in candidate and "purchased" not in candidate for candidate in test["candidates"])
    dev_qrel = json.loads((output / "qrels_dev.jsonl").read_text().splitlines()[0])
    assert dev_qrel["purchased"] == ["p2"]
    candidate_manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert len(candidate_manifest["entries"]) == 3
    assert all(len(entry["candidate_item_ids"]) >= 3 for entry in candidate_manifest["entries"])

    cached_output = tmp_path / "standardized_cached"
    cached_report = tmp_path / "cached_report.json"
    cached = build_standardized_amazon_c4(
        c4_csv_path=paths["c4_csv"],
        history_root=paths["history_root"],
        sampled_metadata_path=paths["sampled"],
        reviews_metadata_dir=paths["metadata_dir"],
        fts_index_path=tmp_path / "items.sqlite",
        output_dir=cached_output,
        report_path=cached_report,
        max_history_len=2,
        bm25_top_k=3,
        metadata_workers=2,
        candidate_cache_path=output / "candidate_manifest.json",
        candidate_cache_report_path=report,
    )
    assert cached["overall_status"] == "passed"
    assert cached["protocol"]["candidate_cache"]["validated_requests"] == 3


def test_missing_history_text_is_masked_and_audited(tmp_path: Path) -> None:
    output = tmp_path / "standardized"
    output.mkdir()
    request = AmazonC4Request(
        request_id="amazon_c4_train_1",
        split="train",
        qid=1,
        query="red shoes",
        user_id="u1",
        positive_item_id="p1",
        positive_category="Fashion",
        history_events=[
            {"item_id": "h1", "ts": 1, "verified_purchase": True},
            {"item_id": "missing", "ts": 2, "verified_purchase": False},
        ],
        candidate_rows=[{"item_id": "p1"}, {"item_id": "n1"}],
    )
    item_map = {
        item_id: {
            "item_id": item_id,
            "title": f"title {item_id}",
            "brand": "",
            "seller": "",
            "cat": ["Fashion", "", ""],
        }
        for item_id in ("h1", "p1", "n1")
    }
    stats = _write_standardized_files(output, [request], item_map, bm25_top_k=2)
    record = json.loads((output / "records_train.jsonl").read_text().splitlines()[0])

    assert [event["item_id"] for event in record["history"]] == ["h1"]
    assert record["masks"]["history_source_events"] == 2
    assert record["masks"]["history_missing_text_events_dropped"] == 1
    assert stats["source_history_event_text_coverage"] == 0.5
    assert stats["history_event_text_coverage"] == 1.0
    assert stats["missing_history_event_drop_fraction"] == 0.5
