#!/usr/bin/env python
"""Build a HEM-compatible split from the Amazon Product Search benchmark files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexed-dir", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    return parser.parse_args()


def _read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def _write_gzip_lines(path: Path, lines: Iterable[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _copy_gzip_to_text(source: Path, target: Path) -> int:
    rows = 0
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as fin:
        with target.open("w", encoding="utf-8") as fout:
            for line in fin:
                rows += 1
                fout.write(line)
    return rows


def _qids_from_qrels(path: Path) -> set[int]:
    qids: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            request_id = line.split()[0]
            qids.add(int(request_id.rsplit("_", 1)[1]))
    return qids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(indexed_dir: Path, benchmark_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = _read_gzip_lines(indexed_dir / "vocab.txt.gz")
    vocab_map = {word: idx for idx, word in enumerate(vocab)}
    benchmark_queries = _read_gzip_lines(benchmark_dir / "query_text.txt.gz")
    benchmark_query_map = {query: idx for idx, query in enumerate(benchmark_queries)}

    missing_query_words: dict[str, list[str]] = {}
    indexed_query_lines: list[str] = []
    for query in benchmark_queries:
        missing = [word for word in query.split() if word not in vocab_map]
        if missing:
            missing_query_words[query] = missing
        indexed_query_lines.append(" ".join(str(vocab_map[word]) for word in query.split() if word in vocab_map))
    _write_gzip_lines(output_dir / "query.txt.gz", indexed_query_lines)

    train_qids = _qids_from_qrels(benchmark_dir / "train.qrels.gz")
    test_qids = _qids_from_qrels(benchmark_dir / "test.qrels.gz")

    dropped_product_queries = []
    train_query_idx_lines = []
    test_query_idx_lines = []
    for product_line_no, line in enumerate(_read_gzip_lines(indexed_dir / "product_query.txt.gz")):
        train_ids: list[str] = []
        test_ids: list[str] = []
        for entry in line.split(";"):
            if not entry:
                continue
            level, word_ids_text = entry.split("\t", 1)
            del level
            words = [vocab[int(idx)] for idx in word_ids_text.split() if idx]
            query_text = " ".join(words)
            qid = benchmark_query_map.get(query_text)
            if qid is None:
                if len(dropped_product_queries) < 20:
                    dropped_product_queries.append(
                        {"product_line": product_line_no, "query_text": query_text}
                    )
                continue
            if qid in train_qids:
                train_ids.append(str(qid))
            if qid in test_qids:
                test_ids.append(str(qid))
        train_query_idx_lines.append(" ".join(train_ids))
        test_query_idx_lines.append(" ".join(test_ids))
    _write_gzip_lines(output_dir / "train_query_idx.txt.gz", train_query_idx_lines)
    _write_gzip_lines(output_dir / "test_query_idx.txt.gz", test_query_idx_lines)

    train_original_ids = set(_read_gzip_lines(benchmark_dir / "train_review_id.txt.gz"))
    review_ids = _read_gzip_lines(indexed_dir / "review_id.txt.gz")
    review_u_p = _read_gzip_lines(indexed_dir / "review_u_p.txt.gz")
    review_text = _read_gzip_lines(indexed_dir / "review_text.txt.gz")
    if not (len(review_ids) == len(review_u_p) == len(review_text)):
        raise ValueError("indexed review files have inconsistent lengths")

    train_txt: list[str] = []
    test_txt: list[str] = []
    train_id: list[str] = []
    test_id: list[str] = []
    for rid, up, text in zip(review_ids, review_u_p, review_text):
        user_idx, product_idx = up.split(" ")
        row = f"{user_idx}\t{product_idx}\t{text}"
        id_row = f"{user_idx}\t{product_idx}\t{rid}"
        if rid in train_original_ids:
            train_txt.append(row)
            train_id.append(id_row)
        else:
            test_txt.append(row)
            test_id.append(id_row)
    _write_gzip_lines(output_dir / "train.txt.gz", train_txt)
    _write_gzip_lines(output_dir / "test.txt.gz", test_txt)
    _write_gzip_lines(output_dir / "train_id.txt.gz", train_id)
    _write_gzip_lines(output_dir / "test_id.txt.gz", test_id)

    train_qrels_rows = _copy_gzip_to_text(benchmark_dir / "train.qrels.gz", output_dir / "train.qrels")
    test_qrels_rows = _copy_gzip_to_text(benchmark_dir / "test.qrels.gz", output_dir / "test.qrels")

    # Keep the product/user/vocab files colocated for scripts that expect split-local paths.
    for filename in ["product.txt.gz", "users.txt.gz", "vocab.txt.gz"]:
        shutil.copyfile(indexed_dir / filename, output_dir / filename)

    report = {
        "benchmark_dir": str(benchmark_dir),
        "benchmark_query_count": len(benchmark_queries),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dropped_product_query_examples": dropped_product_queries,
        "indexed_dir": str(indexed_dir),
        "missing_query_words": missing_query_words,
        "output_dir": str(output_dir),
        "queries_with_missing_words": len(missing_query_words),
        "test_qids": len(test_qids),
        "test_qrels_rows": test_qrels_rows,
        "test_reviews": len(test_txt),
        "train_qids": len(train_qids),
        "train_qrels_rows": train_qrels_rows,
        "train_reviews": len(train_txt),
    }
    for filename in [
        "query.txt.gz",
        "train.txt.gz",
        "test.txt.gz",
        "train_id.txt.gz",
        "test_id.txt.gz",
        "train_query_idx.txt.gz",
        "test_query_idx.txt.gz",
        "train.qrels",
        "test.qrels",
    ]:
        report[f"{filename}_sha256"] = _sha256(output_dir / filename)
    return report


def main() -> int:
    args = parse_args()
    report = build_split(
        indexed_dir=Path(args.indexed_dir),
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
