import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_b6o_hem_benchmark_split import build_split


def _write_gzip(path, lines):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


class B6oHemBenchmarkSplitTest(unittest.TestCase):
    def test_build_split_uses_benchmark_queries_qrels_and_train_review_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indexed = root / "indexed"
            benchmark = root / "benchmark"
            output = root / "out"
            indexed.mkdir()
            benchmark.mkdir()

            _write_gzip(indexed / "vocab.txt.gz", ["phone", "case", "charger", "toy"])
            _write_gzip(indexed / "product_query.txt.gz", ["0\t0 1;0\t2;", "0\t3;"])
            _write_gzip(indexed / "review_id.txt.gz", ["r_train", "r_test"])
            _write_gzip(indexed / "review_u_p.txt.gz", ["0 0", "1 1"])
            _write_gzip(indexed / "review_text.txt.gz", ["0 1", "2"])
            _write_gzip(indexed / "product.txt.gz", ["P0", "P1"])
            _write_gzip(indexed / "users.txt.gz", ["U0", "U1"])

            _write_gzip(benchmark / "query_text.txt.gz", ["phone case", "charger"])
            _write_gzip(benchmark / "train_review_id.txt.gz", ["r_train"])
            _write_gzip(benchmark / "train.qrels.gz", ["u_0 0 P0 1"])
            _write_gzip(benchmark / "test.qrels.gz", ["u_1 0 P1 1"])

            report = build_split(indexed, benchmark, output)

            self.assertEqual(_read_gzip(output / "query.txt.gz"), ["0 1", "2"])
            self.assertEqual(_read_gzip(output / "train.txt.gz"), ["0\t0\t0 1"])
            self.assertEqual(_read_gzip(output / "test.txt.gz"), ["1\t1\t2"])
            self.assertEqual(_read_gzip(output / "train_query_idx.txt.gz"), ["0", ""])
            self.assertEqual(_read_gzip(output / "test_query_idx.txt.gz"), ["1", ""])
            self.assertEqual((output / "train.qrels").read_text(encoding="utf-8"), "u_0 0 P0 1\n")
            self.assertEqual((output / "test.qrels").read_text(encoding="utf-8"), "u_1 0 P1 1\n")
            self.assertEqual(report["train_reviews"], 1)
            self.assertEqual(report["test_reviews"], 1)
            self.assertEqual(report["train_qids"], 1)
            self.assertEqual(report["test_qids"], 1)
            self.assertEqual(report["queries_with_missing_words"], 0)
            self.assertEqual(report["dropped_product_query_examples"][0]["query_text"], "toy")


if __name__ == "__main__":
    unittest.main()
