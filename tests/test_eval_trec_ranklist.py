import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_trec_ranklist import evaluate


class TrecRanklistEvalTest(unittest.TestCase):
    def test_external_ranklist_metrics_are_hand_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qrels = root / "qrels.txt"
            ranklist = root / "ranklist.txt"
            qrels.write_text(
                "\n".join(
                    [
                        "q1 0 d1 1",
                        "q1 0 d3 1",
                        "q2 0 d4 1",
                        "q3 0 d6 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ranklist.write_text(
                "\n".join(
                    [
                        "q1 Q0 d2 1 3.0 run",
                        "q1 Q0 d1 2 2.0 run",
                        "q1 Q0 d3 3 1.0 run",
                        "q2 Q0 d5 1 2.0 run",
                        "q2 Q0 d4 2 1.0 run",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate(qrels, ranklist, map_cutoff=100, mrr_cutoff=100, ndcg_cutoff=10)

            expected_q1_ap = ((1 / 2) + (2 / 3)) / 2
            expected_q2_ap = 1 / 2
            expected_q1_ndcg = (
                (1 / math.log2(3)) + (1 / math.log2(4))
            ) / (1 + (1 / math.log2(3)))
            expected_q2_ndcg = 1 / math.log2(3)
            self.assertAlmostEqual(result["map@100"], (expected_q1_ap + expected_q2_ap) / 3)
            self.assertAlmostEqual(result["mrr@100"], (0.5 + 0.5) / 3)
            self.assertAlmostEqual(result["ndcg@10"], (expected_q1_ndcg + expected_q2_ndcg) / 3)
            self.assertEqual(result["positive_queries"], 3)
            self.assertEqual(result["ranked_queries"], 2)
            self.assertEqual(result["missing_ranklist_queries"], 1)
            self.assertEqual(result["missing_ranklist_examples"], ["q3"])


if __name__ == "__main__":
    unittest.main()
