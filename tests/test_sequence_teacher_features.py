from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.sequence_teacher_features import teacher_item_key


class SequenceTeacherFeatureTest(unittest.TestCase):
    def test_item_key_depends_on_identity_and_visible_content(self):
        base = teacher_item_key("1", "title: shoe")
        self.assertNotEqual(base, teacher_item_key("2", "title: shoe"))
        self.assertNotEqual(base, teacher_item_key("1", "title: hat"))
        self.assertEqual(base, teacher_item_key("1", "title: shoe"))


if __name__ == "__main__":
    unittest.main()

