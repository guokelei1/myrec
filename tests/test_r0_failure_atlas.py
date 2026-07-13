from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_r0_failure_atlas import cohort_tags, query_recall, user_fold  # noqa: E402


def test_query_recall_uses_normalized_character_coverage() -> None:
    assert query_recall("苹果 手机", ["新款苹果手机壳"]) == 1.0
    assert query_recall("苹果", ["运动鞋"]) == 0.0
    assert query_recall("", ["anything"]) is None


def test_cohort_tags_separate_query_aligned_and_conflicting_repeats() -> None:
    aligned = {
        "query": "苹果手机",
        "history": [{"item_id": "a", "title": "旧款手机"}],
        "candidates": [{"item_id": "a", "title": "苹果手机新款"}],
    }
    conflict = {
        "query": "苹果手机",
        "history": [{"item_id": "a", "title": "运动鞋"}],
        "candidates": [{"item_id": "a", "title": "夏季运动鞋"}],
    }
    assert "repeat_query_aligned" in cohort_tags(aligned)
    assert "repeat_query_conflict" in cohort_tags(conflict)


def test_cohort_tags_strict_nonrepeat_history_alignment() -> None:
    record = {
        "query": "蓝牙耳机",
        "history": [{"item_id": "h", "title": "蓝牙无线耳机"}],
        "candidates": [{"item_id": "c", "title": "候选商品"}],
    }
    tags = cohort_tags(record)
    assert "strict_nonrepeat" in tags
    assert "nonrepeat_history_query_aligned" in tags


def test_user_fold_is_deterministic_and_binary() -> None:
    assert user_fold("user-1") == user_fold("user-1")
    assert user_fold("user-1") in {0, 1}
