from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.full_token_baseline import (  # noqa: E402
    build_donor_index,
    choose_fresh_wrong_donor,
    clicked_labels,
    history_length_bin,
    item_text,
)


def record(request_id: str, user: str, ts: int, history_ts: int) -> dict:
    return {
        "request_id": request_id,
        "user_id": user,
        "ts": ts,
        "history": [{"item_id": request_id + "h", "ts": history_ts}],
    }


def test_fresh_wrong_donor_is_prior_and_different_user() -> None:
    records = [
        record("d1", "u1", 8, 6),
        record("d2", "u2", 9, 7),
        record("target", "u3", 10, 8),
    ]
    boundaries = [1, 2, 3, 5]
    index = build_donor_index(records, boundaries)
    donor, slice_start, slice_stop, ratio = choose_fresh_wrong_donor(
        records,
        index,
        2,
        boundaries,
        seed=1,
        freshness_ratio_max=2.0,
        search_back=10,
    )
    assert donor in {0, 1}
    assert (slice_start, slice_stop) == (0, 1)
    assert records[donor]["user_id"] != "u3"
    assert max(event["ts"] for event in records[donor]["history"]) < 10
    assert ratio is not None and ratio <= 2.0


def test_later_donor_request_can_supply_strictly_prior_history() -> None:
    records = [
        record("target", "u1", 10, 8),
        record("later-request", "u2", 20, 9),
    ]
    boundaries = [1, 2, 3]
    donor, slice_start, slice_stop, ratio = choose_fresh_wrong_donor(
        records,
        build_donor_index(records, boundaries),
        0,
        boundaries,
        seed=2,
        freshness_ratio_max=2.0,
        search_back=10,
    )
    assert donor == 1
    assert (slice_start, slice_stop) == (0, 1)
    assert ratio == 2.0


def test_future_tail_is_removed_from_request_time_donor_prefix() -> None:
    target = record("target", "u1", 10, 8)
    donor = {
        "request_id": "donor",
        "user_id": "u2",
        "ts": 30,
        "history": [
            {"item_id": "old", "ts": 9},
            {"item_id": "future", "ts": 20},
        ],
    }
    records = [target, donor]
    boundaries = [1, 2, 3]
    donor_index, slice_start, slice_stop, ratio = choose_fresh_wrong_donor(
        records,
        build_donor_index(records, boundaries),
        0,
        boundaries,
        seed=3,
        freshness_ratio_max=2.0,
        search_back=10,
    )
    assert donor_index == 1
    assert (slice_start, slice_stop) == (0, 1)
    assert donor["history"][slice_start:slice_stop] == [{"item_id": "old", "ts": 9}]
    assert ratio == 2.0


def test_longer_prefix_is_tail_truncated_to_exact_target_length() -> None:
    target = record("target", "u1", 20, 18)
    donor = {
        "request_id": "donor",
        "user_id": "u2",
        "ts": 30,
        "history": [
            {"item_id": "old", "ts": 1},
            {"item_id": "matched", "ts": 19},
        ],
    }
    records = [target, donor]
    boundaries = [1, 2, 3]
    donor_index, slice_start, slice_stop, ratio = choose_fresh_wrong_donor(
        records,
        build_donor_index(records, boundaries),
        0,
        boundaries,
        seed=4,
        freshness_ratio_max=2.0,
        search_back=10,
    )
    assert donor_index == 1
    assert donor["history"][slice_start:slice_stop] == [{"item_id": "matched", "ts": 19}]
    assert ratio == 2.0



def test_full_token_text_bins_and_labels() -> None:
    assert item_text({"title": "T", "brand": "B", "seller": "S", "cat": ["C"]}) == "T B S C"
    assert history_length_bin(0, [1, 2, 5]) == 0
    assert history_length_bin(4, [1, 2, 5]) == 5
    assert clicked_labels({"candidates": [{"clicked": 1}, {}]}) == [1.0, 0.0]

