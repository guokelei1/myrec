from myrec.baselines.instructrec import _stable_internal_dev_request_ids


def test_internal_dev_partition_is_stable_and_nonempty():
    request_ids = {f"request-{index}" for index in range(1000)}
    first = _stable_internal_dev_request_ids(
        request_ids, fraction=0.20, seed=20260717
    )
    second = _stable_internal_dev_request_ids(
        request_ids, fraction=0.20, seed=20260717
    )

    assert first == second
    assert 150 < len(first) < 250
    assert first <= request_ids
    assert _stable_internal_dev_request_ids(
        request_ids, fraction=0.0, seed=20260717
    ) == set()


def test_internal_dev_partition_changes_with_seed():
    request_ids = {f"request-{index}" for index in range(1000)}
    first = _stable_internal_dev_request_ids(
        request_ids, fraction=0.20, seed=20260717
    )
    second = _stable_internal_dev_request_ids(
        request_ids, fraction=0.20, seed=20260718
    )

    assert first != second
