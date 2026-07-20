from pathlib import Path

from myrec.mechanism.candidate_gap_runtime import (
    N10_CANDIDATE_GAP_MANIFEST_PATH,
    N10_CANDIDATE_GAP_MANIFEST_SHA256,
    _load_candidate_manifest,
)
from myrec.utils.hashing import sha256_file


def test_candidate_gap_manifest_digest_is_frozen() -> None:
    path = Path(N10_CANDIDATE_GAP_MANIFEST_PATH)
    assert sha256_file(path) == N10_CANDIDATE_GAP_MANIFEST_SHA256
    manifest = _load_candidate_manifest(path)
    assert manifest["_sha256"] == N10_CANDIDATE_GAP_MANIFEST_SHA256
    assert manifest["parent_contract"]["source_test_opened"] is False
