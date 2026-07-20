from __future__ import annotations

from pathlib import Path

from myrec.mechanism.attention_logit_interventions import LOGIT_MODES
from myrec.mechanism.attention_logit_runtime import (
    N11_MANIFEST_PATH,
    N11_MANIFEST_SHA256,
    _load_n11_manifest,
    attention_logit_runtime_implementation_identity,
)
from myrec.mechanism.attention_logit_scoring import (
    ATTENTION_LOGIT_CONDITIONS,
    MODE_TO_CONDITION,
)
from myrec.utils.hashing import sha256_file


def test_n11_manifest_is_immutable_and_complete() -> None:
    assert Path(N11_MANIFEST_PATH).is_file()
    assert sha256_file(N11_MANIFEST_PATH) == N11_MANIFEST_SHA256
    manifest = _load_n11_manifest(N11_MANIFEST_PATH)
    assert manifest["frozen_inputs"]["request_count"] == 8000
    assert manifest["frozen_inputs"]["blocks"] == [13, 20, 27]
    assert manifest["claim_boundary"]["diagnostic_only"] is True
    assert manifest["claim_boundary"]["architecture_authorized"] is False
    assert set(manifest["model_bindings"]) == {
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    }


def test_n11_condition_registry_is_pairwise_and_ordered() -> None:
    assert LOGIT_MODES == ("identity", "scale_half", "scale_double", "sign_flip")
    assert ATTENTION_LOGIT_CONDITIONS[:4] == (
        "baseline_full",
        "baseline_null",
        "full_qk_identity",
        "null_qk_identity",
    )
    assert set(MODE_TO_CONDITION) == set(LOGIT_MODES)
    assert len(ATTENTION_LOGIT_CONDITIONS) == 10
    assert len(set(ATTENTION_LOGIT_CONDITIONS)) == 10


def test_n11_implementation_digest_covers_new_operator() -> None:
    identity = attention_logit_runtime_implementation_identity()
    paths = {item["path"] for item in identity["files"]}
    assert "src/myrec/mechanism/attention_logit_interventions.py" in paths
    assert "src/myrec/mechanism/attention_logit_scoring.py" in paths
    assert "scripts/score_deep_dive_attention_logits.py" in paths
    assert identity["digest"]

