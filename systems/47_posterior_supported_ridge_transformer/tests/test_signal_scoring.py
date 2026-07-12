from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "probe/signal_scoring.py"
SPEC = importlib.util.spec_from_file_location("c47_signal_scoring", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def fixture():
    query = np.asarray([1.0, 0.5, -0.25], dtype=np.float32)
    history = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    candidates = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    return query, history, candidates


def test_no_history_returns_base_exactly():
    q, _, c = fixture()
    out = MOD.fixed_scores(q, np.empty((0, 3), dtype=np.float32), c)
    assert np.array_equal(out.posterior_supported, out.base)
    assert np.array_equal(out.plain_ridge, out.base)
    assert np.array_equal(out.softmax_attention, out.base)
    assert np.array_equal(out.correction, np.zeros(3, dtype=np.float32))


def test_support_contracts_plain_correction():
    out = MOD.fixed_scores(*fixture())
    assert np.all(out.support >= 0.0)
    assert np.all(out.support <= 1.0)
    assert np.all(np.abs(out.correction) <= np.abs(out.plain_correction) + 1e-7)
    assert out.support[1] == 0.0


def test_candidate_and_history_permutations_preserve_coordinates():
    q, h, c = fixture()
    expected = MOD.fixed_scores(q, h, c)
    candidate_order = np.asarray([2, 0, 1])
    candidate = MOD.fixed_scores(q, h, c[candidate_order])
    assert np.allclose(expected.posterior_supported[candidate_order], candidate.posterior_supported)
    assert np.allclose(expected.support[candidate_order], candidate.support)
    history = MOD.fixed_scores(q, h[::-1], c)
    assert np.allclose(expected.posterior_supported, history.posterior_supported, atol=1e-7)


def test_numpy_operator_matches_explicit_projector():
    q, h, c = fixture()
    out = MOD.fixed_scores(q, h, c, ridge=1.0)
    qn = q / np.linalg.norm(q)
    hn = h / np.linalg.norm(h, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    projector = hn.T @ np.linalg.inv(hn @ hn.T + np.eye(len(hn))) @ hn
    write = projector @ qn
    support = np.einsum("cd,df,cf->c", cn, projector, cn)
    assert np.allclose(out.query_write, write, atol=1e-7)
    assert np.allclose(out.support, support, atol=1e-7)
    assert np.allclose(out.correction, support * (cn @ write), atol=1e-7)
