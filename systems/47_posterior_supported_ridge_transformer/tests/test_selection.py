from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "probe/selection.py"
SPEC = importlib.util.spec_from_file_location("c47_selection", MODULE)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_length_bins_are_right_closed():
    edges = [1, 2, 3, 5, 10, 20, 50]
    assert MOD.length_bin(1, edges) == 1
    assert MOD.length_bin(4, edges) == 5
    assert MOD.length_bin(50, edges) == 50


def test_hashes_are_order_stable_where_declared():
    assert MOD.compact_index_hash([3, 1, 2]) == MOD.compact_index_hash([2, 3, 1])
    assert MOD.stable_key(7, "a", "r") == MOD.stable_key(7, "a", "r")
    assert MOD.stable_key(7, "a", "r") != MOD.stable_key(7, "b", "r")
