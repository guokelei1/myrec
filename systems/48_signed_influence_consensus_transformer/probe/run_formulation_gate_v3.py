"""Unconditional-copy wrapper for length-one negative-stride C48 views."""

from __future__ import annotations

import argparse
import json

import numpy as np

import run_formulation_gate as v1
from freeze_formulation_lock_v3 import verify_formulation_lock_v3


_LOCKED_SCORE_ONE = v1.score_one


def _copy(value):
    return np.array(value, dtype=np.float32, copy=True, order="C")


def score_one(query, history, candidates, config):
    return _LOCKED_SCORE_ONE(_copy(query), _copy(history), _copy(candidates), config)


def run(config):
    verify_formulation_lock_v3(config)
    v1.score_one = score_one
    return v1.run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(v1.load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
