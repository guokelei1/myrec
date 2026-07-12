from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C71_SCORE = REPO_ROOT / "systems/71_logged_choice_gradient_signal_probe/execution/score_gate.py"


def load_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c72_bound_c71_score_helpers", C71_SCORE)
    if spec is None or spec.loader is None:
        raise RuntimeError("C72 cannot load locked C71 score helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
