"""Load C76's exact generator under private module names."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


REPO = Path(__file__).resolve().parents[3]
C76 = REPO / "systems/76_counterfactual_layer_trajectory_transformer"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


_cltt = _module("_c78_private_c76_cltt", C76 / "model/cltt.py")
_old_model = sys.modules.get("model")
_old_cltt = sys.modules.get("model.cltt")
_package = types.ModuleType("model")
_package.__path__ = []
_package.cltt = _cltt
sys.modules["model"] = _package
sys.modules["model.cltt"] = _cltt
try:
    _synthetic = _module("_c78_private_c76_synthetic", C76 / "probe/synthetic.py")
finally:
    if _old_model is None:
        sys.modules.pop("model", None)
    else:
        sys.modules["model"] = _old_model
    if _old_cltt is None:
        sys.modules.pop("model.cltt", None)
    else:
        sys.modules["model.cltt"] = _old_cltt

SyntheticSurface = _synthetic.SyntheticSurface
make_surface = _synthetic.make_surface
NUISANCE_POSITIVE = _synthetic.NUISANCE_POSITIVE
NUISANCE_NEGATIVE = _synthetic.NUISANCE_NEGATIVE
