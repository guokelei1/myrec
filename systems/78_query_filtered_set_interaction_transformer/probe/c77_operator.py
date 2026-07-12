"""Load the locked C77 operator under a private module name."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
PATH = REPO / "systems/77_query_authenticated_token_subgraph_transformer/model/qats.py"
SPEC = importlib.util.spec_from_file_location("_c78_private_c77_qats", PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

QUERY = MODULE.QUERY
CANDIDATE = MODULE.CANDIDATE
HISTORY = MODULE.HISTORY
InteractionBlock = MODULE.InteractionBlock
InteractionTransformer = MODULE.InteractionTransformer
GraphDiagnostics = MODULE.GraphDiagnostics
authenticated_graphs = MODULE.authenticated_graphs
structured_anchor_table = MODULE.structured_anchor_table
