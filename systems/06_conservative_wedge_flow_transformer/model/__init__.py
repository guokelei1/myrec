"""C06 candidate-local Hodge-trusted flow architecture primitive."""

from .wedge_flow import (
    ConservativeWedgeFlowProbeRanker,
    WedgeFlowOutput,
    explicit_wedge_flow,
    low_rank_hodge_calibration,
)
from .information_barrier import (
    CANDIDATE,
    HISTORY,
    PAD,
    QUERY,
    build_information_barrier_mask,
)
from .transformer_core import (
    BlockSparseWedgeFlowTransformerRanker,
    TransformerCoreOutput,
)
from .controls import (
    CenteredCrossAttentionOutput,
    CenteredCrossAttentionProbeRanker,
)
from .complexity import dominant_probe_flops

__all__ = [
    "ConservativeWedgeFlowProbeRanker",
    "WedgeFlowOutput",
    "explicit_wedge_flow",
    "low_rank_hodge_calibration",
    "PAD",
    "QUERY",
    "CANDIDATE",
    "HISTORY",
    "build_information_barrier_mask",
    "BlockSparseWedgeFlowTransformerRanker",
    "TransformerCoreOutput",
    "CenteredCrossAttentionOutput",
    "CenteredCrossAttentionProbeRanker",
    "dominant_probe_flops",
]
