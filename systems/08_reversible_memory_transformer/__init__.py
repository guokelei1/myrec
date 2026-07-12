"""C08 reversible write--probe--undo Transformer prototype."""

try:
    from .reversible_memory import (
        ReversibleCouplingMemory,
        TinyReversibleRanker,
        reversible_coupling_step,
    )
except ImportError:  # Direct pytest collection from a digit-prefixed directory.
    from reversible_memory import (  # type: ignore[no-redef]
        ReversibleCouplingMemory,
        TinyReversibleRanker,
        reversible_coupling_step,
    )

__all__ = [
    "ReversibleCouplingMemory",
    "TinyReversibleRanker",
    "reversible_coupling_step",
]
