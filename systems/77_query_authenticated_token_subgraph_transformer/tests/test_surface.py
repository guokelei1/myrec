import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe.c76_surface import make_surface  # noqa: E402


def test_c76_surface_is_deterministic() -> None:
    left = make_surface(
        requests=32,
        candidates=8,
        history_events=6,
        attributes=8,
        values_per_attribute=8,
        seed=20265701,
        split="validation",
    )
    right = make_surface(
        requests=32,
        candidates=8,
        history_events=6,
        attributes=8,
        values_per_attribute=8,
        seed=20265701,
        split="validation",
    )
    digest_left = hashlib.sha256(left.tokens.numpy().tobytes()).hexdigest()
    digest_right = hashlib.sha256(right.tokens.numpy().tobytes()).hexdigest()
    assert digest_left == digest_right
