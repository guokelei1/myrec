#!/usr/bin/env python3
"""Read-only verifier for the C11 pre-lock review package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "prelock_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    mismatch = {}
    for relative, expected in manifest["sha256"].items():
        observed = digest(ROOT / relative)
        if observed != expected:
            mismatch[relative] = {"expected": expected, "observed": observed}
    if mismatch:
        raise SystemExit(json.dumps({"status": "FAIL", "mismatch": mismatch}, sort_keys=True))
    print(
        json.dumps(
            {
                "status": "PASS",
                "review_status": manifest["review_status"],
                "files": len(manifest["sha256"]),
                "manifest_sha256": digest(MANIFEST),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
