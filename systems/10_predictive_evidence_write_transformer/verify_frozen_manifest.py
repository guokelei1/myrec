#!/usr/bin/env python3
"""Read-only independent verifier for the C10 pre-outcome lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "frozen_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    observed = {name: digest(ROOT / name) for name in manifest["sha256"]}
    mismatch = {
        name: {"expected": manifest["sha256"][name], "observed": observed[name]}
        for name in observed
        if observed[name] != manifest["sha256"][name]
    }
    if mismatch:
        raise SystemExit(json.dumps({"status": "FAIL", "mismatch": mismatch}, sort_keys=True))
    print(json.dumps({"status": "PASS", "files": len(observed), "manifest_sha256": digest(MANIFEST)}, sort_keys=True))


if __name__ == "__main__":
    main()
