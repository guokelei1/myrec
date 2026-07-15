#!/usr/bin/env python3
"""Download and verify the pinned KuaiSearch Full files.

The download target is deliberately a sibling of the existing Lite tree:
``data/raw/kuaisearch_full``.  This script never writes below
``data/raw/kuaisearch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "benchen4395/KuaiSearch"
REPO_TYPE = "dataset"
# The commit containing the currently published Full and Lite files.
REVISION = "09807c773ce67360ed8df30842e372182fcf7ad9"
DEFAULT_ENDPOINT = "https://huggingface.co"

EXPECTED_FILES: dict[str, dict[str, Any]] = {
    "items/train.jsonl": {
        "size": 7_837_345_198,
        "sha256": "312d9d404327990dad9d852dc900d31cb7103e86c2d1398c832778d7c3979e00",
    },
    "rank/train.jsonl": {
        "size": 77_675_031_316,
        "sha256": "78550c4eb2c3e74866bdbe5688174d0958db116eb1b2f37218fc0df43e6b0a62",
    },
    "recall/train.jsonl": {
        "size": 1_211_473_290,
        "sha256": "90935bde721321f8ece70a84065f15e81167725f678b9275c9edf1d4297e5830",
    },
    "users/train.jsonl": {
        "size": 43_805_433,
        "sha256": "cea7c7048ac9bb756c74a891ca07b21e68a3d8ac67ecde6161df8cdf42136ffc",
    },
    "relevance/train.jsonl": {
        "size": 15_827_789,
        "sha256": "b26c79a3ae6fde4aadc756c7b9eda9f6dcc6e112b725db603def868ab9f9c720",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=root / "data/raw/kuaisearch_full",
        help="download directory (must be a direct child of data/raw)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Hugging Face endpoint; defaults to the official endpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel download workers (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the remote manifest and local target without downloading",
    )
    return parser.parse_args()


def assert_safe_target(target_arg: Path) -> Path:
    root = repo_root()
    raw_root = (root / "data/raw").resolve()
    lite_root = (root / "data/raw/kuaisearch").resolve()
    target = target_arg.expanduser().resolve()

    if target.parent != raw_root:
        raise SystemExit(
            f"Refusing target outside the direct data/raw siblings: {target}"
        )
    if target == lite_root or lite_root in target.parents:
        raise SystemExit(f"Refusing to write inside the existing Lite tree: {target}")
    if not (lite_root / "rank_lite/train.jsonl").is_file():
        raise SystemExit(f"Expected existing Lite data was not found: {lite_root}")
    return target


def validate_remote_manifest(endpoint: str, token: str | None) -> dict[str, dict[str, Any]]:
    api = HfApi(endpoint=endpoint, token=token)
    info = api.dataset_info(
        REPO_ID,
        revision=REVISION,
        files_metadata=True,
    )
    if info.sha != REVISION:
        raise RuntimeError(f"Remote revision mismatch: expected {REVISION}, got {info.sha}")

    siblings = {entry.rfilename: entry for entry in info.siblings}
    manifest: dict[str, dict[str, Any]] = {}
    for filename, expected in EXPECTED_FILES.items():
        entry = siblings.get(filename)
        if entry is None:
            raise RuntimeError(f"Remote file is missing at pinned revision: {filename}")
        actual_size = getattr(entry, "size", None)
        actual_lfs = getattr(getattr(entry, "lfs", None), "sha256", None)
        if actual_size != expected["size"] or actual_lfs != expected["sha256"]:
            raise RuntimeError(
                f"Remote metadata mismatch for {filename}: "
                f"size={actual_size}, sha256={actual_lfs}"
            )
        manifest[filename] = {
            "size": actual_size,
            "sha256": actual_lfs,
        }
    return manifest


def validate_disk_space(target: Path, manifest: dict[str, dict[str, Any]]) -> None:
    required = sum(item["size"] for item in manifest.values())
    free = shutil.disk_usage(target.parent).free
    safety_margin = 2 * 1024**3
    if free < required + safety_margin:
        raise SystemExit(
            f"Insufficient free space under {target.parent}: "
            f"need at least {human_bytes(required + safety_margin)}, "
            f"have {human_bytes(free)}"
        )
    print(f"Remote Full payload: {human_bytes(required)}")
    print(f"Free space on target filesystem: {human_bytes(free)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_files(target: Path, manifest: dict[str, dict[str, Any]]) -> None:
    print("Verifying downloaded files...")
    for filename, expected in manifest.items():
        path = target / filename
        if not path.is_file():
            raise RuntimeError(f"Downloaded file is missing: {path}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise RuntimeError(
                f"Size mismatch for {path}: expected {expected['size']}, got {size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: "
                f"expected {expected['sha256']}, got {actual_sha256}"
            )
        print(f"  OK {filename} ({human_bytes(size)})")


def write_manifest(target: Path, manifest: dict[str, dict[str, Any]]) -> None:
    record = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": REVISION,
        "endpoint": DEFAULT_ENDPOINT,
        "files": manifest,
    }
    (target / ".kuaisearch_full_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    target = assert_safe_target(args.target)
    token = os.environ.get("HF_TOKEN")
    print(f"Source: {REPO_ID}@{REVISION}")
    print(f"Target: {target}")
    print(f"Endpoint: {args.endpoint}")
    print("Lite tree is protected: data/raw/kuaisearch")

    manifest = validate_remote_manifest(args.endpoint, token)
    validate_disk_space(target, manifest)
    print("Remote manifest validated:")
    for filename, item in manifest.items():
        print(f"  {filename}: {human_bytes(item['size'])}")

    if args.dry_run:
        print("Dry run complete; no data was downloaded.")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        local_dir=target,
        allow_patterns=list(manifest),
        max_workers=args.workers,
        endpoint=args.endpoint,
        token=token,
    )
    verify_local_files(target, manifest)
    write_manifest(target, manifest)
    print(f"Download complete: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
