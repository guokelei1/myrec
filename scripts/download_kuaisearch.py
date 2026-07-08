#!/usr/bin/env python
"""Download the KuaiSearch Lite raw files listed in the dataset config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/kuaisearch_lite.json")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument(
        "--connections",
        type=int,
        default=8,
        help="Parallel connections when aria2c is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    raw_dir = Path(args.raw_dir or config["raw_dir"])
    repo = config["source"]["hf_repo"]

    for item in config["files"]:
        rel_path = item["path"]
        output = raw_dir / rel_path
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size == int(item["size_bytes"]):
            print(f"present {output}")
            continue
        tmp = output.with_suffix(output.suffix + ".part")
        url = HF_RESOLVE.format(repo=repo, path=rel_path)
        print(f"downloading {rel_path}")
        _download(url=url, output=tmp, connections=args.connections)
        if tmp.stat().st_size != int(item["size_bytes"]):
            raise SystemExit(
                f"size mismatch for {tmp}: got {tmp.stat().st_size}, "
                f"expected {item['size_bytes']}"
            )
        tmp.replace(output)
    return 0


def _download(url: str, output: Path, connections: int) -> None:
    aria2c = shutil_which("aria2c")
    if aria2c:
        subprocess.run(
            [
                aria2c,
                "-c",
                "-x",
                str(connections),
                "-s",
                str(connections),
                "--file-allocation=none",
                "--summary-interval=30",
                "-d",
                str(output.parent),
                "-o",
                output.name,
                url,
            ],
            check=True,
        )
        return

    wget = shutil_which("wget")
    if not wget:
        raise SystemExit("aria2c or wget is required for resumable downloads")
    subprocess.run([wget, "-c", "-nv", "-O", str(output), url], check=True)


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
