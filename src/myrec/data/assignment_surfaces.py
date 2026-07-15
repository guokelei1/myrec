"""Materialize label-free request surfaces from assignment match types."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def materialize_assignment_surfaces(
    assignments_path: str | Path,
    output_dir: str | Path,
    intersection_surfaces: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    assignments_path = Path(assignments_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()
    members: dict[str, set[str]] = defaultdict(set)
    all_requests = set()
    for row in iter_jsonl(assignments_path):
        request_id = str(row["request_id"])
        if request_id in all_requests:
            raise ValueError(f"duplicate request_id={request_id}")
        all_requests.add(request_id)
        match_type = str(row.get("match_type") or "unspecified")
        if not _SAFE_NAME.fullmatch(match_type):
            raise ValueError(f"unsafe match_type={match_type!r}")
        members[match_type].add(request_id)
    if not all_requests:
        raise ValueError("assignment file is empty")
    members["all"] = all_requests
    intersection_sources = {}
    for surface_name, surface_path_value in sorted(
        (intersection_surfaces or {}).items()
    ):
        if not _SAFE_NAME.fullmatch(surface_name):
            raise ValueError(f"unsafe surface_name={surface_name!r}")
        surface_path = Path(surface_path_value)
        surface_ids = {
            line.strip()
            for line in surface_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        intersection_sources[surface_name] = {
            "path": str(surface_path),
            "sha256": sha256_file(surface_path),
        }
        for match_type, request_ids in list(members.items()):
            if match_type == "all" or "__" in match_type:
                continue
            intersection = request_ids & surface_ids
            if intersection:
                members[f"{match_type}__{surface_name}"] = intersection
    files = {}
    for name, request_ids in sorted(members.items()):
        path = output_dir / f"{name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for request_id in sorted(request_ids):
                handle.write(request_id + "\n")
        files[name] = {
            "path": str(path),
            "requests": len(request_ids),
            "sha256": sha256_file(path),
        }
    result = {
        "assignments_path": str(assignments_path),
        "assignments_sha256": sha256_file(assignments_path),
        "label_free": True,
        "intersection_sources": intersection_sources,
        "files": files,
    }
    write_json(output_dir / "manifest.json", result)
    return result
