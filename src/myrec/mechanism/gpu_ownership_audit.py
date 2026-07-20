"""Read-only physical-GPU ownership audit for deep-dive workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def audit_gpu_ownership(
    *,
    proc_root: str | Path = "/proc",
    expected_gpu_count: int = 4,
) -> dict[str, Any]:
    """Scan live processes and fail closed on overlapping project GPU workers."""

    records = _read_process_records(Path(proc_root))
    return audit_gpu_process_records(records, expected_gpu_count=expected_gpu_count)


def audit_gpu_process_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_gpu_count: int = 4,
) -> dict[str, Any]:
    """Audit normalized process records; exposed separately for deterministic tests."""

    if expected_gpu_count <= 0:
        raise ValueError("expected_gpu_count must be positive")
    workers = []
    failures = []
    for record in records:
        normalized = _normalize_worker(record, expected_gpu_count=expected_gpu_count)
        if normalized is not None:
            workers.append(normalized)

    by_gpu: dict[int, list[dict[str, Any]]] = {}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for worker in workers:
        by_gpu.setdefault(worker["physical_gpu"], []).append(worker)
        run_id = worker.get("run_id")
        if run_id:
            by_run.setdefault(run_id, []).append(worker)
    for gpu, rows in sorted(by_gpu.items()):
        if len(rows) > 1:
            failures.append(
                "multiple active project workers on physical GPU "
                f"{gpu}: {','.join(str(row['pid']) for row in rows)}"
            )
    for run_id, rows in sorted(by_run.items()):
        if len(rows) > 1:
            failures.append(
                f"multiple active writers for run {run_id}: "
                + ",".join(str(row["pid"]) for row in rows)
            )

    active_gpus = sorted(by_gpu)
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_gpu_ownership_audit",
        "status": "failed" if failures else "completed",
        "expected_gpu_count": expected_gpu_count,
        "active_worker_count": len(workers),
        "active_physical_gpus": active_gpus,
        "all_expected_gpus_active": active_gpus == list(range(expected_gpu_count)),
        "workers": sorted(workers, key=lambda row: (row["physical_gpu"], row["pid"])),
        "failures": failures,
        "scientific_effect_values_read": False,
        "qrels_files_opened": False,
        "source_test_opened": False,
    }


def _normalize_worker(
    record: Mapping[str, Any], *, expected_gpu_count: int
) -> dict[str, Any] | None:
    pid = record.get("pid")
    argv = record.get("argv")
    environ = record.get("environ")
    if type(pid) is not int or pid <= 0:
        raise ValueError("process record pid must be a positive integer")
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
        raise ValueError("process record argv must be a sequence")
    argv = [str(value) for value in argv]
    if not argv:
        return None
    if not isinstance(environ, Mapping):
        raise ValueError("process record environ must be a mapping")

    executable = Path(argv[0]).name.lower()
    if "python" not in executable:
        return None
    script = next(
        (
            token
            for token in argv[1:]
            if token.endswith(".py") and _is_project_gpu_worker_script(token)
        ),
        None,
    )
    if script is None:
        return None
    device = _argument(argv, "--device")
    if device is None or device == "cpu":
        return None
    if device == "cuda":
        local_index = 0
    elif device.startswith("cuda:") and device[5:].isdigit():
        local_index = int(device[5:])
    else:
        raise ValueError(f"unsupported worker device: {device}")

    visible_raw = str(environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible_raw:
        visible = [value.strip() for value in visible_raw.split(",")]
        if any(not value.isdigit() for value in visible):
            raise ValueError(f"non-numeric CUDA_VISIBLE_DEVICES for pid {pid}")
        visible_indices = [int(value) for value in visible]
        if local_index >= len(visible_indices):
            raise ValueError(f"local CUDA device is outside visibility for pid {pid}")
        physical_gpu = visible_indices[local_index]
    else:
        physical_gpu = local_index
    if not 0 <= physical_gpu < expected_gpu_count:
        raise ValueError(f"physical GPU is outside registered range for pid {pid}")

    return {
        "pid": pid,
        "physical_gpu": physical_gpu,
        "local_device": device,
        "cuda_visible_devices": visible_raw or None,
        "script": script,
        "run_id": _argument(argv, "--run-id"),
        "command": argv,
    }


def _is_project_gpu_worker_script(token: str) -> bool:
    name = Path(token).name
    return name.startswith(
        (
            "score_",
            "extract_",
            "observe_",
            "train_",
            "run_deep_dive_q2_optimizer_replay",
            "run_deep_dive_q3_optimizer_replay",
        )
    )


def _argument(argv: Sequence[str], flag: str) -> str | None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) > 1:
        raise ValueError(f"duplicate process argument: {flag}")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        raise ValueError(f"missing process argument value: {flag}")
    return argv[index + 1]


def _read_process_records(proc_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            cmdline = (path / "cmdline").read_bytes().split(b"\0")
            argv = [value.decode("utf-8", errors="replace") for value in cmdline if value]
            if not argv:
                continue
            environ_values = (path / "environ").read_bytes().split(b"\0")
            environ = {}
            for value in environ_values:
                if b"=" not in value:
                    continue
                key, raw = value.split(b"=", 1)
                environ[key.decode("utf-8", errors="replace")] = raw.decode(
                    "utf-8", errors="replace"
                )
            records.append({"pid": int(path.name), "argv": argv, "environ": environ})
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            # Processes can exit between directory enumeration and file reads.
            continue
    return records
