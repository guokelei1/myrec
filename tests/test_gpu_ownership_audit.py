from __future__ import annotations

from myrec.mechanism.gpu_ownership_audit import audit_gpu_process_records


def _worker(pid: int, gpu: int, run_id: str) -> dict:
    return {
        "pid": pid,
        "argv": [
            "/env/bin/python",
            "scripts/score_deep_dive_postblock_sweep.py",
            "--run-id",
            run_id,
            "--device",
            "cuda:0",
        ],
        "environ": {"CUDA_VISIBLE_DEVICES": str(gpu)},
    }


def test_four_disjoint_workers_pass_physical_gpu_ownership_audit() -> None:
    result = audit_gpu_process_records(
        [_worker(100 + gpu, gpu, f"run-{gpu}") for gpu in range(4)]
    )
    assert result["status"] == "completed"
    assert result["active_worker_count"] == 4
    assert result["active_physical_gpus"] == [0, 1, 2, 3]
    assert result["all_expected_gpus_active"] is True
    assert result["failures"] == []
    assert result["scientific_effect_values_read"] is False
    assert result["qrels_files_opened"] is False


def test_same_physical_gpu_and_duplicate_writer_fail_closed() -> None:
    records = [_worker(100, 2, "same-run"), _worker(101, 2, "same-run")]
    result = audit_gpu_process_records(records)
    assert result["status"] == "failed"
    assert any("multiple active project workers" in value for value in result["failures"])
    assert any("multiple active writers" in value for value in result["failures"])


def test_wrapper_shell_is_not_counted_as_a_worker() -> None:
    result = audit_gpu_process_records(
        [
            {
                "pid": 100,
                "argv": [
                    "bash",
                    "scripts/run_deep_dive_resume_loop.sh",
                    "--",
                    "python",
                    "scripts/score_deep_dive_postblock_sweep.py",
                    "--device",
                    "cuda:0",
                ],
                "environ": {"CUDA_VISIBLE_DEVICES": "0"},
            }
        ]
    )
    assert result["active_worker_count"] == 0
    assert result["active_physical_gpus"] == []


def test_multiple_visible_devices_map_local_index_to_physical_gpu() -> None:
    record = _worker(100, 0, "run")
    record["argv"][-1] = "cuda:1"
    record["environ"]["CUDA_VISIBLE_DEVICES"] = "3,1"
    result = audit_gpu_process_records([record])
    assert result["workers"][0]["physical_gpu"] == 1

