from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q3_QUEUE = ROOT / "scripts/run_deep_dive_q3_after_gate_queue.sh"
Q2_QUEUE = ROOT / "scripts/run_deep_dive_q2_d2_formal_queue.sh"
Q2_SHARD1 = ROOT / "scripts/run_deep_dive_q2_selected_branch_shard1_queue.sh"
Q2_LANE2 = ROOT / "scripts/run_deep_dive_q2_postselected_breadth_queue.sh"
RESUME_LOOP = ROOT / "scripts/run_deep_dive_resume_loop.sh"
Q3_B20_SHORT_BACKFILL = (
    ROOT / "scripts/run_deep_dive_q3_b20_short_backfill.sh"
)
Q2_B27_PARTIAL_BACKFILL = (
    ROOT / "scripts/run_deep_dive_q2_b27_fold1_partial_backfill.sh"
)
Q2_SHORT_BACKFILL_AFTER_ROPE = (
    ROOT / "scripts/run_deep_dive_q2_short_backfill_after_rope.sh"
)
GPU3_Q2_BREADTH_BACKFILL = (
    ROOT / "scripts/run_deep_dive_gpu3_q2_breadth_backfill.sh"
)
MLP_FORMATION_LANE = ROOT / "scripts/run_deep_dive_mlp_formation_lane.sh"
COMPONENT_NECESSITY_LANE = (
    ROOT / "scripts/run_deep_dive_component_necessity_lane.sh"
)
COMPONENT_NECESSITY_EVAL_QUEUE = (
    ROOT / "scripts/run_deep_dive_component_necessity_eval_queue.sh"
)
COMPONENT_DESIGN_SYNTHESIS_QUEUE = (
    ROOT / "scripts/run_deep_dive_component_design_synthesis_queue.sh"
)
QUEUE_SCRIPTS = (
    Q2_QUEUE,
    Q3_QUEUE,
    Q2_SHARD1,
    Q2_LANE2,
    RESUME_LOOP,
    Q3_B20_SHORT_BACKFILL,
    Q2_B27_PARTIAL_BACKFILL,
    Q2_SHORT_BACKFILL_AFTER_ROPE,
    GPU3_Q2_BREADTH_BACKFILL,
    MLP_FORMATION_LANE,
    COMPONENT_NECESSITY_LANE,
    COMPONENT_NECESSITY_EVAL_QUEUE,
    COMPONENT_DESIGN_SYNTHESIS_QUEUE,
    ROOT / "scripts/watch_then_run.sh",
)


def test_future_four_gpu_queue_scripts_are_valid_bash():
    for path in (
        Q3_QUEUE,
        Q2_SHARD1,
        Q2_LANE2,
        RESUME_LOOP,
        Q3_B20_SHORT_BACKFILL,
        Q2_B27_PARTIAL_BACKFILL,
        Q2_SHORT_BACKFILL_AFTER_ROPE,
        GPU3_Q2_BREADTH_BACKFILL,
        MLP_FORMATION_LANE,
        COMPONENT_NECESSITY_LANE,
        COMPONENT_NECESSITY_EVAL_QUEUE,
        COMPONENT_DESIGN_SYNTHESIS_QUEUE,
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_queue_scripts_do_not_reference_undeclared_shell_variables():
    shell_builtins = {
        "BASH_LINENO",
        "BASH_SOURCE",
        "CUDA_VISIBLE_DEVICES",
        "EUID",
        "FUNCNAME",
        "LINENO",
        "MYREC_DEEP_DIVE_RESUME_LOCK_DIR",
        "OLDPWD",
        "PATH",
        "PIPESTATUS",
        "PPID",
        "PWD",
        "PYTHONPATH",
        "RANDOM",
        "SECONDS",
        "SHELL",
        "UID",
        "USER",
    }
    for path in QUEUE_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        references = set(
            re.findall(r"\$(?:\{)?([A-Za-z_][A-Za-z0-9_]*)", text)
        )
        declarations = set(
            re.findall(
                r"(?m)^\s*(?:local\s+|export\s+|readonly\s+)?"
                r"([A-Za-z_][A-Za-z0-9_]*)=",
                text,
            )
        )
        declarations.update(
            re.findall(
                r"(?m)^\s*(?:local|export|readonly)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)\b",
                text,
            )
        )
        declarations.update(
            re.findall(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", text)
        )
        assert references <= declarations | shell_builtins, path


def test_resume_loop_allows_only_one_writer_per_canonical_metadata_path(tmp_path):
    resume_loop = tmp_path / "repository/scripts/run_deep_dive_resume_loop.sh"
    resume_loop.parent.mkdir(parents=True)
    shutil.copy2(RESUME_LOOP, resume_loop)
    resume_loop.chmod(0o755)
    writer = tmp_path / "writer.sh"
    writer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
metadata="$1"
marker="$2"
release="$3"
mkdir -p "$(dirname "$metadata")"
printf '{"status":"running"}\n' > "$metadata"
touch "$marker"
while [[ ! -f "$release" ]]; do sleep 0.05; done
printf '{"status":"completed"}\n' > "$metadata"
""",
        encoding="utf-8",
    )
    writer.chmod(0o755)
    metadata = tmp_path / "run/metadata.json"
    metadata_alias = metadata.parent / ".." / "run" / "metadata.json"
    marker = tmp_path / "writer_started"
    release = tmp_path / "release_writer"
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    first_cwd = tmp_path / "first_cwd"
    second_cwd = tmp_path / "second_cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    env = os.environ.copy()
    env.pop("MYREC_DEEP_DIVE_RESUME_LOCK_DIR", None)
    command = [str(writer), str(metadata), str(marker), str(release)]
    first = subprocess.Popen(
        [str(resume_loop), str(metadata), str(first_log), "--", *command],
        env=env,
        cwd=first_cwd,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        second = subprocess.run(
            [
                str(resume_loop),
                str(metadata_alias),
                str(second_log),
                "--",
                *command,
            ],
            env=env,
            cwd=second_cwd,
            check=False,
            timeout=5,
        )
        assert second.returncode == 7
        assert "writer lock is busy" in second_log.read_text(encoding="utf-8")
        release.touch()
        assert first.wait(timeout=5) == 0
    finally:
        release.touch(exist_ok=True)
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=5)

    completed = subprocess.run(
        [str(resume_loop), str(metadata), str(first_log), "--", *command],
        env=env,
        cwd=second_cwd,
        check=False,
        timeout=5,
    )
    assert completed.returncode == 0


def test_breadth_lane_ownership_is_static_and_disjoint():
    text = Q3_QUEUE.read_text(encoding="utf-8")
    assert 'local model_config="$config"' in text
    assert "$q3_config" not in text
    lane2 = """if [[ "$lane" == 2 ]]; then
  run_q2_attention_breadth 20
  run_block_breadth_followup_model q2 20
  run_q0_breadth
  run_optimizer_replay q2
  exit 0"""
    lane3 = """elif [[ "$lane" == 3 ]]; then
  run_q2_attention_breadth 27
  run_block_breadth_followup_model q2 27
  run_q1_breadth
  run_optimizer_replay q3
  exit 0"""
    assert lane2 in text
    assert lane3 in text

    admitted_dispatch = text.split("if [[ \"$(jq -r '.q3_sweep_admitted'", 1)[1]
    for moved_call in (
        "run_q2_attention_breadth",
        "run_q0_breadth",
        "run_q1_breadth",
        "run_optimizer_replay",
        "run_block_breadth_followup_model q2",
    ):
        assert moved_call not in admitted_dispatch


def test_q2_postselected_queues_release_ownership_before_breadth():
    q2_queue = Q2_QUEUE.read_text(encoding="utf-8")
    shard1 = Q2_SHARD1.read_text(encoding="utf-8")
    lane2 = Q2_LANE2.read_text(encoding="utf-8")
    q3_queue = Q3_QUEUE.read_text(encoding="utf-8")
    assert "compgen -G" not in shard1
    assert "compgen -G" not in q3_queue
    assert "upstream terminal status=$status path=$path" in shard1
    assert "^(1[4-9]|2[0-7])$" in shard1
    assert "^(1[4-9]|2[0-7])$" in q3_queue
    assert "invalid Q2 selected-branch eligibility" in shard1
    assert "invalid Q2 selected-branch eligibility" in q2_queue
    assert "invalid Q2 selected block" in q2_queue
    assert "invalid Q3 selected-branch eligibility" in q3_queue
    assert 'selected_block="$(jq -r \'.selected_block\'' in shard1
    assert 'selected_block="$(jq -r \'.selected_block\'' in q3_queue
    assert "selected_branch_b${selected_block}_smoke_gpu_v1/metadata.json" in shard1
    assert "selected_branch_b${selected_block}_smoke_gpu_v1/metadata.json" in q3_queue
    assert shard1.rstrip().endswith(
        "exec scripts/run_deep_dive_q3_after_gate_queue.sh 3"
    )
    assert 'rg -Fq "bash scripts/run_deep_dive_q2_d2_formal_queue.sh"' not in lane2
    assert "selected_branch_fold1_shard0of2_v1/metadata.json" in lane2
    assert 'wait_completed "$selected_shard0"' in lane2
    assert "selected_eval" not in lane2
    assert "disjoint preregistered breadth run IDs" in lane2
    assert "qrels" not in lane2
    assert lane2.rstrip().endswith(
        "exec scripts/run_deep_dive_q3_after_gate_queue.sh 2"
    )


def test_q3_short_backfill_is_fixed_qrels_blind_and_d2_preemptible():
    text = Q3_B20_SHORT_BACKFILL.read_text(encoding="utf-8")
    assert "b26_fold0_v1/metadata.json" in text
    assert "b27_fold0_v1/metadata.json" in text
    assert "b27_fold0_v1/progress.json" in text
    assert "q3_attention_heads_b20_v1" in text
    assert "q3_attention_groups_b20_v1" in text
    assert "q3_mlp_groups_b20_v1" in text
    assert "b27_fraction_below 0.90" in text
    assert "b27_fraction_below 0.25" in text
    assert "b27_fraction_below 0.75" in text
    assert "score_deep_dive_attention_edges.py" not in text
    assert "score_deep_dive_rope.py" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "metrics.json" not in text


def test_q2_b27_partial_backfill_advances_d2_once_and_excludes_long_jobs():
    text = Q2_B27_PARTIAL_BACKFILL.read_text(encoding="utf-8")
    assert "q3_mlp_groups_b20_v1/metadata.json" in text
    assert "q3_postblock_b27_fold0_v1/metadata.json" in text
    assert "q3_postblock_b27_fold0_v1/progress.json" in text
    assert "q2_postblock_b27_fold1_v1" in text
    assert "q2_postblock_fold0_selection_v1/selection.json" in text
    assert "q3_attention_heads_b27_v1" in text
    assert "q3_b27_fraction_below 0.65" in text
    assert "q3_b27_fraction_below 0.90" in text
    assert "--max-wall-seconds 2100" in text
    assert "run_q2_b27_one_attempt" in text
    assert "deep_dive_resume_locks" in text
    assert "run_deep_dive_resume_loop.sh" in text
    assert text.count("score_deep_dive_postblock_sweep.py") == 1
    assert "q3_mlp_groups_b27_v1" not in text
    assert "score_deep_dive_attention_groups.py" not in text
    assert "score_deep_dive_attention_edges.py" not in text
    assert "score_deep_dive_rope.py" not in text
    assert "metrics.json" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_q2_short_backfill_waits_for_rope_and_yields_to_selected_contract():
    text = Q2_SHORT_BACKFILL_AFTER_ROPE.read_text(encoding="utf-8")
    assert "q2_rope_b13_v2/metadata.json" in text
    assert "q2_selected_branch_contract_v1/contract.json" in text
    assert "remaining_q2_fold1_bundles" in text
    assert "safe_before_contract 2" in text
    assert "safe_before_contract 3" in text
    assert "for block in 20 27" in text
    assert "q2_attention_heads_b${block}_v1" in text
    assert "q2_attention_groups_b${block}_v1" in text
    assert "q2_mlp_groups_b${block}_v1" in text
    assert "score_deep_dive_attention_edges.py" not in text
    assert "score_deep_dive_rope.py" not in text
    assert "metrics.json" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_gpu3_q2_breadth_backfill_uses_only_fixed_registered_anchors():
    text = GPU3_Q2_BREADTH_BACKFILL.read_text(encoding="utf-8")
    assert "q3_postblock_fold0_selection_v1/selection.json" in text
    assert "yielding GPU3 Q2 breadth to the Q3 fold-1 lane" in text
    assert text.count("for block in 20 27") == 3
    assert "q2_attention_heads_b${block}_v1" in text
    assert "q2_attention_groups_b${block}_v1" in text
    assert "q2_mlp_groups_b${block}_v1" in text
    assert "run_deep_dive_resume_loop.sh" in text
    assert "score_deep_dive_postblock_sweep.py" not in text
    assert "selected_block" not in text
    assert "score_deep_dive_attention_edges.py" not in text
    assert "score_deep_dive_rope.py" not in text
    assert "metrics.json" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_mlp_formation_extension_lane_is_fixed_qrels_blind_and_smoke_gated():
    text = MLP_FORMATION_LANE.read_text(encoding="utf-8")
    assert "blocks=(13 20)" in text
    assert text.count("blocks=(27)") == 2
    assert "mlp_formation_b${first_block}_smoke_gpu_v1" in text
    assert "--max-rows 1" in text
    assert "maximum_score_identity_delta" in text
    assert "maximum_product_recomposition_low_precision_ratio" in text
    assert "observe_deep_dive_mlp_features.py" in text
    assert "run_deep_dive_resume_loop.sh" in text
    assert "selected_block" not in text
    assert "metrics.json" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_component_necessity_lane_is_parent_gated_and_result_blind():
    text = COMPONENT_NECESSITY_LANE.read_text(encoding="utf-8")
    assert "fold1_negative_transition_reproduced" in text
    assert "registered_confirmatory_branch_localization" in text
    assert "selected_branch_fold1_v1" in text
    assert "mlp_formation_b13_v1/metadata.json" in text
    assert "mlp_formation_b20_v1/metadata.json" in text
    assert "score_deep_dive_component_necessity.py" in text
    assert "--max-requests 8" in text
    assert "run_deep_dive_resume_loop.sh" in text
    assert "metrics.json" not in text
    assert "qrels" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_component_necessity_evaluator_keeps_gate_stops_in_family():
    text = COMPONENT_NECESSITY_EVAL_QUEUE.read_text(encoding="utf-8")
    assert "--q2-gate-contract" not in text  # constructed from the fixed short name
    assert 'args+=("--${short}-gate-contract" "$contract")' in text
    assert 'args+=("--${short}-bundle" "$bundle")' in text
    assert "evaluate_deep_dive_component_necessity.py" in text
    assert "dev_qrels_folds_v1" in text


def test_component_design_synthesis_waits_for_both_registered_parents():
    text = COMPONENT_DESIGN_SYNTHESIS_QUEUE.read_text(encoding="utf-8")
    assert "component_necessity_eval_v1/metrics.json" in text
    assert "d2_selected_branch_synthesis_v1/metrics.json" in text
    assert "synthesize_deep_dive_component_design.py" in text
    assert "score_" not in text
    assert "dev_qrels" not in text
