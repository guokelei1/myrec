from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_cli(script_name: str) -> ModuleType:
    path = ROOT / "scripts" / script_name
    module_name = f"test_{path.stem}_command_lineage"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load CLI: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "core_name", "argv"),
    (
        (
            "fit_m2_representation_probe.py",
            "fit_train_representation_probes",
            [
                "fit_m2_representation_probe.py",
                "--standardized-dir",
                "standardized",
                "--activation-bundle",
                "activation",
                "--output-dir",
                "probe",
                "--method-id",
                "q2_recranker_generalqwen",
            ],
        ),
        (
            "evaluate_m2_representations.py",
            "evaluate_m2_representations",
            [
                "evaluate_m2_representations.py",
                "--standardized-dir",
                "standardized",
                "--bundle",
                "full",
                "full_bundle",
                "--probe-model-dir",
                "probe",
                "--output-dir",
                "analysis",
                "--analysis-run-id",
                "m2-representation-analysis",
            ],
        ),
        (
            "evaluate_m2_activation_patches.py",
            "evaluate_m2_patches",
            [
                "evaluate_m2_activation_patches.py",
                "--standardized-dir",
                "standardized",
                "--full-baseline",
                "full_scores",
                "--null-baseline",
                "null_scores",
                "--patch",
                "same_request_full_to_null",
                "13",
                "patch_scores",
                "--output-dir",
                "analysis",
                "--analysis-run-id",
                "m2-patch-analysis",
            ],
        ),
    ),
)
def test_m2_cli_passes_exact_argv_to_core_command_lineage(
    script_name: str,
    core_name: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli(script_name)
    captured: dict[str, object] = {}

    def fake_core(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "test-only"}

    monkeypatch.setattr(module, core_name, fake_core)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()  # type: ignore[attr-defined]
    assert captured["command"] == argv
    assert captured["command"] is sys.argv
    assert "test-only" in capsys.readouterr().out
