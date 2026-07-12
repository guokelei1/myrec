from __future__ import annotations

from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import atomic_json, load_config, proposal_sources, sha256_file, timestamp, verify_inputs  # noqa: E402


def main() -> None:
    config = load_config(SYSTEM_ROOT / "configs/diagnostic.yaml")
    verify_inputs(config)
    c71 = load_config(REPO_ROOT / config["paths"]["c71_config"])
    if config["operator"] != c71["operator"] or config["mechanical_gate"] != c71["mechanical_gate"]:
        raise RuntimeError("C72 operator/mechanical settings drifted from C71")
    for key, value in config["evaluation"].items():
        if key == "bootstrap_seed":
            continue
        if c71["evaluation"].get(key) != value:
            raise RuntimeError(f"C72 evaluation setting drifted from C71: {key}")
    target = REPO_ROOT / config["paths"]["proposal_lock"]
    value = {
        "candidate_id": "c72",
        "created_at": timestamp(),
        "decision": "freeze_exposed_fit_same_formula_diagnostic",
        "design_sha256": {name: sha256_file(path) for name, path in proposal_sources(config).items()},
        "claim_boundary": {"fresh": False, "formulation_only": True, "dev_test_qrels": False},
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT)); print(sha256_file(target))


if __name__ == "__main__":
    main()
