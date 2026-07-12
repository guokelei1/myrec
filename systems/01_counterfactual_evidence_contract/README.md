# C01 — Counterfactual Evidence-Contract Transformer

This directory is the isolated workspace for candidate C01.  It implements the
minimal, preregistered screening probe for the **Counterfactual Evidence-Contract
Transformer (CECT)**.  CECT is not yet a validated proposed system.  Its purpose
is to test whether an event-level, counterfactual-calibrated Transformer can add
history value beyond the protected exact-recurrence atom without collapsing to
ordinary target attention.

## Boundaries

- Environment: `myrec-c01` under `CONDA_ENVS_PATH=/data/gkl/conda_envs`.
- Physical device: GPU 0 only, always bound with `CUDA_VISIBLE_DEVICES=0`.
- Run prefix: `20260710_kuaisearch_c01_`.
- Seed: `20260708`.
- Shared source, data, evaluator, manifests, baselines, reports, and the other
  candidate workspaces are read-only.
- Training and scoring never read dev/test qrels.  Test records and test qrels
  are out of scope.
- At most one primary dev evaluator call is authorized.  Passing that screening
  only licenses a request for the later full design gate; it does not license
  multi-seed or cross-dataset training.

The frozen design is in `notes/proposal.md`; the exact operator and nearest-
neighbor boundary are in `notes/mechanism_fingerprint.md` and
`notes/nearest_neighbors.md`; all thresholds and stop rules are frozen in
`notes/gate_protocol.md`.  `notes/proposal_lock.json` binds those files and the
screening config before any C01 outcome is produced.

## Minimal execution

From the repository root:

```bash
export CONDA_ENVS_PATH=/data/gkl/conda_envs
export CUDA_VISIBLE_DEVICES=0

conda run -n myrec-c01 python -m pytest -q \
  systems/01_counterfactual_evidence_contract/tests

conda run -n myrec-c01 python \
  systems/01_counterfactual_evidence_contract/train/run_probe.py \
  --config systems/01_counterfactual_evidence_contract/configs/screening.yaml

conda run -n myrec-c01 python \
  systems/01_counterfactual_evidence_contract/train/score_probe.py \
  --config systems/01_counterfactual_evidence_contract/configs/screening.yaml

conda run -n myrec-c01 python \
  systems/01_counterfactual_evidence_contract/train/check_determinism.py \
  --config systems/01_counterfactual_evidence_contract/configs/screening.yaml

flock tmp/pps_dev_evaluator.lock \
  conda run -n myrec-c01 python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c01_cect_screen_s20260708 \
  --candidate-manifest \
    data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

The evaluator command is run only after unit, smoke, internal-gate, integrity,
and deterministic-rescore checks pass.  Raw checkpoints, scores, and logs remain
under `models/c01_*`, `artifacts/c01_*`, and `runs/20260710_kuaisearch_c01_*`.
