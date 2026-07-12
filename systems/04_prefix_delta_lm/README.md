# C04 — Counterfactual Prefix-Delta Language Recommender

This directory is the isolated C04 candidate workspace. CPDLR is a local,
fixed-candidate masked-Transformer ranker. It scores the same candidate under a
factual `[query, history, candidate]` prefix and a null-history
`[query, NULL_HISTORY, candidate]` prefix with one shared parameter set. Its
final candidate logit is produced from those two LM logits inside the ranking
path; it never calls an online LLM and never adds an external D2p score at
inference.

The current authorization ends after proposal lock, unit/smoke/train-internal
checks, one seed-20260708 dev screening call, and `notes/final_report.md`.
Full-gate, multi-seed, test, Amazon, and JDsearch work are not authorized.

## Frozen resource boundary

- environment: `/data/gkl/conda_envs/myrec-c04` (`myrec-c04`);
- physical GPU: 3, exposed to code only as `cuda:0`;
- run prefix: `20260710_kuaisearch_c04_`;
- candidate manifest SHA256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- source/config/notes writes: this directory only;
- raw outputs: `runs/20260710_kuaisearch_c04_*`,
  `models/c04_prefix_delta_lm/`, and `artifacts/c04_prefix_delta_lm/`.

## Pre-outcome execution order

Run from repository root. The lock must exist before the first command that
uses GPU model computation.

```bash
/data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/materialize_protocol.py

CONDA_PREFIX=/data/gkl/conda_envs/myrec-c04 \
  /data/gkl/conda_envs/myrec-c04/bin/python -m pytest -q \
  -o cache_dir=systems/04_prefix_delta_lm/.pytest_cache \
  systems/04_prefix_delta_lm/tests

CONDA_PREFIX=/data/gkl/conda_envs/myrec-c04 \
  /data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/freeze_proposal.py
```

After the lock, materialize the train-only anchor and train the four controls
plus the paired candidate. Every GPU command uses the assigned binding:

```bash
CUDA_VISIBLE_DEVICES=3 \
  /data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/materialize_probe.py --device cuda:0

for mode in single_pass paired_no_tangent concat_head static_lora identity_shortcut paired_delta; do
  CUDA_VISIBLE_DEVICES=3 \
    /data/gkl/conda_envs/myrec-c04/bin/python \
    systems/04_prefix_delta_lm/scripts/train_probe.py \
    --mode "$mode" --device cuda:0
done
```

Before the sole primary dev call, run two 1,000-request deterministic label-free
rescores, then the complete label-free score export:

```bash
CUDA_VISIBLE_DEVICES=3 \
  /data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/score_dev.py \
  --limit-requests 1000 --no-diagnostics \
  --output-dir artifacts/c04_prefix_delta_lm/determinism_a

CUDA_VISIBLE_DEVICES=3 \
  /data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/score_dev.py \
  --limit-requests 1000 --no-diagnostics \
  --output-dir artifacts/c04_prefix_delta_lm/determinism_b

CUDA_VISIBLE_DEVICES=3 \
  /data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/score_dev.py
```

The one authorized evaluator call is serialized with the common lock:

```bash
flock tmp/pps_dev_evaluator.lock \
  /data/gkl/conda_envs/myrec-c04/bin/python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c04_prefix_delta_screen_s20260708 \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

Subset comparisons use the shared comparison implementation, not a
candidate-local metric:

```bash
/data/gkl/conda_envs/myrec-c04/bin/python scripts/compare_runs.py \
  --run-a 20260710_kuaisearch_c04_prefix_delta_screen_s20260708 \
  --run-b 20260710_kuaisearch_d2p_text_pop_dev_s20260708 \
  --request-ids systems/04_prefix_delta_lm/protocol/nonrepeat_history_present_request_ids.txt \
  --output systems/04_prefix_delta_lm/notes/screen_nonrepeat_vs_d2p.json

/data/gkl/conda_envs/myrec-c04/bin/python scripts/compare_runs.py \
  --run-a 20260710_kuaisearch_c04_prefix_delta_screen_s20260708 \
  --run-b 20260710_kuaisearch_c5r3_d2s_item_only_dev_s20260708 \
  --request-ids systems/04_prefix_delta_lm/protocol/repeat_present_request_ids.txt \
  --output systems/04_prefix_delta_lm/notes/screen_repeat_vs_item.json

/data/gkl/conda_envs/myrec-c04/bin/python \
  systems/04_prefix_delta_lm/scripts/audit_screening.py
```

No command in this directory accepts a qrels path. Training rejects dev/test,
qrels, and evaluator-metric paths mechanically. Scoring checks that every dev
candidate is label-free. Test records, qrels, and test metrics are outside the
candidate's interface.
