# C03 Frozen Screening Protocol

Status: frozen before any C03 dev outcome.

## Fixed identity and budget

- Candidate: `c03`
- Seed: `20260708`
- Environment: `myrec-c03`
- Physical GPU: 2 only; process-visible device `cuda:0`
- Primary run ID: `20260710_kuaisearch_c03_tctt_screen_s20260708`
- Candidate manifest:
  `data/standardized/kuaisearch/v0_lite/candidate_manifest.json`
- Required manifest SHA256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`
- Maximum implementation attempts: 2 (debug retries included)
- Maximum cumulative A40 time: 8 GPU-hours
- Primary dev evaluator calls: exactly 1
- Additional dev evaluator calls: 0 without a pre-outcome coordinator amendment
- Full gate, multi-seed work, test, Amazon-C4, and JDsearch: not authorized.

Training and scoring accept only `records_train.jsonl`, label-free
`records_dev.jsonl`, the candidate manifest, and registered baseline score
files.  Any path whose basename contains `qrels` or `test` is rejected before
opening.  Test records, qrels, and metrics remain unread.

## Frozen config and scores contract

The screening configuration is
`configs/c03_screening.yaml`.  Score rows contain exactly
`request_id`, `candidate_item_id`, `score`, and `method_id`; scores must be
finite.  Every manifest request and candidate must occur exactly once.  Scoring
asserts the manifest SHA256 before reading records.

The external D2p seed-20260708 score is a protected skip contract.  The model
does not transport or route scalar baseline scores: transport is computed from
Transformer hidden states.  Final score is D2p plus the request-centered hidden
state residual.  With no history, that residual is exactly zero.

## Numerical/unit gate

All must pass before training:

1. hand-computed 1x1 augmented plan has the expected conserved marginals;
2. every augmented plan has maximum row/column marginal error `<= 1e-5` in
   float32 (`<= 1e-10` in the float64 reference test);
3. trusted mass and null mass are finite and in `[0,1]` within `1e-6`;
4. padding cannot receive real evidence mass;
5. no-history raw/final residual is exactly zero;
6. all trainable-parameter gradients in the tiny loss test are finite;
7. a protected exact-identity atom cannot reduce `h↔c` real mass in the
   hand-constructed monotone test.

Any NaN/Inf, conservation failure, or no-history mismatch stops C03 before dev.

## Train/internal probe

The deterministic train subset is selected by request-ID hash before labels
are examined, capped as frozen in the config, and split by request hash into
fit/internal-validation partitions.  Candidate subsampling retains all clicked
candidates and hash-selects negatives.  No dev outcome influences training.

The main operator and four parameter-identical degenerations are diagnosed on
the same internal-validation examples.  For clicked, history-present,
non-repeat examples:

- primary mean score drop (`true - corrupt`) must be positive for each of
  wrong-user, shuffle, query-mask, and coarse-only;
- primary mean null increase (`corrupt - true`) must be at least `0.02` for at
  least three of four corruptions and nonnegative for all four;
- the minimum primary corruption score drop must exceed the corresponding
  minimum of both `softmax` and `no_null` by at least `0.01`;
- `no_cycle` must lose at least `0.01` of query-mask or shuffle selectivity;
- mean trusted mass on exact-repeat examples must be at least `0.50` and their
  mean centered residual must be nonnegative;
- all no-history residuals must remain exactly zero.

These are mechanism diagnostics, not paper metrics.  Failure is recorded; it
cannot be repaired by changing thresholds or adding a component.  The round
still permits the single locked screening call if implementation integrity and
score determinism pass, because this prompt requires one preregistered screen;
an internal failure makes the final recommendation no stronger than `stop`.

## Label-free corruption diagnostics on dev inputs

Before the evaluator call, exactly 1,000 requests are selected by the fixed
hash rule without labels.  Primary scores/mass are recomputed for:

- deterministic different-user history;
- deterministic event permutation;
- query mask;
- history text reduced to the deepest available category only.

For a screening survivor, corruptions must increase mean null mass by at least
`0.02` for at least three of four controls, with none decreasing it by more
than `0.005`.  No corruption score is sent to the evaluator.

## Determinism and integrity before dev

After the config and checkpoint are frozen:

1. score the same first 1,000 manifest requests twice;
2. require byte-identical score files and identical diagnostics;
3. require 100% candidate coverage and the frozen manifest hash;
4. require no-history primary scores to be byte-identical floating-point values
   to seed-matched D2p scores;
5. record checkpoint/config/source hashes and GPU time in run metadata.

## Single-seed dev screening adjudication

Run the shared evaluator once under `flock`.  The evaluator alone reads dev
qrels and produces `metrics.json` plus `per_request_metrics.jsonl`.  Subset
adjudication may only aggregate those evaluator-produced request metrics using
request IDs derived from label-free dev records.

All conditions are required for `advance-to-full-gate`:

1. integrity, unit, train/internal, corruption, and determinism gates pass;
2. overall NDCG@10 is at least the seed-20260708 item-only value
   `0.345087358948813`;
3. on repeat-present requests, primary minus seed-matched item-only mean
   evaluator NDCG@10 is `>= 0`;
4. on the 4,677 non-repeat history-present requests, primary minus seed-matched
   D2p mean evaluator NDCG@10 is `>= 0.002`, and a 10,000-sample paired
   bootstrap 95% interval computed from evaluator-produced request metrics has
   lower bound `> 0`;
5. all 4,110 no-history requests have zero score/rank mismatch against D2p;
6. the primary operator is not behaviorally reproduced by softmax/no-null/no-cycle
   in frozen internal and corruption diagnostics.

If integrity fails, the run is invalid.  If a scientific threshold fails,
recommend `stop`; do not inspect test, alter thresholds, select a subset, or add
modules.  A survivor only recommends `advance-to-full-gate`; it does not
authorize that gate or full training.

## Locked commands

```bash
export CONDA_ENVS_PATH=/data/gkl/conda_envs
export CUDA_VISIBLE_DEVICES=2

conda run -n myrec-c03 python -m pytest -q \
  systems/03_triadic_transport_transformer/tests

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml prepare

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml train

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml diagnose

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml determinism

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml score

flock tmp/pps_dev_evaluator.lock \
  conda run -n myrec-c03 python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c03_tctt_screen_s20260708 \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml adjudicate
```
