# C01 Frozen Gate Protocol

Status: frozen before any C01 unit-derived model statistic, train-internal
outcome, dev score, or dev metric.  Unit tests of pure algebra/invariants are not
outcomes; all learned smoke/internal checks occur after the proposal lock.

## Scope, seed, and budget

- Candidate: C01 / CECT.
- Environment: `myrec-c01`; physical GPU 0; all GPU commands bind
  `CUDA_VISIBLE_DEVICES=0` and code sees only `cuda:0`.
- Seed: `20260708` for Python, NumPy, and PyTorch.
- At most two implementation attempts and at most 8 cumulative A40 GPU-hours.
- One primary dev evaluator call at most.
- No test record, test qrel, test score, or test metric access.
- Run prefix: `20260710_kuaisearch_c01_`.

## Data and isolation

- Shared standardized interface:
  `data/standardized/kuaisearch/v0_lite/records_{train,dev}.jsonl` only.
- Candidate manifest SHA256 must equal
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`
  before training, scoring, deterministic rescore, and evaluation.
- Training/scoring source is mechanically scanned for `qrels_`,
  `records_test`, and sibling candidate paths; any match outside explicit
  deny-list assertions fails integrity.
- Dev is label-free scoring only.  Only `scripts/evaluate_scores.py`, under the
  common `flock`, may read dev qrels.
- Frozen D2p seed-20260708 dev scores are the no-history anchor.  C01 never
  changes candidates or request membership.

The retained positive train requests are split by frozen record order:

- fit: first 80%, indices `[0, 77551)`;
- counterfactual calibration: next 10%, `[77551, 87245)`;
- internal screen: final 10%, `[87245, 96939)`.

No internal-screen label is used for fitting, threshold selection, or retry.

## Frozen model and optimization

- Frozen BGE states: `BAAI/bge-small-zh-v1.5`, dimension 512.
- TET: `d_model=96`, 2 layers, 4 heads, FFN 192, dropout 0.10, maximum
  20 most-recent history events.
- Per fit request: all clicked candidates plus at most 15 deterministic
  hash-selected unclicked candidates.
- Stage 1: two epochs, multi-positive listwise ranking plus robust multi-twin
  margin loss; AdamW, LR `3e-4`, weight decay `1e-4`, gradient clip 1.0.
- Robust twin margin `mu_cf=0.20`; log-sum-exp temperature 0.20;
  counterfactual loss weight 0.50.
- Calibration false-admission target `alpha_cf=0.10`; finite-sample upper
  quantile exactly as defined in `mechanism_fingerprint.md`.
- After calibration, TET and certificate head are frozen.
- Stage 2: one epoch for value/readout only with the frozen threshold; same fit
  requests, LR `5e-4`.
- Contract blend `beta=0.30`; transfer scale is sigmoid-constrained to `[0,1]`.
- Plain target-attention control: identical architecture, inputs, exact atom,
  train split, sampled candidates, optimizer steps, and parameter count; no twin
  loss and ordinary softmax event weights.

An implementation retry is allowed only for an exception, NaN/Inf, OOM at the
frozen batch size, or a proven invariant/serialization bug.  It may change batch
size only (not effective examples, steps, model, threshold, or loss).  A weak
metric, weak certificate, or failed falsifier is not a retry condition.

## Train/internal falsifier thresholds

All ranking metrics use the shared `myrec.eval.metrics` implementation.  Paired
bootstrap uses 2,000 resamples and seed `20260708`; these are internal-only
diagnostics, not paper statistics.

1. **Protected recurrence.** On internal requests containing an exact-repeat
   candidate, CECT minus the D2p+exact-atom item-only control must be at least
   `-0.0020` absolute NDCG@10 and its 95% CI lower bound at least `-0.0040`.
2. **Non-repeat transfer.** On history-present internal requests with no exact
   candidate, CECT minus the D2p base must be at least `+0.0020` absolute
   NDCG@10 and the paired 95% CI lower bound must be strictly above zero.
3. **Counterfactual rejection.** On that same non-repeat surface, each of
   wrong-user, event-shuffled, query-masked, and coarse-only rescoring must
   recover at most 25% of CECT's true-history NDCG gain over D2p.  Each twin's
   hard certificate-admission rate must be at least 30% relatively below the
   true rate.  If the true gain is non-positive, this item fails without a ratio.
4. **No-history contract.** For every internal and dev no-history request,
   maximum absolute score difference from D2p is `0.0`, rank mismatches are 0,
   and personalized delta is exactly zero.
5. **Non-collapse and order sensitivity.** On non-exact, history-present
   internal candidate-events: true hard-admission rate must be in `[0.02,0.50]`,
   certificate-energy standard deviation at least `0.05`, true-minus-pooled-twin
   mean energy at least `0.05`, and shuffled-event mean admitted mass at least
   20% relatively below true admitted mass.
6. **C01-specific matched control.** On the internal non-repeat surface, CECT
   must exceed the parameter-matched plain target-attention Transformer by at
   least `+0.0020` NDCG@10 with paired 95% CI lower bound above zero.  Exact
   trainable parameter counts must match exactly.

All six internal items must pass before dev scoring.  Failure means `stop`
without dev evaluation; it cannot be repaired by adding components or changing
subsets/thresholds.

## Smoke and determinism gates

- Unit tests must cover: exact atom admission, quantile indexing, empty-history
  exact fallback, twin construction, masked padding, parameter matching,
  candidate hash rejection, finite score output, and no forbidden path access.
- A CPU/tiny-tensor smoke must produce finite loss and nonzero gradients through
  Transformer attention and both certificate/value heads.
- GPU smoke must complete one optimizer step and show finite loss.
- The frozen checkpoint/config must rescore the first 1,000 dev requests twice;
  request/candidate keys and float scores must be byte-identical, with maximum
  absolute difference 0.0.

## One-call dev screening

Primary run ID:
`20260710_kuaisearch_c01_cect_screen_s20260708`.

Before the evaluator call, scores must contain exactly 575,609 finite rows and
12,229 requests, match every manifest candidate, preserve D2p exactly on all
4,110 no-history requests, and pass deterministic rescore.

The single aggregate dev screening survives only if:

- NDCG@10 is at least `0.3430873589` (the seed-20260708 item-only waterline
  `0.3450873589` minus the frozen `0.0020` screening tolerance);
- it exceeds seed-matched D2p `0.3238158367` by at least 2% relative; and
- all integrity and internal falsifier items already passed.

This tolerance is a **screening stop-loss**, not a paper win or the full common
gate.  Request-slice dev claims, significance versus item-only, and additional
control evaluations are deferred to a separately authorized full design gate.

The only evaluator command is:

```bash
flock tmp/pps_dev_evaluator.lock \
  CONDA_ENVS_PATH=/data/gkl/conda_envs CUDA_VISIBLE_DEVICES=0 \
  conda run -n myrec-c01 python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c01_cect_screen_s20260708 \
  --candidate-manifest \
    data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

The evaluator must append exactly one matching line to
`reports/dev_eval_log.jsonl`.

## Frozen interpretation

- All internal items pass and dev survives: `advance-to-full-gate`; stop after
  the C01 final report and request coordinator authorization/budget.
- Integrity passes but a mechanism item fails before dev: `stop` (no evaluator).
- Internal passes but aggregate dev misses stop-loss: `pivot-before-more-dev` or
  `stop`, with no additional dev call and no post-hoc module/subset/threshold.
- Plain attention matches CECT or the certificate is constant: C01's mechanism
  claim is `reducible`, regardless of aggregate ranking quality.
