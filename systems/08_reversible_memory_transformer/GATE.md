# Locked pre-outcome gate

Lock date: 2026-07-11. The thresholds below were fixed without repository data,
labels, qrels, dev metrics, or GPU execution. A failed stage closes C08; it does
not authorize a revised primitive under the same candidate ID.

## G0 — structural CPU contract (executed)

Command:

```bash
CUDA_VISIBLE_DEVICES="" pytest -q test_reversible_memory.py
```

All of the following must pass in one invocation:

1. write followed by exact inverse has maximum float64 error `< 1e-12`;
2. the coupling Jacobian determinant differs from one by `< 1e-10`;
3. aligned history/probe support gives loop norm `> 1e-3`, while disjoint
   support gives norm `< 1e-12`;
4. an endpoint-collision witness has `Wz0 == z0` exactly but loop norm `> 0.05`;
5. loop and ordinary controls have identical parameter names, shapes, and count;
6. empty history gives an exact-zero residual and bitwise query-only scores;
7. candidate permutation error is at most `1e-6`;
8. a candidate-common residual is removed before scoring, and a common score
   offset leaves ranks unchanged;
9. reversing two overlapping history events changes the loop by `> 1e-4`;
10. two optimizer steps have finite nonzero gradients through both Transformer
    blocks, history axes, query-conditioned probe, and memory readout.

Frozen result: **pass, 8 pytest cases**. Several cases cover more than one
contract. Exact observed results are in `SYNTHETIC_REPORT.md`.

## G1 — learned conditional synthetic falsifier (not executed)

This is the only next action currently recommended. It must generate all tensors
in memory and must not read any repository dataset, record, score, or qrels file.

Frozen setup:

- seeds: `20260711`, `20260712`, `20260713`;
- candidates per request: 8; history length: 8; evidence width: 16;
- 4,096 generated train requests and 1,024 independently generated evaluation
  requests per seed;
- at most 400 optimizer steps, one frozen config, no sweep;
- same lower/upper Transformer, tokenizer-free latent inputs, optimizer, step
  count, parameter count, and initialization for all trainable controls;
- primary synthetic measure: top-1 ranking accuracy on supported non-repeat
  requests; secondary: exact-repeat accuracy and mean score margin;
- controls: RWPU, matched ordinary endpoint memory, same-backbone history
  cross-attention, and a parameter-matched extra-FFN backbone;
- corruptions: wrong history, shuffled event order, query mask, and axes with
  disjoint support.

Pass requires all conditions in all three seeds:

1. exact-repeat accuracy is no worse than the deterministic item-recurrence
   control by more than 1 percentage point;
2. supported non-repeat top-1 accuracy exceeds the best trainable control by at
   least 5 absolute percentage points;
3. RWPU exceeds its ordinary endpoint-memory control by at least 3 points;
4. the true supported score margin is positive, and each corruption retains at
   most 25% of that margin;
5. empty-history scores are bitwise equal to the paired query-only forward;
6. candidate permutations produce permuted scores with maximum error `<=1e-6`;
7. no NaN/Inf and no failed two-step gradient audit.

Any failure means **STOP: no standardized data, no dev evaluator, no GPU**.

## G2 — one real-data dev screening (conditionally frozen, not authorized)

G2 may be scheduled only after G1 passes and the coordinator assigns an isolated
environment, physical GPU, run prefix, and shared-evaluator slot. Until then its
budget is zero.

Preconditions:

- integrate inside the exact query-only Transformer scoring function used by
  the frozen D2p anchor; freeze base weights for this gate;
- scoring/training reads label-free standardized records only;
- assert the candidate-manifest hash before score export;
- qrels are accessible only to the serial shared evaluator;
- run train/internal smoke first, then exactly one single-seed dev screening;
- seed `20260708`; test remains locked; no additional dev call.

Screening pass requires every item:

1. overall click NDCG@10 is at least `0.35228301` (2% relative above the frozen
   item-only mean `0.3453755`); this is a screen, not a paper claim;
2. on repeat-present requests, paired NDCG@10 is non-inferior to item-only with
   a frozen absolute margin of `0.002`;
3. on the frozen 4,677 non-repeat/history-present requests, NDCG@10 is at least
   2% relative above D2p and the paired-bootstrap 95% CI lower bound is `>0`;
4. true history beats wrong-user, shuffled-event, query-masked, and coarse-only
   diagnostic inputs with paired-bootstrap lower bound `>0`; each corrupted
   improvement over D2p is less than half the true improvement;
5. on all 4,110 no-history requests, score mismatch count, rank mismatch count,
   and per-request metric mismatch count versus the frozen D2p anchor are zero;
6. exported request/candidate pairs exactly match the candidate manifest and
   the one dev evaluation is appended by the shared evaluator.

Failure closes C08 before full implementation/training. Passing G2 would only
authorize the already-required multi-seed design gate; it would not authorize a
test run or a paper claim.
