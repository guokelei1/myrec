# C09 Frozen Pre-Outcome Gate

Frozen on 2026-07-11 before any cohort, label, qrels, dev metric, GPU run, or
test access.  Thresholds may not be relaxed after seeing an outcome.  Any new
hypothesis requires a new pre-outcome document.

## Authorization boundary

Currently authorized and completed: mathematical audit, primary-source audit,
CPU reference implementation, and synthetic-tensor unit tests.

Not authorized here: standardized records, cohort membership, train labels,
qrels, shared dev evaluator, GPU, checkpoints, score dumps, full training, or
test.  C09 has no GPU/environment/run-prefix allocation.  Later stages require
coordinator authorization and the shared execution protocol.

## G0 — Structural mechanism gate

All conditions are conjunctive:

1. the reduction audit contains an off-diagonal witness against global and
   candidate-local diagonal gates;
2. Q-first mediator is invariant to candidate replacement;
3. C-first mediator is invariant to query replacement;
4. all-pair disagreement returns base scores exactly;
5. no-history and query-masked inputs return base scores bit exactly;
6. candidate permutation equivariance passes;
7. common-mode offsets in either view do not change output;
8. gradients are finite and nonzero for both agreeing views and the candidate
   value path;
9. singleton candidates degenerate to the base;
10. the hand-computed three-candidate result equals `(35/19, 2/5, 0)`;
11. the end-to-end token/history/rank Transformer backpropagates through the
    shared encoder, shared mediator attention, and shared rank Transformer;
12. no dataset/category/query-type branch or forbidden mechanism exists.

**Stop rule:** one failure closes C09 until a new pre-outcome mechanism is
frozen.  Do not repair a mathematical failure by weakening exactness.

Current status: **PASS (14/14 CPU tests; see `test_report.md`)**.

## G1 — Minimal CPU synthetic falsifier (not run)

G1 is the next permissible step only after explicit authorization.  It reads
no repository data and uses no GPU.

### Frozen construction and budget

- seeds: `20260711`, `20260712`, `20260713`;
- per seed: 2,048 generated train requests, 512 generated validation requests,
  8 candidates, latent width 8, history length 6;
- dual-causal generator: base utility is a query-candidate bilinear term and
  the history residual is a product of a query-history compatibility term and
  a candidate-history compatibility term; neither factor alone determines the
  residual sign;
- corruption generators: candidate-history factor flipped, query-history
  factor flipped, shuffled history, and all-pair view disagreement;
- at most 200 optimizer steps per method, one frozen learning rate, no sweep;
- objective coefficients exactly as in `proposal.md`;
- controls: base, Q-first-only, C-first-only, view mean, PoE fusion, global
  scalar gate, diagonal gate, ordinary learned candidate attention,
  constant-value CMA, and parameter-matched enlarged base;
- primary diagnostic: pairwise ordering accuracy on non-tied generated utility;
- total budget: 3 CPU-hours and zero dev evaluations.

### G1 pass criteria

All must hold:

1. CMA exceeds the best base/single-view/scalar/diagonal control by at least
   **5.0 absolute pair-accuracy points in at least 2/3 seeds**, with a
   three-seed mean surplus of at least 5.0 points;
2. CMA exceeds ordinary learned candidate attention by at least 2.0 points in
   at least 2/3 seeds; otherwise the conjunctive construction has no measured
   rent;
3. forced all-pair disagreement produces exactly zero score correction on
   every request;
4. each single-factor flip or shuffled-history corruption removes at least
   75% of CMA's clean surplus over the base;
5. constant-value CMA does not come within 1.0 point of full CMA; otherwise the
   mechanism has collapsed to confidence gating;
6. no-history and query-masked mismatch counts remain zero;
7. all seeds finish with finite losses/gradients and no outcome-dependent
   changes.

**G1 stop rule:** any criterion failure closes C09.  A result tailored to the
dual-causal generator is not evidence for the paper; G1 only establishes that
the primitive is trainable and behaviorally distinct enough to justify one
blind dev screen.

## D0 — Pre-evaluator integrity gate (not authorized)

Before any future dev call:

1. coordinator supplies the frozen dataset version, candidate manifest hash,
   D2p score/control definition, environment, run prefix, and evaluation
   budget;
2. scoring/training code contains no qrels path/import/open and sees blind dev
   records only;
3. candidate IDs and candidate-manifest hash match exactly for every request;
4. the future Transformer base `b` is verified rank-equivalent to frozen D2p on
   all 4,110 no-history requests **without reading labels**; mismatch count must
   be zero;
5. the C09 no-history path is bit-identical to that verified base;
6. configs, seed, package versions, command, and run metadata are frozen;
7. the shared evaluator is called under its external lock and the one-screen
   budget is registered before results are visible.

The local prototype proves only `C09 == b` under missing evidence.  It does not
assume or claim `b == D2p`; failure of D0 item 4 is an immediate stop.

## D1 — Single-seed dev screen (not authorized)

One seed (`20260711`) and at most one shared dev-evaluator call.  No threshold
or control selection may use test.

All pass criteria are conjunctive:

1. overall click NDCG@10 is at least `0.35228301`, i.e. 2% relative above the
   frozen item-only watermark `0.3453755`;
2. on repeat-present requests, C09 minus item-only has point estimate >= 0 and
   paired-bootstrap 95% CI lower bound >= `-0.002` absolute;
3. on the 4,677 non-repeat/history-present requests, C09 exceeds D2p by at
   least 2% relative and the paired-bootstrap 95% CI lower bound is > 0;
4. true-history gain over D2p is significantly greater than wrong-user and
   shuffled-event gains, and each corruption retains at most 25% of the clean
   gain;
5. coarse-only evidence retains at most 25% of the clean gain;
6. query-masked correction mismatch count is zero;
7. all 4,110 no-history requests have zero candidate/rank mismatch against
   D2p;
8. matched global/diagonal gates, Q-first-only, C-first-only, ordinary learned
   attention, and parameter-matched base do not meet all criteria 1--7; if a
   simpler control does, the C09 mechanism claim stops even if quality is high;
9. candidate hashes, score finiteness, deterministic 1,000-request rescore,
   and dev-log append all pass.

**D1 stop rule:** any failure closes C09 with no second dev call.  Passing D1
only requests coordinator review for a separately budgeted multi-seed gate; it
does not authorize full training or test.

## Test lock and claim boundary

Test is forbidden throughout G0/G1/D0/D1.  A future positive claim requires a
new coordinator-authorized multi-seed gate, the common evaluator, equal tuning
budgets, matched capacity/compute, and all protocol controls.  Neither G0 nor a
future G1 pass is a ranking result.
