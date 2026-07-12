# Minimal falsifier design if C12 is ever reopened

Status: design only, not authorized or hash-locked.

A successor may reuse this falsifier only after separately proving a generic,
exact normalization strategy whose end-to-end cost meets the architecture
budget.  It must receive a new candidate fingerprint.

## Stage P0: symbolic and causal integrity

Before optimization:

1. reproduce the hidden-similarity witness in
   `symbolic_reduction_audit.md` numerically in FP64;
2. hold target-logit difference fixed while varying the candidate-specific
   partition increment, and require centred likelihood ratios to change;
3. assert strict causal masking by perturbing `c_i,t:` and proving the predictor
   at `t` is bitwise unchanged;
4. make history and NULL sequences length/mask/position matched;
5. assert candidate permutation equivariance, hidden-write zero sum/norm bound,
   monotone internal exact coordinate, and bitwise no-history fallback.

Failure ends the candidate before learned training.

## Stage P1: construct audit

Use role-balanced synthetic product-token sequences with iid candidate-local
variant marginals.  Construct exact/non-repeat membership by changing history,
never candidate tokens.  Before labels are opened require:

- positive/negative token-position and local-token total variation below frozen
  bounds;
- uniform target position;
- exact membership equals the flag and non-repeat membership is zero;
- at least three query-compatible hard negatives per request;
- a trained history-blind base below a frozen NDCG ceiling.

No category/token-range information may enter model code.

## Stage P2: learned mechanism gate

Compare the primary with prefix hidden similarity, candidate-independent event
LLR, pooled prefix LLR, centred attention, and paired scalar delta under matched
initialization, ranking objective, parameter budget, and candidate set.

The gate is conjunctive:

- positive non-repeat gain in every seed;
- primary advantage over every neighbour, not just base;
- same-checkpoint exact-repeat non-inferiority to the internal item-only path;
- wrong-user, event-shuffle, query-mask, and target-leak diagnostics;
- non-collapsed candidate-centred partition contribution and `t>=2` contribution;
- at least a frozen fraction of transfer candidate orders changed;
- no-history bitwise identity and zero-sum/bounded write;
- measured wall time, peak memory, and throughput within a predeclared multiple
  of centred attention on the same hardware.

## Binding cost falsifier

For the current standard full-vocabulary decoder, reject before P1 because the
symbolic `Omega(C(H+1)TVd)` normalization lower bound is already outside the
intended lightweight envelope.  A future normalization design must declare its
exact likelihood semantics and pass matched numerical error tests; an
approximation cannot silently inherit C12's non-reduction proof.
