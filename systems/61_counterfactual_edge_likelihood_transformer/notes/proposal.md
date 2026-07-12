# C61 proposal — counterfactual edge likelihood

Status: pre-outcome.  C60 established a safe residual interface: one-sided
adjacent base-margin transport nearly preserved the strong base and made true
history significantly better than wrong/history-free evidence.  It failed
because fixed cosine evidence did not identify actual base errors.  C61 changes
the evidence estimator, not C60's neighborhood, capacity, or aggregation.

## Architecture primitive

Candidates are canonically ordered by the shared strong base.  A shared
Transformer edge verifier `F(q, c_low, c_high, H)` scores the proposition that
the lower-base candidate should beat its adjacent higher-base neighbor.  It is
made exactly antisymmetric and history-counterfactual:

```text
A(H; low, high) = 1/2 [F(q,low,high,H) - F(q,high,low,H)]
lambda = A(H; low, high) - A(NULL; low, high)
logit P(low > high | q,H) = -base_gap + lambda.
```

The same parameters evaluate factual and NULL history.  Therefore no history
gives `lambda=0` algebraically; query/candidate shortcuts cancel before the
write.  Candidate swap flips the sign exactly.  Frozen BGE contextual tokens
are projected and refined by a trainable token Transformer; an ordered edge
token cross-attends query-conditioned history-event states.  The LM/Transformer
is the ranking core, not an offline feature followed by an MLP.

The final edge probability opens exactly C60's registered one-sided transport:
score mass may move only toward the challenger, each edge is capped by its
original base gap, and request score sum is conserved.  Exact recurrence uses
the item-only anchor; missing query/history uses the base.  No dataset,
category, query-type, rank-k, threshold, scale, or router branch exists.

## Training target

Only adjacent pairs with unequal fit labels contribute binary cross-entropy:
the target is one exactly when the lower-base candidate is positive and the
higher candidate is not.  The base prior `-gap` is inside the pair likelihood;
the Transformer learns a residual log-likelihood ratio rather than a complete
ranking score.  All ranks are weighted equally.  Training uses all 6,000
already exposed C26-fit requests, two fixed epochs, three seeds, and no
candidate sampling.

## Matched controls

Four equal-capacity/equal-schedule modes share initialization:

- `counterfactual_edge`: primary factual-minus-NULL antisymmetric ratio;
- `factual_edge`: removes NULL subtraction;
- `ordinary_candidate_attention`: scores each candidate after ordinary
  history attention and takes the adjacent difference;
- `candidate_only_edge`: antisymmetric NULL/query-candidate verifier.

The primary checkpoint is additionally scored with its NULL subtraction
removed, with wrong history, and with C60's fixed semantic edge law.  A gain
must survive every control; otherwise it is generic pairwise capacity,
ordinary attention, or the safe transport alone.

## Staging

G0 is label-free and precedes training: contextual coverage/hashes, exact
antisymmetry, exact factual=NULL zero, no-history/repeat anchors, conservation,
edge capacity, candidate permutation, and a hand-constructed nonzero edge
must all pass.  Only then is the execution lock frozen and exposed fit labels
may train.

Fresh internal-A labels remain closed through training and A0.  A0 requires
all-seed convergence, activity, true/wrong-history Top-10 sensitivity,
same-checkpoint NULL-ablation sensitivity, exact fallbacks, determinism,
permutation, conservation, and capacity.  Only passed A0 opens 1,200 A labels.
A1 requires positive-CI NDCG gains over base, wrong history, same-checkpoint
ablation, all trained controls, and fixed C60, with every seed and hash fold
positive.  Failure closes C61 without epoch/loss/width/scale/edge/depth rescue.
