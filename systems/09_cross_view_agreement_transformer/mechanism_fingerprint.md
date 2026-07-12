# C09 Mechanism Fingerprint

This fingerprint is frozen before any synthetic or dev outcome.

| Axis | C09 value |
|---|---|
| Primitive | Conjunctive Margin Attention (CMA) |
| Backbone | one end-to-end token/history/rank Transformer |
| View count | two deterministic attention-path views |
| Parameter relation | token encoder, history encoder, mediator MHA, rank Transformer, and score head all shared |
| Q-first barrier | history mediator sees query + history, never candidates |
| C-first barrier | each history mediator sees candidate + history, never query |
| Agreement object | ordered pairwise margins of history residual logits, not raw logits or probabilities |
| Conjunction algebra | positive parallel sum `a_+ b_+/(a_+ + b_+)` |
| Intervention locus | candidate-to-candidate contrast attention matrix inside the ranker |
| Value path | off-diagonal `W_V(z_i-z_j)` base-state contrasts |
| Fallback | bit-exact base on no history, query mask, or all-pair disagreement |
| Set symmetry | candidate permutation equivariant |
| Score symmetry | invariant to candidate-common offsets in either view |
| Training safeguard | separate train-only rank losses; no agreement-reward loss |
| Complexity | `O(N^2 d)` CMA, three shared rank passes |

CMA is a mask-construction primitive inside ordinary set attention.  It is not
claimed to be irreducible to arbitrary attention.  Irreducibility is claimed
only against the frozen global-scalar and candidate-local diagonal gate
classes, using the witness in `reduction_audit.md`.

## Positive identity tests

C09 remains C09 only if all are true:

1. the two mediators pass their information-barrier interventions;
2. history enters the final score only through the pair matrix `K`;
3. `K_ij>0` iff both residual margins for `(i,j)` are positive;
4. the value update has a measurable off-diagonal candidate Jacobian;
5. no-history and query-masked requests return the base exactly;
6. candidate permutations commute with the model;
7. the matched single-view and diagonal-gate controls are run before a positive
   mechanism claim.

## Negative identity tests

Any of the following is a fingerprint collision and forces a stop or a new
pre-outcome proposal:

- averaging, summing, voting, or multiplying the two view predictions;
- a fixed or learned router choosing a view/model;
- a request-level or candidate-level scalar multiplying a local residual;
- an edge-flow divergence, Hodge decomposition/projection, graph potential, or
  rank reconstruction from pairwise flows;
- a hypernetwork/hyperadapter, prefix/prompt, or transport/Sinkhorn operator;
- dataset, category, query-type, or hand-authored cohort rules;
- separate view encoders or heads whose parameter diversity, rather than the
  information barrier, creates agreement;
- an offline LLM feature followed by an MLP;
- a prompt-only or fixed-score reranker.

## Collision-resistant short name

Use **CMA: shared-path residual-margin conjunctive attention**.  Do not use the
broad labels “multi-view agreement,” “consensus ranking,” or “adaptive
personalization” as novelty claims; those families predate C09.
