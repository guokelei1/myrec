# C28 proposal — margin-local evidence contest

Status: pre-outcome design; no C28 internal-A/delayed-B label opened.

## Hypothesis

C27 established that antisymmetric evidence contests can carry complete-order
and aggregate top-10 margins, but uniform all-pair Borda aggregation diluted
wrong-history effects at top-set boundaries.  C28 tests one new primitive:
**comparison locality should be induced by continuous base-score uncertainty,
not by an evaluation rank or a learned post-hoc scalar**.

## Primitive

Token encoding, query-pivot evidence states, node laws, and odd pair comparator
are unchanged from C27.  For request-zscored D2p scores, define

```text
g_ij = s_i^D2p - s_j^D2p
w_ij = exp(-abs(g_ij)) / sum_{l != i} exp(-abs(g_il))
p_ij = sigmoid(g_ij + delta_ij)
p0_ij = sigmoid(g_ij)
s_i = s_i^D2p + logit(sum_j w_ij p_ij)
                  - logit(sum_j w_ij p0_ij).
```

The kernel scale is fixed to one because `g` is request-standardized; it is not
learned or swept.  `delta_ji=-delta_ij` remains exact.  Zero evidence returns
D2p bitwise, including ties.  The kernel names no top-k cutoff and applies at
all ranks.  Query/no-history and repeat contracts remain exact.

## Matched controls

Every mode executes all local/uniform contests and node laws with the same
parameters and schedule:

- `local_evidence_contest` — primary multiplicative evidence nodes plus fixed
  margin-local kernel;
- `uniform_evidence_contest` — exact C27 all-pair aggregation control;
- `local_generic_contest` — additive candidate/history node law;
- `local_candidate_contest` — history-free query-candidate node law;
- `additive_node` — centered independent scalar residual.

Primary must beat all four in A1/A2.  A0 must first pass the same all-seed
wrong-history top-10 contract that closed C27.

## Boundary

Uncertain-pair selection, pairwise ranking, Bradley–Terry models, kernel
weighting, and hard-negative ranking are prior art.  C28 claims no global
novelty.  A positive staged result would only authorize deeper review of fixed
margin-local inference aggregation for token-grounded personalized evidence.
