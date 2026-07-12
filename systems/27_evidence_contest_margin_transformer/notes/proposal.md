# C27 proposal — evidence-conditioned antisymmetric contest margins

Status: pre-outcome design; no C27 internal-A/delayed-B label opened.

## Failure-derived hypothesis

C26 proved that wrong histories can alter token-bridge corrections while those
changes remain below ranking margins.  Its post-terminal fit-only audit also
showed a seed-level scale/direction collapse: two scalar heads became nearly
candidate-common, while the larger-scale seed lacked positive clicked
direction.  C27 tests one new hypothesis: **candidate-relative evidence must be
structural in the ranking readout, not learned independently and centered only
afterward**.

## Primitive

A shared compact Transformer contextualizes frozen BGE WordPiece embeddings.
Query-pivot late interaction creates a query-candidate state `c_i` and
candidate-conditioned history-event states; a history Transformer aggregates
the latter into `e_i`.  Three node laws are evaluated in the same computation:

```text
u_i^evidence  = F(c_i * e_i)
u_i^generic   = F(c_i + e_i)
u_i^candidate = F(c_i)
```

For any candidate pair `(i,j)`, a bias-free linear–tanh–linear comparator `O`
is an odd function.  Therefore

```text
delta_ij = delta_max * tanh(O(u_i - u_j))
delta_ji = -delta_ij,  delta_ii = 0.
```

The active non-repeat score is not `D2p + scalar residual`.  It is induced by
pair contests:

```text
p_ij = sigmoid((s_i^D2p - s_j^D2p) + delta_ij)
b_i  = mean_{j != i} p_ij
p^0_ij = sigmoid(s_i^D2p - s_j^D2p)
b^0_i  = mean_{j != i} p^0_ij
s_i  = s_i^D2p + logit(b_i) - logit(b^0_i).
```

With zero evidence contrast, every score equals D2p exactly, including base
ties.  Candidate-common history states cancel before the comparator.  Query
absence/no history returns D2p exactly; any repeat-present request returns the
registered item-only anchor exactly.

## Matched controls

All modes instantiate and execute the same token encoder, query pivots,
history Transformer, node map, odd comparator, pair matrices, and additive
surface with identical parameters and schedules:

- `evidence_contest` — primary multiplicative candidate/history node law plus
  antisymmetric contest readout;
- `generic_contest` — additive candidate/history node law with the same contest;
- `candidate_contest` — history-free query-candidate node law with the same
  contest;
- `additive_node` — primary evidence node reduced to a centered bounded scalar
  residual, the direct C26-style readout control.

Primary must beat all three.  Otherwise the gain belongs to ordinary listwise
comparison, candidate-only token matching, or generic/additive personalization.

## Boundary

C27 does not claim that odd comparators, Bradley–Terry probabilities, soft
Borda, or candidate-set modeling are new.  A positive A0/A1/A2 would authorize
deeper novelty review of their evidence-conditioned composition; a failure
closes this pairwise-margin readout primitive.
