# C39 pre-outcome implementation review

## Verdict

C39 has passed its synthetic/operator design gate and is suitable for proposal
locking and the bounded Amazon train-internal gate. It is not yet a validated
architecture, cross-domain result, or novelty claim.

## Evidence before real labels

- `reports/pps_c39_design_gate.json` has SHA-256
  `f338f4a30caa1c384701f7b4c0b7875dcc7980011efba1b2df6b100efb7c5591`.
- D0 passed 12/12 checks on physical GPU 3. Five modes had identical 3,648
  synthetic-smoke parameters and identical initial state hash; halfspace
  violation was `1.90e-8`; permutation and three fallback errors were zero;
  every projection/FFN parameter group received finite nonzero gradients.
- D1 passed 8/8 checks for all three fixed seeds on a randomly rotated,
  same-pooled-value witness. The primary gain was `+0.294103` NDCG@10 per seed.
  This is explicitly an operator-capacity witness, not real-data evidence.
- The real-gate implementation has 12 passing tests covering algebra, same-
  aggregate separation, paired capacity/initialization, exact masks,
  permutation, score-ray equivalence, gradients, predecessor-role isolation,
  label-free feature collection, role-scoped labels, and full CPU train/score.
- Formal C39 selection contains exactly 6,000 C38-fit training requests and
  1,200 A requests from C38-unused, with 399 reserve requests. Overlap with
  C38 internal-A/delayed-B/escrow is zero. Wrong donor coverage and same-length-
  bin matching are both 100%, with zero same-user donors.
- A real CUDA smoke on 32 label-free Amazon requests produced finite 101-way
  corrections for all modes. Each real model has exactly 197,376 parameters;
  the primary reached `Q/K/V/W_O` and FFN-down gradients. Five-mode inference
  over the 32 requests took 0.188 seconds on one A40, so the bounded gate is
  comfortably inside the execution budget.

## Architecture conformance

- The ranking core is a multi-head Transformer cross-attention/FFN block over
  frozen LM states, not an embedding-fed MLP or fixed-score router.
- The common global write is standard unprojected query-attended history value;
  the only new primitive is the candidate/event pre-aggregation halfspace
  projection inside `V/W_O`.
- All modes instantiate the same learned projections and FFN. No pair MLP,
  learned gate, target-category feature, candidate scalar head, tangent
  projection, or dataset/query-type branch exists.
- `ray_only` has the primary's same immediate nonnegative linear score
  contribution but discards score-neutral vector content. Beating it is the
  binding test that representation formation, rather than a renamed scalar
  boost, pays rent.
- No-history/query and repeat contracts are implemented as early exact returns.
  Candidate permutation equivariance follows from set means and per-candidate
  operations and is tested directly.

## Known limitations

1. The design gate was deliberately constructed for the primitive. It cannot
   establish that Amazon histories contain the required eventwise structure.
2. BGE is frozen in the minimal gate. A pass would authorize a fuller LM/PEFT
   implementation only after an independent KuaiSearch confirmation.
3. The value certificate is local to the `V/W_O` write before the shared FFN;
   only A1 can establish that this constraint improves final ranking.
4. A generic edge network can approximate the operator. C39's claim is the
   fixed KKT law and its exact certificate, which must outperform raw,
   post-pool, ray-only, and global controls to pay mechanism rent.
5. Amazon-C4 has one known positive per request; unobserved relevance applies
   equally to all modes and remains a secondary-track limitation.

Proposal lock, label-free feature encoding, G0, execution lock, three GPU
seeds, A0, and A1 must execute in that order. Internal-A labels may not open
before every A0 check passes; reserve/dev/test remain closed throughout.
