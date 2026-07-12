# Exact reparameterization audit

## Subprobability factorization

For one candidate and head, let `w_j>=0` and `sum_j w_j<=1`.  Define

```text
rho = sum_j w_j,
p_j = w_j/rho  if rho>0,
```

with arbitrary `p` when `rho=0`.  Then `w=rho p` and a NULL event receives
`w_NULL=1-rho`.  This factorization is unique away from `rho=0`.

Conversely, for `0<rho<1`, choose NULL logit zero and real-event logits

```text
l_j = logit(rho) + log p_j.
```

Softmax over `[NULL, real events]` returns exactly

```text
alpha_NULL = 1-rho,
alpha_j    = rho p_j.
```

Zeros are represented by `-infinity` masks or a sparse transform.  Therefore
C14 and softmax+zero-NULL have the same attainable weight vectors and the same
head output for every value matrix.

## Numerical construct witness

Take two events with values `v_1=(1,0)`, `v_2=(0,1)`, `rho=0.4`, and
`p=(0.75,0.25)`.  C14 writes `(0.3,0.1)`.  NULL/real unnormalized masses
`(1,0.5,1/6)` normalize to `(0.6,0.3,0.1)` and write the same vector exactly.
The construction applies independently to every candidate and head.

## Ordinary attention times a scalar gate

For each head,

```text
o_i,h = rho_i,h [sum_j p_i,j,h V_h(h_j)].
```

This is definitionally ordinary conditional attention multiplied by a
candidate/head scalar gate.  A gate shared across heads is the familiar scalar
version; head-specific `rho` is per-head output gating.  Placing it before
`W_O` does not change the equivalence because the gate multiplies each head
before concatenation.

If event-specific support factors `s_j` are introduced and weights become
`s_j p_j`, their sum can again be called `rho` and their normalized values the
new `p`.  As long as weights stay nonnegative with total at most one, they are
still one NULL-attention distribution.

## Deterministic postprocessing does not rescue novelty

Candidate centring, a global norm bound, residual addition, and small non-zero
LayerScale receive identical `o` under both parameterizations.  Applying the
same deterministic map after equal attention outputs preserves equality.

## What would break the proof

Signed/vector-valued gates, candidate-coupled column constraints, support-
conditioned value transformations, or non-probability mass conservation could
leave the null-softmax family.  They are not the stated subprobability primitive:

- signed/vector gates become value/output gating and are covered by stronger
  gated-attention neighbours;
- column constraints or multi-role intersections recreate transport/C03;
- an extra support-conditioned `V/W_O` network is a second primitive;
- unconstrained independent event mass becomes sigmoid/screening attention.

No eligible one-primitive version remains after the exact reduction.
