# C09 Mathematical Reduction Audit

Decision: **conditionally pass formulation; do not claim function-class
novelty**.

The audit is first because an agreement mechanism can easily be an ordinary
gate with new notation.  C09 proceeds only under a falsifiable architectural
definition of “ordinary scalar gate.”

## 1. The comparison classes

Let `b in R^N` be the no-history candidate logits and let `R in R^(N x d)` be
candidate-local history residual states.

An ordinary global scalar gate has the form

```text
s = b + alpha(x) r,                  alpha(x) in R.
```

An ordinary candidate-wise diagonal hidden-state gate has the form

```text
Z' = Z + diag(g_1(x_1), ..., g_N(x_N)) R,
```

where candidate `i` can only rescale its own proposed residual `R_i`.  Static
mixtures, zero-vector attention used only to control an overall amount of
personalization, and a learned per-candidate sigmoid multiplying the same
candidate's history vector are members of these classes.

There is no meaningful theorem against an *unrestricted* “generic gate.”  For
any model output `s`, one can define `r := s-b` and `g := 1`, or define a gate
that internally recomputes the whole target model.  Such a definition makes
every architecture a gate and cannot decide novelty.  C09 therefore makes only
the architectural claim above: the proposed update cannot be implemented by a
global scalar mixture or a candidate-local diagonal rescaling unless the
nonlocal mechanism is hidden inside the object called “residual.”

## 2. The C09 operator

Two shared-parameter, information-restricted Transformer passes produce logits
`u_i` (query-first) and `v_i` (candidate-first), alongside the no-history base
`b_i`.  Define history residuals and their ordered candidate margins:

```text
r_i^Q = u_i - b_i                     r_i^C = v_i - b_i
m_ij^Q = r_i^Q - r_j^Q                m_ij^C = r_i^C - r_j^C.
```

For an ordered pair `(i,j)`, the conjunctive positive-margin strength is

```text
kappa(a,b) = a_+ b_+ / (a_+ + b_+)    when a_+ + b_+ > 0,
             0                         otherwise,

K_ij = kappa(m_ij^Q, m_ij^C),         K_ii = 0.
```

Thus `K_ij > 0` iff both views say the history residual prefers `i` over `j`.
The parallel sum is no larger than the weaker positive margin and is exactly
zero on disagreement.  No view scores are averaged.

Let `z_i` be the base candidate hidden state.  CMA performs one-sided
candidate-set contrast attention:

```text
A_ij = K_ij / (1 + sum_l K_il)
Delta z_i = sum_{j != i} A_ij W_V (z_i - z_j)
s_i = b_i + I[query present AND history present] w_o^T Delta z_i.
```

The `+1` is a null-attention sink, not a learned router.  A row without an
agreed pair is exactly zero.  This is standard set-attention-style aggregation;
it is not an antisymmetric edge-flow reconstruction, graph divergence, Hodge
projection, transport solve, or pairwise-score averaging.

## 3. Non-reduction witnesses

### 3.1 Not a global scalar mixture

The hand-computed three-candidate test uses

```text
z = (3, 1, 0),  r^Q = (2, 0, -1),  r^C = (1, 0, -2),
W_V = w_o = 1.
```

It gives

```text
Delta = (35/19, 2/5, 0).
```

`Delta` is not collinear with `r^Q` or `r^C`: its third coordinate forces a
putative global multiplier to zero, while its first coordinate is nonzero.
Therefore neither `b + alpha r^Q` nor `b + alpha r^C` represents this update.

### 3.2 Not a candidate-local diagonal hidden gate

Take two one-dimensional base states `z=(0,2)` and let both views have residual
scores `(1,0)`.  Then `K_01=1/2`, `A_01=1/3`, and

```text
Delta z_0 = (1/3)(z_0-z_1) = -2/3.
```

Candidate 0 changes even though its own local state is zero.  Every diagonal
update `g_0 z_0` is zero.  The nonzero off-diagonal Jacobian
`partial Delta z_0 / partial z_1` is the decisive fingerprint.

At the final *scalar-logit* level, an arbitrary per-candidate gate can always be
fit after the fact.  The witness is therefore about the implemented information
flow, not universal function expressivity.

### 3.3 It is a constrained ordinary-attention instance

CMA is not irreducible to the broad class of attention operators.  Given `K`
as an attention mask/weight matrix, ordinary set attention with a null sink and
contrast values implements the same update.  The potentially distinct object
is the construction of `K` from two information-restricted history-residual
margin fields, not the weighted-sum attention calculation.

C09 therefore makes no “new attention family” claim.  Ordinary learned
candidate attention with the same value/output projections is a decisive
control.  If it matches CMA, the conjunction pays no rent and C09 stops.  If
the required standard were “not representable by ordinary attention,” this
audit would fail.

## 4. Required degeneracy controls

C09 is rejected as a disguised gate if any of the following happens in the
implementation used for a probe:

1. `K` is reduced to one request-level or candidate-level scalar before the
   candidate interaction;
2. `W_V(z_i-z_j)` is replaced by a constant or by candidate `i`'s own residual,
   leaving only scalar attenuation;
3. a learned selector sees both unrestricted views and chooses a score stream;
4. the two view logits are averaged, summed, multiplied as probabilities, or
   routed as experts;
5. candidate interaction is removed and the result is matched by
   `b + g_i r_i` without moving equivalent nonlocal computation into `r_i`;
6. a one-candidate set produces a history correction (there is no candidate
   margin to agree on).

The CPU tests cover items 1/2 indirectly via the off-diagonal witness, item 4
with an explicit disagreement case, and item 6 exactly.

## 5. Audit conclusion

CMA exceeds the *ordinary* scalar/diagonal gate class by a concrete nonlocal
information-flow witness, so a minimal prototype is justified.  It remains a
constrained instance of ordinary set attention.  The result is
not a proof of broad novelty: “use agreement,” “use attention,” and “use
multiple views” are all established ideas.  The only defensible candidate
contribution is their narrow composition at history-residual candidate margins
under structural view barriers.  If matched single-view or learned-attention
controls explain any later gain, C09 stops rather than broadening the gate
definition.
