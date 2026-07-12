# CHHT mechanism fingerprint

## Operator

For layer `l`, let `x_lc in R^d` be the candidate FFN input, `W_l` its frozen
or shared base map, and let `U_l in R^(d x r)` have orthonormal columns.  The
ordered strictly-prior events are `h_1,...,h_m`, with event/recency features
`e_j`.  Candidate and query states are `c` and `q`.

Two independently projected event coordinates are

```text
a_ljc = tanh(A_h h_j + A_q q + A_c c + A_e e_j)
b_ljc = tanh(B_h h_j + B_q q + B_c c + B_e e_j).
```

An evidence coefficient `rho_ljc` is bounded with `tanh`; it is computed from
the same triadic inputs and includes the exact item-recurrence bit as observed
evidence, not as a separate score.  The non-diagonal event-composed core is

```text
M_lc = sum_j rho_ljc a_ljc b_ljc^T / max(1, sum_j |rho_ljc|)
S_lc = kappa * skew(M_lc) / max(kappa, ||skew(M_lc)||_F)
skew(M) = (M - M^T) / 2.
```

With `I_r` the rank-space identity,

```text
R_lc = (I_r - S_lc) (I_r + S_lc)^(-1)
Delta W_l(q,c,H) = U_l (R_lc - I_r) U_l^T W_l
y_lc = (W_l + Delta W_l(q,c,H)) x_lc.
```

The implementation uses a batched linear solve and the equivalent activation
form; it does not materialize a `d x d` matrix.  Since `S` is skew-symmetric,
`R` is orthogonal up to numerical precision.  Therefore the intervention is a
bounded rotation in a learned rank-`r` subspace rather than unconstrained
history-score addition.

## Frozen choices

- intervention: the candidate token's final compact-Transformer FFN output;
- number of modulated layers: one;
- rank: `r=8`;
- compact hidden width: `d=96`;
- maximum skew Frobenius radius: `0.35`;
- generator inputs: frozen D2t query state, candidate item state, ordered
  history item states, click/purchase bit, recency, and exact-recurrence bit;
- training seed: `20260708`;
- maximum observed history: 20 most recent strictly-prior events;
- inference: one request-history encoding, vectorized candidate kernels, one
  candidate forward pass, no stored user parameters, no test-time optimization.

## Boundary proofs

### No history

For `m=0`, the masked sum is exactly zero.  Hence `M=S=0`, `R=I`,
`Delta W=0`, and the emitted score is the unchanged frozen D2p score.  This is
an algebraic property, not a learned gate.  Unit tests require exact tensor
equality, and the screen requires zero score/rank mismatches for all 4,110
no-history requests.

### Rank and magnitude

`rank(Delta W) <= r`.  The Cayley transform is orthogonal in rank space, and
the `S` radius plus bounded score readout prevents an unbounded per-candidate
update.  Tests check skew symmetry, Cayley orthogonality, rank, and finite
gradients.

### Exact recurrence

Exact recurrence is an input bit to the same kernel and is protected only by
the training-time preservation loss:

```text
L_pres = KL(softmax(s_item / T) || softmax(s_CHHT / T))
         + max(0, margin - repeat_logit_margin_CHHT).
```

The item-only score is a teacher on train records only.  It is not read or
mixed at inference.  Removing `L_pres` is the rent-paying preservation
ablation for a later full gate.

## Training signal

The probe uses train labels only:

```text
L = L_listwise + lambda_pres L_pres
    + lambda_corr sum_z max(0, mu + ||S_z|| - ||S_true||)
    + lambda_norm ||S_true||^2,
```

where `z` is one of wrong-user, event-shuffled, query-masked, or coarse-only
history.  The corruption term calibrates the functional update itself; it does
not create additional inference branches.

## Degenerations and controls

- **ordinary static LoRA:** replace the Cayley core with one learned constant
  `B A`, independent of `q,c,H`;
- **DISeL/Ouroboros-style degeneration:** replace the off-diagonal skew/Cayley
  core by a diagonal rank-component gate; this is explicitly not CHHT;
- **output gate:** keep a history representation but multiply a score residual
  after the ranking head; no internal `Delta W`;
- **mean-history residual:** pool history and score its interaction with the
  candidate without changing the Transformer map;
- **history-only HyperAdapter:** generate `S(H)` with query/candidate generator
  inputs masked; application still sees the candidate activation;
- **no preservation:** retain CHHT but remove `L_pres`.

## Parameter and compute accounting

All screening variants share the same frozen D2t/item states, compact
Transformer width/layers, batches, candidate sets, optimizer steps, and epoch
budget.  Candidate-local metadata records total and operator-active parameters,
wall/GPU time, peak allocated memory, and observed candidate/history counts.
The controls use the same instantiated backbone; any mechanism-specific
parameter mismatch above 2% is a gate failure.

## Non-reducibility claims (bounded)

- Not ordinary LoRA: the update changes per `(q,c,H)` and is zero by
  construction without history.
- Not rank gating: a skew matrix has a zero diagonal and its Cayley map mixes
  rank directions nonlinearly.
- Not an output gate/router: history never chooses among fixed scores; it
  changes an FFN's internal linear operator before the score head.
- Not a profile adapter: no user-level weights are stored or reused; changing
  the candidate while holding the user/history fixed changes the kernel.
- Not DIN/ZAM/TEM attention: attention may encode context, but the
  load-bearing intervention is a candidate-specific functional update, and its
  degeneration back to pooled/target attention is an explicit control.

These are operator-level distinctions, not a claim of global novelty.  The
literature verdict remains bounded by `notes/nearest_neighbors.md`.
