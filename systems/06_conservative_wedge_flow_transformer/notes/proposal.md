# C06 proposal: Candidate-Local Hodge-Trusted Flow Transformer

Status: **pre-outcome architecture hypothesis; no C06 data-fit outcome exists**

## Observation -> architecture consequence -> falsification

### Observation

C05's final target-attention checkpoint was not a zero-movement no-op.  It
instead produced a model-internal tanh residual of exactly `+1` for all 54,637
valid internal candidates and changed zero rankings.  Ordinary history
attention therefore has an architectural nuisance direction: history can drive
an arbitrarily large candidate-common hidden update even though listwise
ranking is invariant to the resulting common score translation.

The new design requirement is dataset-independent:

> Personalization may redistribute relative score among candidates, but may not
> create request-level score mass shared by every candidate.

### Architecture consequence

Use one Transformer/LM with a block-sparse information-flow mask. Query tokens
see query tokens only. Each candidate segment sees the query and its own
segment, but neither other candidates nor history. Ordered history states see
query/history. Candidate identity is not encoded as an absolute list position.
Consequently, the standard LM head emits a jointly trained but history-blind
base score `b_i`, and the wedge layer is the sole history-to-logit and
cross-candidate path.

For candidate `i` and history event `j`, form a triadic state and two bounded
factors:

```text
z_ij = tanh(Wq q + Wc c_i + Wh h_j)
a_ij = tanh(Wa z_ij)
b_ij = tanh(Wb z_ij)
```

For each history event, these factors define the conceptual candidate-edge
field

```text
F_ikj = (a_ij^T b_kj - b_ij^T a_kj) / (2r)
```

so `F_ikj = -F_kij`. Its common-degree divergence induces the globally
rankable Hodge projection, while the remainder is cyclic inconsistency:

```text
u_ij = (1/n) sum_k F_ikj
G_ikj = u_ij - u_kj
C_ikj = F_ikj - G_ikj
EG_ij = sum_k G_ikj^2
EC_ij = sum_k C_ikj^2
t_ij = EG_ij / (EG_ij + EC_ij + 1e-12)
```

Unlike a single event-level scalar, `t_ij` is candidate-local: a cycle incident
to one candidate need not suppress a consistent candidate elsewhere. The raw
cycle direction is never scored. It can only lower the symmetric trust of a
projected gradient edge:

```text
T_ikj = (t_ij t_kj / 2) (u_ij - u_kj)
v_ij = (1/n) sum_k T_ikj
d_i = sum_j omega_j v_ij
rho = rho_max tanh(raw_rho)
delta_i = rho d_i
s_i = b_i + 1[history present] delta_i
```

`T_ikj=-T_kij`; a pure potential has `t≈1`, while a pure cycle has `t=0`.
Flipping the sign of a cycle leaves `EC`, `t` and the final ranking unchanged.
Query-history confidence supplies nonnegative event weights `omega_j` with
`sum_j omega_j < 1`.

The edge matrix is not materialized. With candidate means `mean(a_.j)` and
`mean(b_.j)`, the same raw potential is

```text
u_ij = (a_ij^T mean(b_.j) - b_ij^T mean(a_.j)) / (2r).
```

Centering both factors gives the cyclic residual directly:

```text
Ac_ij = a_ij - mean_i(a_ij)
Bc_ij = b_ij - mean_i(b_ij)
C_ikj = (Ac_ij^T Bc_kj - Bc_ij^T Ac_kj) / (2r).
```

FP64 low-rank energy moments compute every `EC_ij`, giving
`O(B*H*C*r^2)` time and `O(B*C*H*r)` state rather than a materialized
`O(B*H*C^2)` graph.

The architecture has hard properties before training:

- `sum_i delta_i = 0`; a nonzero common translation is impossible;
- candidate-common factors yield `delta_i=0` exactly;
- `|delta_i| < rho_max`, directly in final-logit space;
- no history and one-candidate requests return the base exactly;
- candidate permutation permutes the output;
- reliability is candidate/event local and cannot be folded into the scalar
  history-event weight;
- pure cyclic evidence can suppress trust but cannot inject its cyclic sign as
  a ranking direction.

`t` is candidate-set conditioned, so C06 intentionally makes no
subset-independence or per-distractor Lipschitz claim. Nested, duplicated and
distractor-augmented pools are mandatory audits. The hard guarantee is only
that every resulting correction remains conservative and score bounded.

### Why this is still an LM/Transformer ranker

The frozen-base wrapper in `model/wedge_flow.py` is only an algebraic falsifier.
A final eligible system must place the same layer in the LM ranking head and
jointly train the query, candidate, history, base-score and flow states under
the block-sparse barrier. It is
not a router between D2p/item-only scores, an offline feature, or an MLP over LM
embeddings.  A candidate token attending directly to history, history features
concatenated into `b_i`, or any auxiliary history score would violate C06.

### Falsification

The operator is useful only if it beats all of the following under matched
capacity, steps and candidate sets:

1. ordinary target attention with the same final candidate-centering and score
   trust region;
2. the same factors with `t=1`, isolating whether local Hodge trust contributes;
3. the rejected global event-level Hodge scalar, testing whether locality is
   actually load-bearing;
4. a parameter-matched direct learned candidate/event gate, testing whether the
   Hodge decomposition adds anything beyond generic gating;
5. a history-null groupwise Transformer;
6. pairwise Transformer preferences with ordinary additive/Borda aggregation;
7. a potential-flow restriction that reduces to centered unary evidence;
8. a MIR/SetRank-style history-aware groupwise block;
9. the clean registered D2p coordinate.

Failure against centered attention means C06 merely repaired C05's nuisance
direction and is not an architectural contribution.  Parameter movement,
nonzero edges, attention mass and loss reduction cannot substitute for ranking
gain.

## Exact recurrence is deliberately later

The first C06 probe is non-repeat only.  If and only if relative transfer passes,
exact recurrence can enter as another conservative edge field:

```text
r_i = sum_exact_events positive_action_weight * exp(-positive_decay * age)
F_exact_ik = tanh(r_i - r_k)
```

Its derivative with respect to `r_i` is nonnegative.  Exact-matching events are
excluded from the free semantic field so they cannot be reversed downstream.
This extension needs its own lock and item-only non-inferiority gate.

## Non-goals

- No claim that pairwise ranking, antisymmetry, Borda aggregation, Hodge
  decomposition, set attention, or wedge factorization is individually new.
- No category/coarse-semantic feature, dataset branch or query-type threshold.
- No claim of independence from candidate-pool composition; it is audited, not
  assumed away.
- No C05 internal retry, dev evaluation, full LM training or test access under
  the current authorization.
