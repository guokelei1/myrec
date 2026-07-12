# C03 Mechanism Fingerprint

Status: frozen before any C03 dev outcome.

## State construction and intervention point

For one request-candidate pair, the frozen BGE encoder produces a query vector
`e_q`, a candidate vector `e_c`, and one vector `e_hj` for each strictly-prior
history event.  A trainable compact Transformer jointly contextualizes

```text
[Q, H_1, ..., H_m, C]
```

with role, event-type, and temporal-position embeddings, producing
`q`, `h_1..h_m`, and `c`.  Transport is inserted after this interaction block
and before the ranking logit.  Thus history personalization passes through both
the Transformer and the transport operator.

## Pairwise partial transports

Learned projected similarities define real-real scores

```text
A_qh[j] = <W_qh^q q, W_qh^h h_j> / sqrt(d)
A_hc[j] = <W_hc^h h_j, W_hc^c c> / sqrt(d)
          + b_id * 1[item(h_j) = item(c)]
A_qc    = <W_qc^q q, W_qc^c c> / sqrt(d).
```

`b_id >= b_floor > 0` is a constrained protected identity atom inside the
transport cost.  It is not emitted as a score.

For each pair, append a dustbin row and column with a learned finite score and
solve entropy-regularized OT using SuperGlue-style partial-assignment
marginals.  The generic implementation is log-domain Sinkhorn.  Because every
C03 plan has a singleton query or candidate side, the executed path solves the
same Sinkhorn scaling equations as one differentiable scalar root, which gives
machine-accurate marginals at lower cost.  Let the resulting plans be `P_qh`,
`P_hc`, and `P_qc`.  With one
query/candidate state and `m` real history states, rescale the relevant
real-real mass to

```text
a_j = (m + 1) P_qh[Q, H_j]
b_j = (m + 1) P_hc[H_j, C]
d_qc = 2 P_qc[Q, C].
```

All lie in `[0,1]`; row and column marginals are conserved by construction.

## Cycle-intersection mass and null

The event overlap and mismatch are

```text
u_j = sqrt(a_j b_j)
Delta_cycle = sum_j |a_j - b_j| / (sum_j a_j + sum_j b_j + eps)
g_j = d_qc * exp(-lambda_cycle * Delta_cycle) * u_j
t = sum_j g_j.
```

By Cauchy-Schwarz and the partial-assignment marginals, `0 <= t <= 1` up to
floating-point tolerance.  `g_j` is zero if either query-to-event or
event-to-candidate mass is zero; `d_qc` additionally requires direct
query-candidate support.  The diagnostic null mass is `1 - t`.  The three
learned dustbin scores decide where unsupported mass goes; softmax cannot leave
mass unassigned.

## Candidate update and signed residual

With normalized non-null event weights `w_j = g_j / (t + eps)`,

```text
h_bar = sum_j w_j h_j
c_plus = c + t W_o h_bar
r_raw = t * (s(q, c_plus) - s(q, c) + softplus(b_mass)).
```

Within one fixed candidate set, the emitted history residual is centered:

```text
r_i = gamma * (r_raw_i - mean_k r_raw_k)
score_i = D2p_i + r_i.
```

Centering makes the residual signed without inventing a separate negative
history channel.  `gamma` is frozen from train-internal calibration.  With no
history, the implementation bypasses neither model nor dataset: the evidence
mask algebraically sets every `r_raw` and `r_i` to exactly zero, so the score is
exactly D2p for every candidate.

## Training signal

The probe trains only on `records_train.jsonl` labels.  It combines request
ranking loss, exact-atom mass preservation, and train-internal contrastive
corruption losses for wrong-user, event shuffle, query mask, and coarse-only
history.  Dev remains label-free until the shared evaluator reads qrels.

## Complexity

For history length `m <= 20`, one local Transformer layer costs
`O((m+2)^2 d)`.  The three partial plans cost `O(Km)` for fixed Sinkhorn
iterations `K`; the 1-by-1 direct plan is constant.  Memory is `O(m d + m)` per
candidate.  There are no online LLM calls.

## Exact matched-capacity degenerations

All variants instantiate the same module and parameter tensors; only the
operator changes.

- `softmax`: ordinary candidate-to-history softmax target attention; no
  capacity conservation and no null.
- `no_null`: real-only normalized pairwise masses; every event distribution
  must be allocated.
- `no_cycle`: dustbin transport remains, but only `h↔c` mass updates the
  candidate; `q↔h` and direct `q↔c` intersection are removed.
- `mean_pool`: uniform masked history mean with the same update/ranking head.

If C03 and these degenerations have equivalent corruption behavior or ranking
outcomes, the load-bearing primitive fails.
