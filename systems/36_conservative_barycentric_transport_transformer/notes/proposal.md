# C36 proposal: conservative barycentric tangent transport

Status: pre-outcome draft; immutable after proposal lock.

## Observation → consequence → falsification

**Observation.** C35 repaired C34's near-always-on admission law: roughly 25%
of candidate rows abstained and over 90% of active requests mixed admitted and
rejected candidates. Yet its relative-only write was below D2p for every seed
and below the candidate-shared global tangent control. Selectivity alone is
therefore insufficient; candidate competition can discard or reverse the weak
shared query-history signal.

**Consequence.** Factor the attention residual into a shared tangent write and
a conservative candidate contrast. For adapted unit query `q`, authenticated
events `h_j`, and candidates `c_i`:

```text
u_j = (I-qq^T) h_j
v_i = (I-qq^T) c_i
g   = (I-qq^T) sum_j softmax(<q,h_j>/tau) h_j
k_ij = cos(v_i,u_j)
s_ij = ReLU(k_ij - mean_l k_lj)
r_i  = sum_j s_ij u_j / (1 + sum_j s_ij)
A    = {i : sum_j s_ij > 0}
delta_i = 1[i in A] (r_i - mean_{l in A} r_l)
lambda  = ||g|| / (||g|| + max_i ||delta_i|| + eps)
d_i     = g + lambda delta_i
q_i^H   = normalize(q + d_i)
score correction_i = 2[<c_i,q_i^H> - <c_i,q>].
```

This one conservation law has three algebraic consequences before labels:

1. `mean_i d_i = g` exactly;
2. candidates outside `A` retain the exact global state;
3. when `g != 0`, every `d_i` has positive inner product with `g`, because
   `lambda ||delta_i|| < ||g||`.

There is no learned gate, candidate scalar head, pair MLP, threshold, category,
query-type, or dataset branch. Repeat requests return item-only; missing query,
history, or authenticated evidence returns D2p.

**Falsification.** Before opening C36-A labels, every seed must preserve the
global mean and inactive state within `1e-6`, have zero nonpositive global
alignment, keep the bounded residual strictly smaller than the global write,
retain at least 10% exact relative abstention and 50% mixed-admission requests,
and materially change ranking versus every reduction. A1 then requires a
positive-CI gain of at least `0.001` NDCG@10 over D2p and at least `0.0005`
over every matched control, with every seed and hash fold positive.

## Formulation audit and isolation

A label-free audit on the permanently excluded C35-A surface found global-mean
error at most `2.98e-8`, exact inactive-state error `0`, no direction reversal,
25.32%--25.85% inactive candidates, and 5.03%--7.69% top-10 change versus
global-only full scores. No label or metric was read by that audit.

C36 reuses the fixed C35 fit because the local strict-nonrepeat pool cannot
provide another isolated 10k fit. C35-A is never a C36 target. C36-A is C35
delayed-B, whose features/scores/labels were never opened; C36 delayed-B is C35
escrow. Only A1 passage can authorize delayed-B. No dev/test access is allowed.
