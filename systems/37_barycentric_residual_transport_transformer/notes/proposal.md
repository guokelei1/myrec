# C37 proposal: barycentric residual transport

Status: pre-outcome draft; immutable after proposal lock.

## Observation → consequence → falsification

**Observation.** C36 passed 35/36 label-free checks. Its global write,
candidate-relative admission, and active-set barycenter were all load-bearing,
but the extra soft max-norm shrinkage changed only 2/600 top-10 sets versus the
unbounded barycentric reduction. The failed operator must be deleted, not
rescued by lowering its activity threshold.

**Consequence.** For adapted query `q`, authenticated history `h_j`, and
candidates `c_i`:

```text
u_j = (I-qq^T) h_j
v_i = (I-qq^T) c_i
g   = (I-qq^T) sum_j softmax(<q,h_j>/tau) h_j
s_ij = ReLU(cos(v_i,u_j) - mean_l cos(v_l,u_j))
r_i  = sum_j s_ij u_j / (1 + sum_j s_ij)
A    = {i : sum_j s_ij > 0}
delta_i = 1[i in A] (r_i - mean_{l in A} r_l)
d_i     = g + delta_i
q_i^H   = normalize(q + d_i)
correction_i = 2[<c_i,q_i^H> - <c_i,q>].
```

The candidate-axis conservation law gives `mean_i d_i=g` and exact global
state for candidates outside `A`. Unlike C36, there is no coefficient that can
silently collapse the residual. The same shared rank-16 LM adapter is the only
trainable representation change. Repeat uses item-only; absent query/history/
authenticated evidence uses exact D2p.

**Falsification.** Before A labels, the primary must preserve the global mean
and inactive state within `1e-6`, retain at least 10% exact relative abstention
and 50% mixed-admission requests, keep every candidate write positively aligned
with the global write and every residual norm below its global norm, and differ
materially from all reductions. A1 requires positive-CI NDCG@10 gain of at
least `0.001` over D2p and `0.0005` over every control, with every seed and hash
fold positive.

## Isolation

On the permanently excluded C36-A surface, the surviving operator had
global-mean error `2.98e-8`, inactive-state error `0`, zero reversed writes, and
maximum residual/global norm ratio 0.699--0.717. It changed 35.83% of complete
orders and 5.33% of top-10 sets versus global-only. No labels or metrics were
read.

C37 reuses the fixed fit. C37-A is untouched C36 delayed-B; C37 delayed-B is
untouched C36 escrow. C36-A is excluded. Only A1 passage may authorize
delayed-B; no dev/test access is allowed.
