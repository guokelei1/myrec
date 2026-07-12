# C44 partial-evidence logit-flow proposal

## Observation

C42 confirmed metric-coupled history utility on fresh Amazon data, but C43
showed that this did not become evidence-faithful personalization on
KuaiSearch: true history tied wrong-user history and a shifted metric loop was
nominally stronger. All C40--C43 modes pooled history into a request/head
profile before candidates read it.

C34/C35 did make the write candidate-local, but converted support into a new
transported vector; selective admission then failed to align with relevance.
C27/C28 compared candidate pairs after scoring, while C03's three-plan OT was
too expensive and multiplicatively conservative.

## Primitive

For head `r`, candidate `i`, and event `j`, first compute the score change that
event `j` alone would induce:

```text
s_rij = <R_r(c_i), normalize(R_r(q) + R_r(h_j))>
        - <R_r(c_i), R_r(q)>.
```

Each event then distributes one unit of partial mass across candidates and a
fixed null sink:

```text
[p_r1j,...,p_rCj,n_rj] = softmax([s_r1j,...,s_rCj,0] / tau).
f_ri = mean_j p_rij
d_i  = scale * mean_r (f_ri - mean_k f_rk).
```

Thus unsupported events can spend mass on null, supported events must choose
among the actual alternatives, and the total candidate correction is exactly
zero. No value vector, output projection, scalar head, router, pair MLP, or
post-hoc score mixture can change the sign. Query/history/candidate states are
inside the same Transformer metric loop.

## Matched reductions

Every mode owns identical low-rank tensors and paired initialization:

1. `partial_logit_flow` — primary;
2. `forced_logit_flow` — removes only the null sink;
3. `partial_vector_write` — keeps the same partial allocation but converts it
   back into a candidate-local history-vector write;
4. `global_vector_write` — pools history before the candidate decision.

The synthetic task plants one relevant event among multiple irrelevant but
query-plausible events. Only the relevant event has positive candidate
surplus; irrelevant events have heterogeneous negative surplus. This directly
tests whether null, candidate competition, and logit-space writing all pay
rent.

## Interpretation

Candidate-axis normalization is covered by Slot Attention. Doubly-stochastic
attention and optimal-transport normalization are covered by Sinkformers and
successors; sparse/empty attention is also established. C44 does not claim any
of those ingredients. Its only potentially distinct object is the
ranking-specific composition of counterfactual event surplus, a candidate-plus-
null partial assignment, and a centered logit flow as the sole history channel.
Novelty remains uncertain until both nearest-neighbor controls and real data
pass.
