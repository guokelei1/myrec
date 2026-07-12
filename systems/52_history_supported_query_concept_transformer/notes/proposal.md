# C52 proposal — history-supported query-concept attention

Status: pre-outcome formulation.  C47-A labels are already exposed and may be
used only after this proposal and its execution settings are locked.  The
KuaiSearch/Amazon reserve roles remain unopened.

## Failure-derived hypothesis

C47--C51 show that a pooled LM state contains a cross-domain history-subspace
signal, but candidate support, sign consensus, prediction error, orthogonal
dual memory, and centered covariance do not turn it into a stable new readout.
C26 separately showed token-level history sensitivity, but compressed it into
an independently learned bounded scalar after all representation formation.
C27/C28 changed final candidate competition only after token evidence had
already been pooled.

The remaining narrow hypothesis is: **history should change which query
concepts a candidate attends to, without being allowed to rewrite the semantic
value carried by those concepts.**

## Operator

A shared frozen LM supplies normalized query-token states `q_k`, candidate
token states `c_il`, and history-event states `h_j`.  First perform ordinary
late interaction within each candidate:

```text
c*_ik = sum_l softmax_l(<q_k,c_il>/tau_c) c_il
b_ik  = <q_k,c*_ik>
P_H   = H^T (H H^T + I)^-1 H
e_ik  = <c*_ik, P_H q_k>
```

The primary changes only the attention allocation over query concepts:

```text
a0_ik = softmax_k(b_ik/tau_q)
aH_ik = softmax_k((b_ik+e_ik)/tau_q)
```

Its fixed formulation score is the corresponding log-partition displacement,
in the same cosine unit as the frozen base:

```text
r_i = tau_q logsumexp_k((b_ik+e_ik)/tau_q)
    - tau_q logsumexp_k(b_ik/tau_q)
s_i = base_i + r_i.
```

If the formulation gate passes, the trainable architecture will feed `aH`
and the unchanged semantic carrier pairs `(q_k,c*_ik)` into a shared
Transformer block before candidate scoring.  No history vector is a value and
no external router chooses whether to personalize.

Ridge and all three temperatures are fixed to `1` and `0.1`, inherited from
C47/BGE cosine geometry.  There is no dataset, category, query-type, history-
length, or rank branch.  Empty history makes `e=r=0` exactly.

## Binding controls

- `linearized_token_krr`: `sum_k a0_ik e_ik`, the first-order/post-allocation
  reduction with identical token states;
- `token_softmax`: replace `P_H q_k` by ordinary query-to-history softmax,
  retaining the same query-concept partition;
- C47 `plain_ridge`: pooled Cubit/KRR scalar;
- C47 fixed `softmax_attention` and posterior-supported ridge;
- wrong-user history with the primary token operator;
- frozen query/candidate LM cosine base.

The primary must beat every direct control, base, and wrong history with fixed
positive effects, positive paired intervals, and all hash-fold signs on both
domains.  Clicked direction and clicked true-minus-wrong specificity must also
be positive.  Otherwise C52 closes before trainable implementation or fresh
reserve.

## Claim boundary

C52 does not claim KRR, late interaction, setwise ranking, or history-aware
attention as new.  Its only possible contribution is the constrained use of a
history KRR projection as a candidate-specific query-token allocation bias
while keeping values semantic and history-free.  The current gate can only
reject or authorize implementation of that primitive.
