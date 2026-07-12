# C47 proposal — Posterior-Supported Ridge Transformer (PSRT)

Status: pre-outcome formulation. No C47 data role or label has been opened.

## Observation → architecture consequence → falsification

**Observation.** The same fixed ridge history projector improved a query-only
ranker with true-over-wrong specificity on already-open strict-nonrepeat
KuaiSearch and Amazon-C4 cohorts. The surface is cross-domain, unlike exact
recurrence. Plain KRR is not the missing innovation: Cubit already replaces
attention by KRR, and C47's exposed Kuai margin over plain ridge is uncertain.

**Architecture consequence.** Replace one history-to-query attention read by a
ridge token mixer, but allow its mean write into candidate `c` only in
proportion to the candidate's support under the *same* posterior geometry. For
normalized lower-Transformer states and `lambda>0`,

```text
P_H   = H^T (H H^T + lambda I)^-1 H
u_q   = P_H q
rho_c = c^T P_H c
q_c'  = q + rho_c W_O u_q
score_c = UpperTransformer([q_c', c]).read()
```

`rho_c` is not predicted by a classifier and does not select a fixed scorer.
The lower Transformer supplies `q,H,c`; the ridge solve changes the internal
query token presented to the upper Transformer. The identical operator and
weights apply to every dataset. Missing history is the only structural mask.

**Falsification.** C47 closes unless, on fresh KuaiSearch strict-nonrepeat and
Amazon-C4 roles, it beats the query base, true history beats matched wrong
history, and the posterior-supported model pays stable incremental rent over
plain Cubit-style ridge, ordinary softmax history attention, and a
parameter-matched free candidate scalar gate. A Kuai-only or Amazon-only pass
is failure. Repeat requests must preserve registered item-only behavior and
no-history must return the registered base exactly before any dev access.

## One primitive

The primitive is **self-supported posterior write**, the multiplication of a
KRR mean write by the candidate's quadratic support computed from the same
normal equation. KRR, frozen LM embeddings, exact recurrence, and the final
rank loss are not individually claimed as contributions.

For `P_H` above, `P_H` is positive semidefinite and its eigenvalues lie in
`[0,1)`. Therefore `rho_c in [0,1)` for unit `c`. The primary can only contract
the plain-ridge write; it cannot amplify an unsupported semantic direction.
If `H c = 0`, both `rho_c` and the personalized correction are exactly zero.
Repeated aligned evidence monotonically raises support toward one in the
one-dimensional duplicate case.

## End-to-end Transformer placement

```text
query/history/candidate text + identity tokens
                  |
          shared lower Transformer
                  |
      P_H solve + candidate self-support
                  |
     candidate-specific query-token write
                  |
          shared upper Transformer
                  |
             ranking logit
```

The fixed-form diagnostic is not eligible as the final system. Eligibility
requires this lower/mixer/upper path to be load-bearing after matched training.

## Controls

1. `plain_ridge`: same states and solve, `rho_c=1` (Cubit boundary);
2. `softmax_attention`: same Q/K/V/O projections and upper Transformer;
3. `free_scalar_gate`: same ridge mean with a learned candidate/query/history
   scalar and matched active parameters;
4. `support_only`: posterior support gates a matched mean-history write;
5. query/candidate base with matched capacity;
6. DeltaNet/Gated-DeltaNet-style sequential fast-weight update when the full
   trained gate is reached.

If any simpler control satisfies the full gate, C47 may be useful as a model
but has no architecture attribution and cannot be the proposed primitive.

## Scope and complexity

The literal head costs `O(H^3 + CHd)` using the dual solve; `H<=50` in the
unified interface. A primal or Cholesky implementation may reduce constants,
but no efficiency advantage is claimed. There are zero online LLM calls. No
dataset/category/query-type branch, qrels access, fixed-score router, user-ID
score, or external feature lookup is allowed.
