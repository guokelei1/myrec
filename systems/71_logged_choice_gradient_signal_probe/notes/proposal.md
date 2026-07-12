# C71 proposal — logged-choice gradient signal

Status: pre-outcome KuaiSearch train-only signal gate. C71 is not C70 and is
ineligible as the proposed architecture.

## Question

C69 showed that an item relation learned from positive-only adjacent behavior
and semantic-matched cross-user negatives is not relevance-aligned. C70
proposes replacing mined negatives with a user's actual historical opportunity
set, but only KuaiSearch currently exposes recoverable historical slates.

C71 asks the narrower prerequisite: does that logged choice object carry a
ranking direction worth pursuing at all?

## Frozen signal

For historical query `q_t`, selected history item `c_t+`, and the candidate
slate `C_t` from the strictly earlier source request:

```text
p_tk = softmax_k(cos(q_t,c_tk) / 0.1)
g_t  = normalize(c_t+ - sum_k p_tk c_tk).
```

For the current query `q`, linked episodes are selected by

```text
a_t = softmax_t(cos(q,q_t) / 0.1)
m   = sum_t a_t g_t
d_i = cos(c_i,m) * ||m||
s_i = cos(q,c_i) + d_i.
```

The last expression is implemented as the equivalent dot product `c_i @ m`
with normalized LM states. It has no learned coefficient. Empty episode memory
returns the query-only score exactly.

## Controls

- `positive_only`: replace `g_t` with the selected item state;
- `uniform_slate`: subtract the unweighted slate mean rather than the
  historical-query expectation;
- `semantic_history`: current-query attention over positive history items;
- `primary_wrong`: use a different user's episode memory matched on episode
  count, current candidate count, and current-query cosine;
- `base`: frozen query--candidate cosine.

All use the same embeddings, temperatures, current candidates, and fixed
coefficient. C71 passes only if logged-choice gradient beats every control and
wrong history with positive intervals and every fixed-fold sign, while its
clicked correction direction is positive.

## Isolation

The historical packed pool contains 96,939 train requests, and every one of
its 29,277 strict-nonrepeat requests has appeared in a prior candidate role.
C71 therefore selects targets only from the 66,778 train requests absent from
that packed pool. Selection uses request/user/time/query/history/candidate IDs
but not clicked or purchased fields. Target labels may open once after the
label-free A0 report passes. Historical source labels are never read.

No dev/test/qrel path is authorized. A failure closes this fixed logged-choice
gradient signal; there is no temperature, normalization, coefficient, bin,
cohort, or pseudo-negative rescue.
