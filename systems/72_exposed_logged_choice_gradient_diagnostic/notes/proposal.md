# C72 proposal — exposed logged-choice formulation diagnostic

Status: pre-outcome, formulation-only.

C71's label-free gate passed, but its fresh unpacked target role contained no
click or purchase positives. C72 therefore uses only C47's 6,000-request
KuaiSearch fit role whose labels were already opened by C53. It does not claim
freshness.

C72 copies C71 exactly:

```text
p_tk = softmax_k(cos(q_t,c_tk)/0.1)
g_t  = normalize(c_t+ - sum_k p_tk c_tk)
a_t  = softmax_t(cos(q,q_t)/0.1)
m    = sum_t a_t g_t
s_i  = cos(q,c_i) + c_i^T m.
```

The positive-only, uniform-slate, semantic-history, matched wrong-user, and
query-only controls; scale; normalization; mechanical thresholds; and utility
thresholds are inherited without alteration. The target subset and wrong
donors are frozen by structural fields before C72 scoring.

If C72 fails, the fixed logged-choice gradient direction closes. If it passes,
the only conclusion is that acquiring a second real logged-choice domain is
worthwhile. A pass does not validate C70 or authorize dev/test.
