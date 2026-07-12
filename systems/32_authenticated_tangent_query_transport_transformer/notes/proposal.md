# C32 proposal: Authenticated Spherical-Tangent Query Transport

Status: pre-outcome draft; immutable after proposal lock.

## Single hypothesis

C31 established a formal positive direction across all three seeds but failed
CI/fold stability.  Its history profile contains a component parallel to the
adapted query, which can reinforce query-common geometry without providing a
candidate-relative direction.  C32 tests whether the admissible history write
should instead lie in the query's tangent space on the unit sphere.

For frozen BGE embedding `x=LM(text)` and the unchanged rank-16 adapter
`e(x)=normalize(x+UVx)`:

```text
a_h = softmax(cos(x_q,x_h)/0.1), h in H_auth
p   = sum_h a_h e(h)
t   = p - <p,e(q)> e(q)
q_H = normalize(e(q) + t)
d_c = 2 [cos(e(c),q_H) - cos(e(c),e(q))]
s_c = z(D2p_c) + center_candidates(d_c)
```

The identity `<t,e(q)>=0` is the architecture contract.  C32 changes no data
slice, capacity, attention temperature, scale, loss, optimizer, epoch, or
fallback.  It deliberately keeps C31's raw semantic attention so this candidate
tests only tangent projection; adapted-space attention is a later matched
ablation, not part of the primary.

## Evidence boundary

On now-open C31-A, a fixed post-terminal six-operator audit estimated raw
tangent transport at +0.002682 NDCG@10, positive in all three seeds but with one
slightly negative fold and a zero-crossing CI.  Adapted-attention tangent was
+0.002506 with all seed/fold signs positive.  These values formulate the
hypothesis only.  C31-A is excluded from C32 fit and gates.

C32 fit is the unchanged 10,000-request C31 fit set.  C32-A is C31 delayed-B,
never previously feature-materialized, scored, or labeled.  C32 delayed-B is
C31 escrow and remains closed.  Three new seeds are fixed before any C32-A
feature access.

## Gate

G0 must pass the same strict-past true/wrong authentication contract.  A0 must
pass activity, wrong-history corruption, exact repeat/no-history/no-auth/query
fallbacks, determinism, and candidate permutation before A labels open.  A1
requires mean gain >=0.001, positive CI, every seed and every hash fold positive,
and positive-CI true over wrong history.  Only A1 authorizes delayed-B controls:
unprojected C31 geometry, adapted-attention tangent, and unauthenticated tangent.
Dev/test remain closed.
