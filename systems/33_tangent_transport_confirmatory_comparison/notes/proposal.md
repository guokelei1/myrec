# C33 proposal: fresh paired tangent confirmation

Status: pre-outcome draft; immutable after proposal lock.

C32 is the first candidate with a positive overall confidence interval and all
three seeds positive, but it failed one preregistered hash fold and did not earn
control access.  C33 does not reinterpret that failure.  It asks a new,
narrower question on entirely fresh requests: does the exact tangent projection
replicate and outperform the capacity-matched unprojected transport?

Both models share frozen BGE query/history/candidate embeddings, the same
rank-16 residual adapter, causal authentication, raw semantic attention,
profile scale 1, correction scale 2, full-request losses, optimizer, one epoch,
and seed-specific initialization.  The only difference is:

```text
primary:    t = p - <p,q>q ; q_H = normalize(q+t)
control:    t = p           ; q_H = normalize(q+p)
```

Each seed uses identical initial adapter tensors and request order in the two
modes.  C33-A/B/escrow are selected afresh from the label-free reserve and have
zero overlap with C32-A, C32 delayed-B, or C32 escrow.  Thus C33 is independent
confirmation, not delayed-B rescue or optional continuation of C32.

G0 and label-free A0 retain authentication, activity, wrong-history, exact
fallback, determinism, permutation, and candidate-hash contracts.  A0 also
requires the tangent constraint to change at least 2% of complete control
orders and 0.5% of control top-10 sets.  A1 requires the primary versus D2p
effect/CI/every-seed/every-fold checks and true-over-wrong CI, plus at least
+0.0005 NDCG@10 over the unprojected control with positive CI and every seed and
fold positive.  Failure closes the tangent claim.  Passage authorizes only a
separate delayed-B confirmation protocol, never dev/test automatically.
