# Design feedback after the C05 G2a failure

This is post-outcome guidance, not authorization for a rescue run.

## What the result changes

The first load-bearing issue is now more precise than “history may be noisy.”
An ordinary target-attention residual can consume history, move many parameters,
and reduce its training loss slightly while producing an almost perfectly
candidate-common score translation.  Reacting to history is therefore still
not evidence of personalized ranking.

The next primitive, if separately authorized, should operate directly in the
candidate-relative logit space:

```text
r_i = evidence_logit(q, c_i, H)
z_i = r_i - mean_valid_candidates(r)
delta_i = epsilon * tanh(z_i / temperature)
s_i = s_base_i + 1[history_present] * delta_i
```

This sketch is deliberately incomplete.  Its required properties are more
important than the exact formula:

1. **Common-mode null is algebraic.** Candidate-common history response must
   yield exactly zero score update, not a harmless-looking `+1` translation.
2. **The trust region is in score space.** `epsilon` bounds the final relative
   logit change; an L1 attention budget or bounded hidden residual is not enough.
3. **Order-changing capacity is audited before NDCG.** Frozen diagnostics must
   require nonzero within-request delta dispersion, positive-vs-negative delta
   margin, and a nontrivial candidate-order-change rate.  Parameter movement,
   attention mass and global delta magnitude are insufficient.
4. **Exact recurrence is separate and monotone.** If reintroduced, its
   click/purchase and recency semantics must reach the final logit through a
   nonnegative monotone path.  A positive attention bias does not protect it.
5. **Temporal claims need temporal states.** A history-set-invariant operator
   cannot use shuffle as a binding falsifier.
6. **Evidence audits stay held out.** Wrong-user, query-mask, event replacement,
   shuffle and coarse-only families used for the decisive gate cannot be the
   same families taught by the loss.

## How to validate from simple to complex

Any successor must use a new pre-outcome lock and an untouched cohort; the
exposed C05 internal set cannot be used to tune the revision.

1. Algebra/unit gate: exact no-history and common-mode zero, permutation
   equivariance, finite empty paths, final-logit trust-region bound.
2. Label-free real-state gate: verify nonzero candidate-relative response under
   true non-repeat history and zero response to an explicitly common history
   construction.  This is only a sensitivity check, not a positive result.
3. Fresh train-internal ranking gate: compare the relative-logit primitive with
   ordinary target attention and a parameter-matched history-free groupwise
   control; use full candidate sets and a new untouched split.
4. Held-out authenticity gate: only after ranking gain, test wrong-user,
   query-mask, matched replacement, coarse-only and (if temporal states exist)
   shuffle.
5. Exact-repeat gate: add the monotone item recurrence path and require
   non-inferiority to the registered item-only control.
6. Full Transformer gate: internalize the verified base and ranking head only
   after all earlier gates pass; then request dev authorization.

The architectural direction remains plausible because it directly removes the
observed common-mode failure.  It is not yet validated, and implementing it now
against the same internal outcomes would be post-hoc rescue rather than clean
experimentation.
