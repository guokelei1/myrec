# C29 proposal: Causally Authenticated Mediation Transformer

Status: pre-outcome draft; immutable after proposal lock.

## Hypothesis

The remaining failure is not history responsiveness but provenance fidelity.
A full pretrained Transformer can learn a small ranking-aligned history
residual, while unrelated histories reproduce or exceed it.  A history event
that survives in the same user's strictly earlier memory is a stronger
provenance witness than semantic similarity, category, or update magnitude.

## Primitive

For request `r=(u,q,c,H,t)`, let `M_u(t-)` be item identities found in user
`u`'s history snapshots at timestamps strictly below `t`.  Authentication is

```text
a(h;u,t) = 1[h.item_id in M_u(t-)].
H_auth = {h in H : a(h;u,t)=1}.
```

Same-timestamp requests are all scored before any memory update.  `M_u` never
contains candidate outcomes from the recipient request.  User identity and
memory have no candidate-scoring edge.

For each candidate, a shared pretrained BGE Transformer receives

```text
[CLS] query [SEP] candidate [SEP] authenticated-history summaries.
```

The factual score `f(q,c,H_auth)` and null score `f(q,c,empty)` share every
parameter.  The only personalized residual is

```text
d(q,c,H) = 2 tanh(f(q,c,H_auth) - f(q,c,empty)).
s(q,c,H) = z(D2p(q,c)) + center_candidates(d).
```

Training uses three proper terms with the same candidate label: true ranking,
wrong-history return to the D2p probability, and true-minus-wrong residual
direction.  Registered wrong donors are history-length/time matched and have
zero recipient-candidate overlap and a different user identity.  Wrong events
are re-authenticated against the recipient user's memory.  At inference no
wrong donor is required.

The scalar readout is initialized to exact zero, so every seed starts at the
base-neutral function; seed variation comes only from the frozen minibatch
permutation.  Encoder dropout is fixed to zero so identical factual/null inputs
cancel exactly during training as well as inference.  These are part of the
architecture contract, not tuned outcomes.

Repeat-present requests use the registered item-only anchor as an exact final
fallback.  No history, no authenticated event, or absent query gives exact
D2p.  Thus reliable recurrence cannot be diluted by cross-item transfer.

## Why 10,000 fit requests

The primary core has 23,954,432 trainable parameters, whereas C28's core had
0.14M trainable parameters.  The 3,000-request post-terminal probe was a
representation test, not an adequate formal fit budget: only 1,767 requests
had any authenticated event.  C29 freezes 10,000 label-free-selected,
history-present strict-nonrepeat requests with no query/category/label slice.
Every later matched control must use the identical fit set and one epoch.  The
size is frozen before C29-A is materialized or scored and cannot be enlarged
after outcome.

## Stage plan

After the proposal lock, G0 opens only label-free C29-A structure and requires
at least 50% true-authentication coverage, true-minus-wrong authenticity at
least 0.25, true greater than wrong on at least 50% of requests, and wrong
authenticity at most 0.05.  Delayed-B is not feature-materialized.

Phase 1 trains only the primary for three fixed seeds.  Label-free A0 must pass
activity, corruption, authentication, determinism, permutation, and fallback
contracts.  Only then are C29-A labels opened.  A1 requires stable utility over
D2p, true over wrong, and clicked correction direction.  Failure closes C29.

If and only if A1 passes, the same proposal authorizes matched controls on the
still-unopened delayed-B: unauthenticated mediation, authenticated factual-only
readout, and an authenticated randomly initialized Transformer with the
identical BGE config and parameter count.  C29 cannot claim architectural rent
before those matched-capacity controls pass on delayed-B.

Passing delayed-B authorizes review, not dev/test.
