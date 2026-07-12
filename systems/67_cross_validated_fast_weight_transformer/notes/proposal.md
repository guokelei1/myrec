# C67 proposal — cross-validated fast-weight Transformer

Status: pre-outcome, data-free architecture falsifier. No repository record or
label is authorized.

## Observation to primitive

C64 showed that end-to-end LM adaptation can move rankings, while C65--C66
showed that subtracting a NULL state and penalizing wrong-history output still
does not make the identity of history load-bearing. C62--C63 further showed
that rank loss alone does not teach a forward-written latent memory what to
bind. The missing object is therefore not another output gate; it is an
internal write rule that distinguishes an event that generalizes across a
user's history from an event that merely reconstructs itself.

For independently LM-encoded history event views `(k_e, v_e)`, let a
request-local linear learner start at shared state `W_0`. Each event proposes
one inner update:

```text
g_e   = grad_W 0.5 ||W_0 k_e - v_e||^2
W_e   = W_0 - eta g_e
u_e   = mean_{j != e} [loss_j(W_0) - loss_j(W_e)]
w_e   = relu(u_e) / (sum_i relu(u_i) + epsilon)
W_H   = W_0 - eta sum_e w_e g_e.
```

The exact post-update held-out improvement `u_e`, not self-fit or update norm,
is the evidence admission quantity. If fewer than two events exist or every
proposal hurts held-out events, `W_H = W_0` exactly. The current query and
candidate set cannot affect `W_H`.

At read time, the same LM produces a query key and candidate target view. The
only personalized candidate signal is the functional delta between the frozen
request learner and its shared initialization:

```text
d_H(q,c) = -||W_H k(q) - v(c)||^2 + ||W_0 k(q) - v(c)||^2
score     = strong_base + center_candidates(d_H).
```

No history gives an exact base identity and repeat-present requests retain the
registered item-only fallback. The eventual real model, if ever authorized,
must use the same operator for KuaiSearch and Amazon-C4 with no dataset,
category, query-type, candidate-count, or score-threshold branch.

## Binding controls

- `standard_ttt_write`: average every event gradient without validation.
- `self_validated_write`: weight by the event's own reconstruction improvement.
- `gradient_agreement_write`: replace exact held-out improvement with the
  first-order gradient inner product, testing whether C67 reduces to known
  gradient agreement.

Every mode has the same projections, fast learner, step parameter, score
function, optimizer, training tasks, and total parameter count.

## Data-free falsifier

Each synthetic request defines an unseen linear law from keys to targets.
History contains repeated observations from that law plus independently drawn
nuisance observations. Candidates contain the true target for a new query and
matched distractors. A second unsupported regime gives every event a different
law, where a reliable writer should have little held-out support.

C67 advances only if all three seeds satisfy structural contracts, correct
history is load-bearing, nuisance events receive less write mass, and exact
held-out validation beats standard TTT, self-validation, and first-order
gradient agreement by the locked margins. A tie to gradient agreement rejects
the claimed primitive even if absolute accuracy is high.

## Outcome boundary

A pass is conditional learnability evidence only. A failure closes C67 before
repository data. A pass authorizes a separate implementation review, not
automatic real-data training. No generator, dimension, step, threshold, seed,
noise, or nuisance-count rescue is allowed after the first outcome.
