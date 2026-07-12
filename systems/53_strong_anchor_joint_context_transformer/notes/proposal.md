# C53 proposal — strong-anchor joint-context foundation

Status: pre-outcome exposed-cohort foundation probe.  It claims no novelty.
Fresh C47 reserves, dev, test, and qrels remain closed.

## Why this probe exists

C52 made token-level history fully rank-active but did not make its direction
correct.  A post-terminal audit then found registered D2p at `0.603050` on the
same Kuai cohort where raw BGE was `0.300870`.  C47--C52 were information
probes, not viable rankers.  The next architecture must start from the strong
anchor rather than replace it.

Earlier candidates either scored candidates independently (C02/C26), reduced
history before candidate competition (C27/C28/C31--C43), or tested candidate
self-attention only on recurrence features (C24).  C53 asks the simpler
foundation question that remains: can one Transformer jointly contextualize
the complete candidate set with query/history and learn a stable increment
over the strong base?

## Known architecture

For frozen LM states, construct one position-free candidate set and an ordered
history context:

```text
x_q = W q + type_q
x_hj = W h_j + type_h + recency_j
x_ci = W c_i + type_c + base_feature(s_i^base)
```

A two-layer Transformer uses a directed mask:

- query/history tokens attend only query/history tokens;
- every candidate attends query/history and every candidate;
- candidates have no rank-position embedding, so caller-order permutation is
  exact.

Candidate states produce a request-centered residual, and final scores are

```text
s_i = s_i^base + [g(x_ci^L) - mean_k g(x_ck^L)].
```

For empty history the residual path is structurally skipped and the output is
bit-exact base.  C53 uses seed-20260708 D2p as the Kuai base and therefore feeds
the residual Transformer that checkpoint's normalized query state and
item-adapter states, not the weaker pre-finetuning BGE space.  It uses the
registered frozen-BGE base and corresponding BGE states on Amazon.  This
difference is an upstream baseline boundary; the Transformer operator, masks,
optimizer, and gates are identical.  Both domains use history-present
strict-nonrepeat fit/A roles.

## Binding same-weight ablations

- `independent_candidates`: at inference, remove only candidate-to-other-
  candidate edges from the trained checkpoint;
- `wrong_history`: replace only the history with the frozen matched donor;
- `base`: remove the entire residual path;
- `no_history`: algebraic exact-base contract.

The same-weight edge ablation is a load-bearing test, not yet a fair retrained
architecture comparison.  A positive foundation result would still require a
separately trained independent control before any mechanism claim.

## Gate

Three seeds per domain train for two fixed full-fit epochs.  Before exposed A
labels are used, every seed must be finite, deterministic, candidate-
equivariant, exact on no history, and materially sensitive to both removal of
cross-candidate edges and wrong history.  A1 then requires positive-CI gains
over base, the same-weight independent ablation, and wrong history, with every
seed/fold positive and positive clicked direction/specificity in both domains.
The proposal lock binds every actually consumed embedding/offset/ID array, and
the frozen candidate-set hash is asserted before scoring and again immediately
before the common metric opens A labels.

Any failure closes generic joint-context conditioning as the immediate
foundation.  There is no width/layer/loss/epoch/scale/domain rescue.
