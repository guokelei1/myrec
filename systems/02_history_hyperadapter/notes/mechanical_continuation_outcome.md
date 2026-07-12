# C02 mechanical-continuation outcome

Date: 2026-07-11 (Asia/Shanghai)

Status: **TERMINAL FAIL at the frozen train-internal gate; close before dev.**

## Outcome

The one authorized mechanical continuation completed cleanly.  The only model
change was the pre-locked differentiable zero for an all-no-history corruption
batch.  Fourteen candidate tests pass, the continuation lock revalidates, all
five variants completed two epochs and 3,942 optimizer steps, and every
variant has 322,036 trainable parameters.  This is therefore a valid negative
mechanism result rather than another numerical or protocol failure.

CHHT passed only two of six conjunctive internal checks:

| check | actual | frozen requirement | verdict |
|---|---:|---:|---|
| finite loss and decrease | finite, `3.957445 -> 3.990273` | final < first | fail |
| non-repeat CHHT − D2p NDCG@10 | `0.000000` | `>= 0.001` | fail |
| repeat CHHT − item teacher | `+0.074407` | `>= -0.003` | pass |
| non-repeat margin over best control | `0.000000` | `>= 0.0005` | fail |
| true/corrupt core-norm ratio | approximately `1.000000` for all four twins | each `>= 1.05` | fail |
| no-history maximum score delta | `0.0` on 1,010 requests | `<= 0.0` | pass |

The selected CHHT checkpoint is epoch 1.  The four controls select epoch 2.
No setting, threshold, subset, seed, or checkpoint rule changed after outcome.

## Collapse diagnosis

A read-only rescore of the same already-opened 3,000 train-internal requests
shows why norm-based responsiveness was misleading:

- every history-present candidate core is at the fixed Frobenius cap `0.35`;
- true, wrong, shuffled, coarse, and query-masked histories all have core-norm
  ratios indistinguishable from one;
- the history-present score delta has RMS about `1.5`, at the tanh score cap,
  but the mean within-request candidate delta range is only
  `2.44e-6` on non-repeat and `2.52e-6` on repeat requests;
- the frozen base's adjacent score-gap median is `0.01484`;
- under the shared evaluator's candidate-hash tie-break, only 3/1,112
  non-repeat and 6/878 repeat requests change any ordering, and zero requests
  change top-10 membership.  Packed stable argsort gives 2+6 instead of 3+6;
  this tie-only diagnostic difference does not affect the gate.

Thus CHHT does not suffer C06's tiny absolute write.  It exhibits the opposite
failure: a **large, saturated, almost candidate-common translation**.  The
skew/Cayley core is candidate-conditioned algebraically, but training uses the
available radius without learning candidate-relative ranking direction.  A
bounded update norm is neither abstention nor evidence fidelity.

The saturation occurs in three stages.  Before its first `tanh`, `rho` has
mean absolute logit `16.32` and is 100% saturated to `+/-1`; the `a/b` event
coordinates are within `0.001` of the endpoints for 74.7%/64.9% of entries,
and `rho` has zero candidate-centred RMS.  The raw skew Frobenius norm then
averages `5.6407`, so every history-present candidate is projected to radius
`0.35` with mean scale about `0.06205`; candidate cores have cosine about
`0.99999999` to their request mean.  The corruption loss compares this
post-projection norm, so all twins receive radius `0.35` and its radial signal
cannot separate them.  Finally, the raw score residual has mean absolute value
`5.834`; `1.5*tanh` maps it to about `+1.499974` with local derivative only
about `5.2e-5`, shrinking a mean raw candidate range `0.0482` to `2.47e-6`.

This is not a no-learning artifact: the score-head norm moved from about
`0.200` to `0.526`, and other Transformer/FFN parameters moved.  The listwise
and preservation objectives are invariant to request-common translation, and
the architecture does not centre the write before the final tanh, so training
preferentially amplified the common component instead of a ranking direction.

## Evidence boundary and terminal action

The continuation opened only the pre-registered last 3,000 train rows and
their train labels.  A label-free dev feature store already existed from the
initial attempt, but this continuation produced no dev score file, created no
C02 run directory, made zero shared evaluator calls, and read no dev qrels,
test records, test labels, test qrels, or test metrics.  The dev-eval log SHA256
remains `54b3760a...0cb2bb44`.

C02 is closed.  Do not relax the core radius, score cap, corruption ratio,
checkpoint selection, or internal thresholds; do not rerun this cohort and do
not use dev to rescue it.  The binding machine-readable summary is
`reports/pps_c02_mechanical_continuation_gate.json`.
