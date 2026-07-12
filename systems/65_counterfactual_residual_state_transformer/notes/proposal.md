# C65 proposal — counterfactual residual-state Transformer

Status: pre-outcome architecture proposal.  C64 validation labels and every
fresh role remain unopened.

## Observation → architecture consequence → falsification

**Observation.** C64 proved that adapting pretrained BGE layers makes the
joint ranker highly active, but correct-to-wrong history replacement was not
consistently Top-10-load-bearing.  The adaptive history model could learn most
of the same generic query-candidate reranking function as its history-free
control.  More LM capacity therefore amplifies the shortcut rather than making
personalized evidence identifiable.

**Architecture consequence.** C65 removes the generic output path.  One shared
adaptive LM and directed joint Transformer computes candidate states under the
factual and structurally NULL histories:

```text
z_H    = JointLM(q, H, candidates, base_coordinate)
z_0    = JointLM(q, NULL, candidates, base_coordinate)
r_H    = LayerNorm(z_H - stopgrad(z_0))
delta  = center_candidates(head(r_H))
score  = strong_base + delta.
```

When history is absent, both paths coincide and the personalized output is
structurally skipped, so the base is exact.  The NULL branch is a representation
reference, not a second scorer or fixed-score router.  Ranking gradients train
the factual LM path end to end; the stopped NULL state prevents the reference
from moving merely to manufacture a residual.

For a matched wrong history `H_w`, the same network forms `r_w`; training adds a
rank-neutrality penalty on its centered output.  The true branch still learns
only the ordinary listwise target.  This does not assert causal user identity:
wrong donors are a counterfactual specificity control and training object.

## Matched modes

- `hidden_residual_wrong_neutral` (primary): internal factual-minus-NULL state
  residual plus wrong-history neutrality.
- `hidden_residual_no_wrong`: identical forward architecture without the
  wrong-neutrality term, isolating the training object.
- `ordinary_factual_wrong_neutral`: head reads factual candidate state directly
  while retaining the same wrong-neutrality loss, reproducing C64's shortcut.
- `logit_difference_wrong_neutral`: subtracts shared-head factual/NULL logits,
  the closest C04/classifier-free-guidance reduction.

All modes instantiate the same pretrained LM, trainable last two layers,
directed joint Transformer, residual normalization, and head.  They share
initialization, sampled candidates, optimizer, steps, and strong base.

## Falsification

Before validation labels, every seed must pass exact no-history/repeat,
candidate permutation under frozen fp32 scoring, determinism, finite gradients,
candidate hashes, and true/wrong Top-10 activity.  A1 then requires primary to
beat the strong base, wrong history, and all three modes with positive interval,
seed, and fixed-fold evidence.

Failure closes this hidden-state residual plus wrong-neutrality primitive.  No
loss-weight, stop-gradient, LM-depth, precision, epoch, history-length,
candidate-sampling, seed, sign, scale, or cohort rescue is allowed.

## Innovation and reduction boundary

Factual/NULL subtraction itself is not novel: C04 tested paired-prefix logit
deltas and C61 tested an edge likelihood ratio.  C65's provisional primitive is
the combination of (i) a load-bearing internal candidate-state residual that
removes the generic factual-state path, (ii) a stopped shared reference, and
(iii) wrong-history rank neutrality applied only to that residual.  The three
binding controls must establish that each part matters.  Otherwise C65 is a
reimplementation of known counterfactual/logit-difference training and is
rejected.

## Outcome boundary

C65 may use C64's 4,800 exposed training requests.  The same 1,200 exposed-fit
validation labels were never opened by C64 and remain closed through C65
training and label-free A0.  C26 internal-A, delayed roles, Amazon reserve, dev,
test, and qrels remain closed regardless of C65's outcome.
