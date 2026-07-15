# Controlled history composition: empirical motivation synthesis

Status: exploratory motivation established; architecture and confirmation remain
locked.

Chinese reader guide:
[`36_controlled_history_composition_reader_guide_zh.md`](36_controlled_history_composition_reader_guide_zh.md).

## Executive decision

The original universal statement—ordinary Transformers respond to history but
cannot give that response the correct candidate-relative direction—is too broad.
Amazon-C4 directly falsifies it, and JDsearch has direction that is weak but
reliably better than chance.

The experiments support a narrower and stronger paper motivation:

> Ordinary full-token LLM4Rec rankers can read and exploit user history, but
> their joint score does not controllably compose query–candidate base relevance
> with a candidate-relative history update. On nonrepeat product search, the
> history response is often low-precision and low-efficiency; at the same time,
> learning to rely on history can erode the underlying base ranker. Consequently,
> a large true-over-null gain may merely repay damage to the null-history path,
> while easy recurrence masks the failure in aggregate metrics.

This is not the claim that history always hurts, that every dataset has random
direction, or that a new architecture is already necessary. It is the claim
that **history responsiveness and true-over-null utility are insufficient
evidence of well-integrated personalization**.

## The accounting that reveals the problem

For a matched ordinary FULL ranker and an independently trained query-only QC
ranker, the net value of personalization can be written exactly as:

```text
FULL-true − QC
= (FULL-null − QC) + (FULL-true − FULL-null)
= base retention      + history utility
```

The usual true-versus-null comparison observes only the second term. It can look
strong even when the first term is more negative. This is not a cosmetic metric
choice: on JDsearch strict-nonrepeat requests and Amazon-C4 with longer history,
history is directionally useful relative to FULL-null, yet FULL-true does not
beat QC because base erosion is larger than the recovered history utility.

JDsearch includes a two-by-two control crossing QC/FULL weights with pure-query
and structured-empty-history inputs. The empty-history marker causes only a
small, statistically unstable change; the large QC-to-FULL loss persists under
both serializations. The JD base loss is therefore localized primarily to the
jointly history-trained weights, not to the literal empty-history text.

The same factorial on Amazon-C4 finds a checkpoint-dependent format interaction,
but the FULL weights remain dramatically below QC under both pure-query and
structured-empty inputs. Across both independent sources, serialization cannot
account for the base deficit created by joint history training.

## Cross-dataset evidence matrix

| Evidence source | What the ordinary Transformer succeeds at | What remains uncontrolled | Role in the story |
|---|---|---|---|
| KuaiSearch Lite and independent Full-source scout | Reads history, changes nearly every candidate pair, and learns exact recurrence | Strict-nonrepeat direction is chance-like or population-unstable; only a very small fraction of fixed response-direction headroom is converted | Primary natural-language direction-allocation failure |
| KuaiSearch Full with prior behavioral queries restored | Uses the richer query–item history and beats QC as a whole | Restoring the missing query does not repair strict-nonrepeat direction conversion | Rejects the missing-history-query adapter explanation |
| JDsearch v3 | Beats BM25, QC, and a recent-behavior control overall; true history has real strict-nonrepeat utility and strong recurrence utility | Strict-nonrepeat conversion is far below repeat; base erosion is larger than nonrepeat history utility, so the net result does not beat QC | Independent functional replication and exact base/history accounting |
| Amazon-C4, short and long history | Produces strongly correct candidate-relative direction and large true-over-null/wrong utility | Increasing history strengthens history utility but erodes base even more; the longer-history FULL model becomes worse than QC and its shorter-history version | Positive boundary that falsifies universal direction failure while exposing the composition tradeoff |

KuaiSearch Lite and Full are different populations from one source, not two
independent datasets. JDsearch is the independent product-search replication,
but its anonymous terms support only a functional ranking law, not a
pretrained-semantic mechanism. Amazon-C4 uses a constructed, target-revealing
query and is therefore a deliberately easier semantic boundary rather than a
natural-search replication.

## What the experiments rule out

The surviving motivation is not based on a single weak checkpoint or a raw-logit
shift.

- Candidate-relative activity is measured after slate centering and is nearly
  universal on the relevant surfaces.
- Repeat requests provide a positive control on KuaiSearch and JDsearch: the
  same checkpoints can learn history, assign useful direction, and improve
  ranking when the answer recurs in history.
- The KuaiSearch Lite pattern survives encoder versus decoder-only Transformer
  families and pairwise versus pointwise encoder objectives.
- Token-coverage audits preserve the query and configured history; request-aligned
  scoring removes dynamic-padding counterfactual drift.
- Adding the original query to every retained KuaiSearch history event does not
  repair the nonrepeat gap.
- Amazon history expansion rules out “two events were too little context” as the
  reason Amazon is directionally successful.
- JDsearch candidate-order leakage was detected before model evaluation,
  invalidated, and replaced with deterministic label-free hash ordering.
- JDsearch's QC/FULL weight gap remains under both pure-query and
  structured-empty serialization, rejecting the empty-marker explanation.
- Amazon-C4 repeats that weight-gap conclusion under both serializations even
  though the direction of the smaller format effect differs by checkpoint.

Wrong-user evidence is deliberately scoped. JDsearch true history beats the
predominantly global donor control on strict-nonrepeat requests, but the smaller
same-query other-user intersection does not establish user specificity. This
prevents the story from overclaiming provenance when donor support is limited.

## Why this is a better CCF-A motivation than the starting hypothesis

The starting hypothesis tried to prove that a standard Transformer could not
read or direct history. That invites easy refutation: Amazon shows that it can,
and JDsearch shows statistically correct nonrepeat direction.

The revised problem is harder and more consequential. A production ranker must
simultaneously preserve query–candidate competence and add only the useful,
candidate-specific part of history. The experiments show that ordinary joint
fusion can succeed on either axis without controlling both:

- KuaiSearch preserves a usable base but allocates the history update poorly;
- Amazon extracts highly correct history direction but damages the base enough
  to lose net ranking quality with more history;
- JDsearch sits between them: nonrepeat history is genuinely useful, yet most of
  its apparent true-over-null gain pays back base capability lost during joint
  training, while recurrence hides the problem overall.

This gives a coherent research object: **base retention and directional history
utility form two independently measurable obligations of LLM4Rec history
integration**. A future method must improve their joint frontier, not merely
increase attention to history, response magnitude, or true-over-null gain.

## What is not yet established

This round establishes a paper-worthy exploratory motivation, not an architecture
Failure Card under doc 31.

The following remain open:

- whether a normally tuned standard objective, history dropout, anchoring loss,
  or other simple training repair removes the base/history tradeoff;
- whether nonrepeat direction headroom is recoverable from the available train
  information rather than only label-oracle diagnostic headroom;
- whether the same joint failure survives a second adequate model family on an
  independent dataset and a frozen confirmation population;
- where inside the Transformer computation the two obligations become
  entangled;
- whether an architectural change is necessary, rather than an objective or
  interface contribution.

Accordingly, no proposed architecture source tree or architecture GPU training
is authorized by this document.

## Next gate

The next stage should freeze one narrow Failure Card around the accounting above,
not reopen broad architecture search. The cheapest discriminators are:

1. a standard training repair control that explicitly tests whether base
   retention and true-history utility can be recovered simultaneously;
2. a train-only, non-oracle recoverability witness for candidate-relative
   direction on the strict-nonrepeat surface;
3. one independent-family replication of the base/history accounting;
4. frozen confirmation only after the effect definition and simple controls are
   fixed.

If a simple repair closes the gap, the contribution should be stated as an
objective or training-interface result. If the tradeoff survives, it becomes a
legitimate entry point for a Transformer architecture hypothesis whose target is
controlled composition, not generic “more history modeling.”
