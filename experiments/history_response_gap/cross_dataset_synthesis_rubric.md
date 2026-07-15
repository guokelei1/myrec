# Cross-dataset motivation synthesis rubric

Frozen on 2026-07-14 after the KuaiSearch query-history probe and before any
JDsearch Transformer outcome or Amazon-C4 history-budget-8 outcome.

This is an exploratory discovery rubric, not a confirmation endpoint. Its
purpose is to keep an unfavorable dataset as a boundary rather than discard it,
and to prevent one dataset from vetoing or redefining every other result.

## Fixed dataset roles

- KuaiSearch Full-source scout is the natural-language product-search source.
  Its strict-nonrepeat surface is the primary localization surface; recurrence
  is the mechanics/learnability positive control.
- JDsearch v2 is the independent functional personalized-product-search source.
  Its text consists of anonymized terms, so it can replicate a ranking and
  credit-assignment failure but cannot support a pretrained-semantic claim.
- Amazon-C4 is the English semantic boundary. Its query is generated from the
  target review and its candidate pool is constructed, so it tests whether the
  problem survives a much more target-revealing query, not whether all natural
  search traffic behaves identically.

## Primary motivation branch

Call the response-to-direction bottleneck cross-source only if JDsearch has a
working model (nontrivial QC and a repeat positive control) and its
strict-nonrepeat surface jointly shows:

1. broad candidate-relative true-versus-null response;
2. directional accuracy statistically compatible with chance or clearly much
   weaker than its repeat surface;
3. small or unstable true-over-null/wrong ranking utility; and
4. a large fixed-delta label-aligned improvement relative to the actual delta
   attribution.

No single scalar threshold decides this branch. Confidence intervals,
mechanical controls and effect coherence decide it together. Amazon may remain
a positive boundary without invalidating this branch.

## Falsification and redirection branches

- If JDsearch reliably converts nonrepeat history into correct direction and
  utility, the broad cross-source direction claim is rejected. KuaiSearch then
  supports only a natural-query/source-specific failure, unless a shared
  alternative is directly measured.
- If Amazon budget 8 remains strongly directional, limited Amazon history
  length is not a sufficient explanation for its positive result. The boundary
  should be attributed primarily to its target-revealing query/candidate
  construction, with that attribution stated as an inference rather than a
  randomized causal result.
- If Amazon budget 8 loses the budget-2 direction/utility result, history
  dilution or reliance calibration becomes a live motivation. It may replace
  the primary branch only if an analogous length/relevance effect is measured
  on at least one other source.
- FULL-null degradation relative to independent QC is recorded on every source.
  It becomes the main motivation only if it replicates and cannot be explained
  by an inadequate QC/FULL training comparison. It is not silently merged with
  response-direction failure.

## Architecture boundary

This round may establish and localize a problem, not propose a repair. Even a
positive synthesis authorizes only a concise Failure Card stating the affected
surface, surviving alternatives and measurable lost ranking opportunity.
Architecture formulation remains a later step.
