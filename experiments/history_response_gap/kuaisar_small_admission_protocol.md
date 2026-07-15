# KuaiSAR Small exploratory admission protocol

Date frozen: 2026-07-14, before inspecting any model outcome on KuaiSAR.

Status: **closed before model training**. Source acquisition exposed a
non-contiguous-session adapter issue, but no model outcome was produced. The
user subsequently narrowed the dataset criterion to established personalized
product-search benchmarks from a distinct platform. KuaiSAR is therefore not
used as the next replication source; this closure is a scope decision, not a
positive or negative motivation result.

## Why this dataset is next

Amazon-C4 is retained as a semantic positive control, but its generated query
and one-positive retrieval construction do not reproduce natural search-slate
competition. KuaiSAR is selected next because it records real search sessions,
shown items and clicks together with earlier search/recommendation behavior.
The Small release is used first as the cheapest source/schema and power probe;
Full is used only if Small passes admission and is underpowered or yields a
scientifically material pattern worth replicating.

KuaiSAR uses anonymized query/caption token IDs. It can therefore support a
functional claim about query-conditioned sequential ranking, but it cannot by
itself support a claim about pretrained natural-language semantics.

## Outcome-blind source admission

Before training or evaluation, the adapter must establish all of the following:

1. a search request can be reconstructed as one query/session with at least two
   exposed candidates and at least one clicked and one unclicked candidate;
2. every history event is strictly earlier than the target request;
3. query and item token fields can be represented without consulting labels;
4. candidate identity/order and candidate-set hashes are frozen across true,
   null and wrong-history conditions;
5. label-free history-present and strict-nonrepeat surfaces are large enough
   for request-clustered uncertainty estimates;
6. training, development and held-out partitions can be separated by time (and
   session) without target leakage.

Failure of a source condition closes KuaiSAR as a binding replication source;
it does not count as evidence for or against the motivation.

## Frozen first-pass measurement contract

- Ordinary model: the existing ranking-pretrained v2-m3 QC/FULL pairwise recipe,
  adapted only where anonymized token serialization mechanically requires it.
- Counterfactuals: the same FULL checkpoint scores true, null and matched
  wrong-user history on the same request/candidate slate.
- Primary population: history-present strict-nonrepeat requests.
- Positive control: exact-repeat requests, if sufficiently populated.
- Diagnostics: candidate-relative activity, active-pair direction accuracy,
  true-minus-null and true-minus-wrong NDCG@10, FULL-null versus independently
  trained QC, and the fixed-response direction intervention already used on
  KuaiSearch and Amazon-C4.
- Statistics: request-clustered bootstrap confidence intervals with the same
  evaluator and label isolation used by the current exploration.

No single KuaiSAR outcome can veto results from another dataset. The conclusion
will report which information objects exhibit the gap and which do not. A
cross-dataset motivation requires the pattern on more than one independent
source; a KuaiSAR negative result narrows prevalence but does not erase the
KuaiSearch observation.

## Stop and escalation rules

- Stop before model training if source admission fails.
- If Small is admitted and has adequate dev population, finish the frozen
  first-pass bundle before changing the recipe.
- If Small is admitted but underpowered, escalate to Full without interpreting
  the Small model outcome.
- If a normally trained model is inactive on both repeat and nonrepeat, treat it
  as a mechanics/learnability failure, not direction-gap evidence.
- Do not add a proposed architecture, open test labels, or call this independent
  confirmation during this pass.
