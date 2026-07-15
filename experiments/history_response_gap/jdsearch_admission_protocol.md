# JDsearch exploratory admission protocol

Date frozen: 2026-07-14, before inspecting any JDsearch model outcome.

Status: acquisition, provenance verification, source audit and ordinary-model
exploration authorized. Proposed architecture, confirmation and test claims
remain unauthorized.

## Pre-model candidate-order amendment

The first difficulty audit found NDCG@10 = 1.0 for the released candidate
order. A full source audit then established that every row with a positive
label places all positive candidates in a prefix. This is a release-format
label-order artifact, not a usable production-rank feature. No Transformer had
been trained or evaluated on JDsearch when it was found.

Version `hash_scout10k_v1` is therefore invalid for model work. The sole v2
amendment is a deterministic seeded ordering by `(request_id, item_id)` hash,
which never reads candidate labels. The paired labels move with their item only
inside the data materializer and remain physically absent from development
records. Source position exposed to methods is replaced with this permuted
position. Re-running source-order after v2 must remove the perfect-rank signal;
failure closes JDsearch before neural training.

Before either v2 training completed or produced a checkpoint/model outcome, a
second input audit found that the release codes `ORD` and `FLW` had only been
lower-cased. Version `hash_scout10k_v3` canonically maps `ORD -> purchase`,
`CLICK -> click`, `CART -> cart`, and `FLW -> follow` while retaining the v2
outcome-blind candidate order. Both in-progress v2 trainings were interrupted
without saving checkpoints. All JDsearch model evidence must use v3.

## Why JDsearch is the replacement source

JDsearch is a SIGIR 2023 dataset created specifically for personalized product
search. It supplies real issued queries, displayed product candidates, graded
candidate interactions and each user's earlier query/product behavior. Recent
sequential personalized-product-search work evaluates on the established Amazon
PPS construction and JDsearch together. It therefore matches the paper's task
community and gives a platform-independent complement to KuaiSearch.

The official distribution is difficult to automate from JD Cloud. A public
Kaggle mirror may be used only if its two extracted files match the official
repository's documented names, line formats, sample records, counts and license.
Mirror mismatch closes that copy before model training.

JDsearch text is anonymized term IDs. It can support a functional claim about
query-conditioned personalized product ranking, but not a claim about failure
to exploit pretrained plaintext semantics.

## Outcome-blind admission conditions

Before any model evaluation, establish all of the following:

1. each row contains one real query, aligned candidate IDs and labels, earlier
   query/product/type/time sequences, with the documented list lengths;
2. candidates are unique after the official duplicate-removal convention and
   each retained request has at least one positive and one zero-label candidate;
3. product metadata joins candidates and retained history at sufficient coverage;
4. history order is causal by the documented sequence/interval semantics;
5. method-visible development records are physically label-free and candidate
   hashes are invariant across true, null and wrong histories;
6. a stable outcome-blind request sample and user-disjoint train/development
   split are frozen before evaluating model scores;
7. history-present strict-nonrepeat development requests are numerous enough
   for clustered uncertainty estimates.

## Frozen first-pass contract

- Stable scout size: at most 10,000 eligible source rows selected by a seeded
  hash of immutable source-row identity, not by model outcome.
- Split: 80% train and 20% development by a second stable hash. Because the
  official release contains one target query per sampled user and no absolute
  comparable target timestamp, no false chronological split claim is made.
- Ordinary model: matched BGE reranker v2-m3 QC/FULL pairwise controls using the
  existing recipe. Token/history budget may change only from label-free real-
  preprocessing coverage audits.
- Counterfactuals: the same FULL checkpoint under true, null and matched
  wrong-user history on the identical candidate slate.
- Main surface: history-present strict-nonrepeat. Repeat requests are a positive
  control if adequately populated.
- Labels: graded interaction is primary; click-or-stronger is a robustness view.
- Diagnostics: candidate-relative activity, active-pair direction, true-minus-
  null/wrong NDCG@10, FULL-null versus independently trained QC, and fixed-delta
  attribution intervention, all through the shared evaluator.

## Interpretation and stopping

No individual dataset has veto power over another. JDsearch can replicate,
narrow or contradict the prevalence of the KuaiSearch pattern. Amazon-C4 remains
reported as a positive boundary case. Cross-dataset support requires the gap on
more than one independent source, not unanimity across every information object.

Stop before training on provenance/schema/admission failure. If the admitted
scout is underpowered, enlarge the frozen hash sample without interpreting its
model outcome. If the ordinary FULL model is inactive on both repeat and
nonrepeat, treat it as mechanics/learnability failure rather than direction-gap
evidence. Do not change datasets merely because the result is unfavorable.
