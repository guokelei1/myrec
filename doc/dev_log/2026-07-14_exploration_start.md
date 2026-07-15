# Exploration start — 2026-07-14

## Question

How should the project explore freely without treating an early Lite outcome
as evidence about Full or losing the ability to make a later confirmation
claim?

## Action

Separated open exploration from frozen confirmation in doc 34 and the active
manifest. Authorized source audits, ordinary-model pilots, and logged dev
diagnostics; kept test, independent confirmation, and proposed architecture
closed.

## Direct observations

- Local data include KuaiSearch Lite, Amazon-C4 with a temporal history
  companion, and JDsearch schema samples.
- KuaiSearch Full and KuaiSAR are not local.
- Candidate-relative history-response metrics and their unit tests already
  exist, but no current standardized dataset or score bundle exists.
- The prior E0 draft treated admission as a one-way gate before exploration,
  which would make a Lite insufficiency too easy to overgeneralize.

## Interpretations and uncertainty

The active workspace is ready for data and instrumentation exploration, not
yet for a scientific direction claim. The absence of Full data is a resource
gap, not negative evidence. Existing metrics are mechanically tested, but no
real score bundle has established their numerical behavior or usefulness.

## Correction

Dataset admission and fixed stopping rules now belong to confirmation. During
exploration, each dataset result keeps its own scope and can motivate a new
probe without erasing prior observations.

## Next action

Run a streaming source audit on KuaiSearch Lite to measure actual slate,
history, repeat-query, recurrence, label, and natural-collision opportunity.
Use the result to specify what Full must be compared against; do not decide
whether Full works from Lite.

## Selective archive reference

Need: avoid rediscovering KuaiSearch request-key and contiguous ranking-row
parsing mistakes while writing the new source audit. Consulted only:

- `archive/legacy_20260714/source/src/myrec/data/data/kuaisearch_audit.py`;
- `archive/legacy_20260714/source/src/myrec/data/data/kuaisearch_leakage.py`;
- `archive/legacy_20260714/source/src/myrec/analysis/r0_context_collision.py`.

The new implementation reuses the source field boundary and the general idea
of log-internal temporal cross-reference. It does not import archive code,
thresholds, old gates, cohorts, or model outcomes. Its exact-query opportunity
scout and observation output are newly implemented and unit-tested in the
active tree.

## Integrity correction during the first run

The first complete source run revealed that
`recall_lite/train.jsonl` physically contains both source `train` and source
`test` rows. The initial implementation aggregated both and therefore accessed
click/purchase fields on held-out source rows. It did not score a model or
compute a test metric, but this still violated the intended test-label closure.

That first report is invalidated and must not support any observation. The
scanner was changed to count only the split name and discard non-train rows
before accessing query, user, candidate, or behavior fields. A regression test
uses malformed held-out behavior fields to prove they are not interpreted, and
the public entry point is locked to `split=train`. The corrected report
supersedes the first run.

## Corrected KuaiSearch Lite source observation

Artifact:
`reports/history_response_gap_kuaisearch_lite_source_audit.json`.

Command:

```bash
python scripts/audit_kuaisearch_source.py \
  --raw-dir data/raw/kuaisearch \
  --report-path reports/history_response_gap_kuaisearch_lite_source_audit.json \
  --collision-query-limit 500 \
  --collision-requests-per-query 50 \
  --rank-history-sample-size 1000
```

Direct observations from source-train only:

- the file contains a large repeated-exact-query population;
- most requests have at least one same-user interaction strictly earlier in
  the observed recall log, and most of those requests are strict-nonrepeat;
- an outcome-free scout over frequent exact queries finds many cross-user
  requests sharing at least two nonrepeat candidates;
- raw ranking history has no per-event timestamps; most sampled raw history
  items cannot be located in the recall log, and some locatable items appear
  only after the target request;
- locating dispersed rank histories requires a full scan of a 16.7 GB JSONL
  file, so Full exploration needs an indexed/columnar intermediate boundary.

Interpretations:

1. Lite contains enough structural opportunity to begin mechanism and metric
   exploration. Full is not required to create the first pilot population.
2. Collision pair count is highly dependent because queries, requests, users,
   and candidates are reused. Query-cluster coverage and concentration matter
   more than the raw pair count.
3. Raw `recently_*` fields cannot presently carry a strict causal-history
   claim. The active pilot should reconstruct history from recall events with
   `event_time < request_time` and explicitly retain its missing-log caveat.

Explanations that weaken the thesis remain live: available collisions may not
contain opposite or predictable preference directions; repeated queries may
be dominated by popular head traffic; strict-nonrepeat can be misclassified
when prior events fall outside the observed log; and a strong ordinary model
may use the remaining signal correctly.

Next reversible probes:

1. quantify how collision opportunities concentrate by exact-query cluster;
2. materialize a source-train-only, time-split, label-isolated Lite scout using
   reconstructed prior history;
3. run simple query-candidate and history witnesses before expensive LLM
   training.

Selective adapter reference: while implementing the scout, consulted
`archive/legacy_20260714/source/src/myrec/data/data/kuaisearch_standardize.py`
only for the raw field names and the prior-event reconstruction boundary. The
active scout uses the current record contract, source-train-only filtering,
physical dev qrels separation, and a new test; it does not restore the old
window, split, thresholds, manifests, or outputs.

## Lite scout and simple-control observation

The first local standardized version is
`data/standardized/kuaisearch/lite_scout10k_v1/`; its tracked summary is
`reports/history_response_gap_kuaisearch_lite_scout.json`. It uses the latest
source-train time window after a label-free candidate-size filter, with an
earlier train and later dev split. It contains no official source-test rows.
Dev records are label-free and qrels are physically separate.

Four exploratory dev calls used one candidate manifest: source order,
train-click popularity, recent behavior, and request-local BM25. Their raw run
state is under `runs/`; surface comparisons are the tracked
`reports/history_response_gap_scout_recent_vs_*.json` files.

Direct observations:

- query-text BM25 and recent-history scoring both beat the weakest controls on
  the overall scout, so neither query relevance nor history is completely
  absent;
- the recent-history advantage over popularity is localized to requests where
  a candidate item repeats in history;
- on strict-nonrepeat requests, recent-history scoring shows no advantage over
  popularity or BM25 within the exploratory intervals;
- the latest-window cohort has substantial history-present and strict-
  nonrepeat coverage, but its recency selection may overrepresent active users
  and current head traffic.

Interpretation: the simple witness currently recovers recurrence, not the
nonrepeat query-conditioned direction needed by the paper motivation. This
does not show that such direction is absent, and it does not say how a full-
token Transformer behaves. Plausible alternatives are that lexical/category
history transfer is too weak, the reconstructed log misses relevant events,
the scout window is unrepresentative, or a strong model handles the signal
correctly.

Next action: materialize label-free true/null/matched-wrong assignments and run
one ordinary zero-shot full-token cross-encoder instrumentation pilot. Its
role is to validate response mechanics and expose what training/data work is
needed; it cannot count as an adequate model family.

## Lite full-token instrumentation and supervised pilot

Artifacts:

- `reports/history_response_gap_lite_scout_history_assignments.json`;
- `reports/history_response_gap_lite_scout_bge_base_zero_shot_surfaces.json`;
- `reports/history_response_gap_lite_scout_bge_base_pairwise_pilot_surfaces.json`;
- paired QC/BM25/FULL reports under
  `reports/history_response_gap_lite_scout_*.json`.

The stronger planned v2-m3 checkpoint download stalled before any scoring.
Because the user had not prohibited model downloads, this was a local scheduling
choice rather than a policy boundary: the first mechanical pilot switched to the
fully cached `BAAI/bge-reranker-base`. Model downloads are authorized for later
adequacy work; KuaiSearch Full data acquisition remains outside this action.

### Counterfactual integrity correction

The first zero-shot true/null/wrong scoring used global candidate buffers. Even
with identical checkpoint and candidate order, dynamic padding made a small
number of identical no-history inputs differ numerically across conditions.
That first evaluator run is invalid. Scoring now flushes at each request slate;
all no-history candidate scores are then exactly equal for true, null, and wrong
conditions. A regression test fixes request-aligned inference as part of the
counterfactual signature.

### Direct observations

1. The corrected zero-shot ranker actively changes candidate-relative scores on
   almost every history-present request, but has near-chance direction and harms
   strict-nonrepeat utility. This validates the instrumentation, not the thesis,
   because a zero-shot history format is out of distribution.
2. A matched supervised pilot trained independent QC and FULL checkpoints from
   the same base, on the same train-only positive/negative pairs and the same
   optimizer-step budget. No dev qrels were read during training or scoring.
3. Supervised FULL-true improves over FULL-null overall and is competitive with
   the local BM25 control. The overall improvement is not broad personalization:
   it is concentrated on requests where a candidate repeats in history.
4. On strict-nonrepeat requests, FULL still changes almost every slate, but
   pairwise direction is near chance, signed alignment is near zero, and neither
   true-over-null nor true-over-independent-QC utility is reliable.
5. The label-free repeated-query intersection sharpens the split. Repeated-query
   repeat requests show reliable positive direction and utility; repeated-query
   strict-nonrepeat requests show an adverse point estimate. The latter is an
   exploratory, inspected cohort and not independent confirmation.
6. FULL-null is below independent QC in point estimate, while independent QC is
   not significantly above BM25. This keeps base adequacy and training
   interference as live alternative explanations.

### Interpretation and thesis-weakening explanations

The current Lite evidence supports a narrow working hypothesis: ordinary
full-token training readily learns recurrence, yet its nonrepeat history response
does not reliably become correct candidate-relative ranking direction. It does
not establish a shared Transformer blind spot or an architecture entry.

Explanations that would weaken or close the architecture thesis remain active:

- BGE-base plus one fixed pairwise recipe may be an inadequate query-candidate
  ranker; a stronger ranking-pretrained checkpoint may remove the gap;
- ordinary pointwise/listwise objectives, history dropout, or truncation tuning
  may solve it without architecture;
- most wrong-user assignments use global rather than exact-query donors;
- reconstructed histories omit events outside the observed recall log;
- the strict-nonrepeat labels may contain little recoverable personalized signal,
  in which case a signal witness should fail and the architecture story closes;
- the latest-window Lite scout and single seed may not represent Full.

### Mechanical training corrections

Two direct-fp16 attempts were rejected before producing a usable checkpoint:
one failed at gradient unscale, and the next showed asymmetric overflow skips.
The active recipe uses fp32 master parameters with A40-native bf16 autocast.
Only the two completed bf16 checkpoints count as model results.

### Next action

Do not formulate a model. First make encoder QC adequate with a stronger
ranking-pretrained checkpoint under a small frozen recipe budget. Then compare
matched pointwise and pairwise/listwise QC/FULL recipes and build a train-only
recoverable-direction witness on strict-nonrepeat. The gap advances only if a
strong base, ordinary objective controls, and recoverability all survive.

After the Lite pilot was frozen, the user explicitly confirmed that model
downloads are allowed. `BAAI/bge-reranker-v2-m3` was downloaded through ordinary
HTTP after the earlier Xet transfer stalled. Commit
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` now passes a
`local_files_only=True` load check. This changes readiness, not evidence: no
v2-m3 training or dev evaluation has been run yet.

## v2-m3 encoder adequacy exploration

The first v2-m3 QC score reused the FULL-null serialization
`query + [PRIOR USER HISTORY] + (empty)`. This is information-equivalent to a
query-only model but is not a clean zero-shot QC input. It was therefore kept as
a valid scored run but removed from the QC adequacy role. The scorer and trainer
now support `query_only_text_v1`, for which the first sequence is the unmodified
query. FULL-null intentionally keeps the structured empty-history input because
it must use the exact FULL checkpoint and counterfactual signature.

Direct observations:

- v2-m3 zero-shot remains below BM25 after correcting to pure-query input;
- two train-only pairwise adaptations used identical examples, steps and seed,
  differing only in learning rate;
- the lower learning rate gives the best v2-m3 QC point estimate and better
  purchase ranking, while the higher rate has lower training loss but worse dev
  ranking;
- paired uncertainty places the lower-rate checkpoint statistically alongside
  BM25, not reliably above it, and it is not reliably above BGE-base QC either.

Interpretation: v2-m3 adapts to the dataset, but this small recipe budget has not
established a superior strong base. Rather than keep tuning after each dev look,
the lower-rate checkpoint is provisionally selected for one matched FULL
discriminator. If a higher-capacity FULL still shows a repeat-only gain, the
phenomenon strengthens; if it fixes strict-nonrepeat, weak capacity/base quality
was a sufficient explanation. This selection remains development evidence and
does not freeze a confirmation checkpoint.

### Token-budget correction before matched FULL

A label-free v2-m3 tokenizer audit showed that the ten-event serialization
exceeded 512 tokens for a large fraction of history-present pairs. Because the
history context is normally the longer side of the pair, default longest-first
truncation can discard its tail, which contains the most recent serialized
events. A v2-m3 FULL run already in progress under budget 10 was therefore
interrupted without saving a checkpoint.

History budgets 4, 6 and 8 were audited before reading another outcome. Budget
6 preserves more events than budget 4 while producing zero measured overflow on
both Lite train and dev; budget 8 retains a small overflow tail. The active
matched FULL recipe uses the six most recent events. This is an input-integrity
repair, not a post-outcome rescue, because selection used only tokenizer lengths.

## Matched v2-m3 FULL discriminator

The zero-overflow history-budget-6 FULL model was trained with the same train
pairs, optimizer-step budget, seed, learning rate and encoder initialization as
the provisionally selected pure-query QC. Its true/null/wrong score bundle passed
all candidate and scoring-signature checks. All candidate scores for target
requests without history are exactly equal across the three conditions.

The stronger encoder reproduces the earlier qualitative split. True history
improves overall ranking over null, but most score-response variance is common
mode and candidate-pair direction is only near chance. Surface analysis localizes
the reliable gain to exact recurrence. Strict-nonrepeat requests change almost
universally, yet their directional alignment is not reliably positive, their
true-minus-null utility interval crosses zero, and FULL-true does not outperform
the independently trained QC there. Repeated-query strict-nonrepeat and the
same-query other-user donor intersection do not remove the gap.

This is stronger Lite mechanism evidence, not an architecture Failure Card. The
selected QC is comparable to BM25 rather than demonstrably superior, only one
ordinary pairwise objective and seed have been tested, no second model family or
dataset has replicated the result, and recoverable nonrepeat personalized signal
has not been established. Architecture remains unauthorized. The next cheapest
discriminator is a train-only strict-nonrepeat recoverability witness; only if it
succeeds should one matched pointwise or listwise objective control be trained.

## Fixed-response-budget direction intervention

To test the motivation directly rather than infer it only from pair accuracy, a
label-oracle diagnostic was added. For every request it holds the null scores,
candidate slate, Transformer checkpoint, and exact multiset of observed
true-minus-null score deltas fixed. It compares the actual candidate attribution
against random delta permutations and a gain-aligned reassignment. The latter is
explicitly an analysis upper bound, not a model or proposed method.

On strict-nonrepeat, the actual delta attribution is not reliably better than a
random permutation of the same values under query-cluster bootstrap. In contrast,
gain-aligned reassignment of the unchanged values exposes a large positive ranking
headroom. Repeat requests act as the positive control: their actual attribution is
reliably above random and converts more of the available direction headroom.

This strengthens the narrow Lite motivation: the Transformer is not merely
insensitive to history. It produces ample candidate-relative variation, but on
nonrepeat transfer that variation is allocated almost as if candidate direction
were random, with measurable ranking opportunity left unused. The diagnostic does
not prove recoverability, architectural cause, or cross-dataset generality. A
matched pointwise QC/FULL control is now running to test whether the effect is a
pairwise-objective artifact; train-only recoverability remains the next auxiliary
test. Full remains an independent future population rather than a gate decided by
Lite.

## Pointwise objective replication

QC and FULL v2-m3 models were trained from the same initialization, sampled train
pairs, optimizer-step budget and seed as the pairwise controls; only the loss was
changed to ordinary pointwise binary cross entropy. Counterfactual scoring again
passed candidate coverage and exact no-history equivalence.

The central pattern survives. FULL-true improves overall and clearly improves the
repeat surface, but does not beat matched QC on strict-nonrepeat. Strict-nonrepeat
pair direction is indistinguishable from random, true-minus-null and
true-minus-wrong utility intervals cross zero, and actual fixed-delta attribution
does not beat random permutation. Gain-aligned attribution of the unchanged delta
multiset still exposes large label-oracle headroom.

One mechanism claim does not survive intact: pointwise BCE has much less common-mode
energy than pairwise training, even though direction failure remains. Therefore the
motivation should be stated as failure to convert history response into correct
candidate-relative direction. Common-mode shift is an objective-dependent symptom,
not the defining mechanism.

## Train-only recoverability witness v1

A separate diagnostic used frozen `bge-small-zh-v1.5` query/item/history embeddings,
explicit category/brand/recency/query-conditioned similarity features, and nested
base-only versus base-plus-history gradient-boosting models. It trained only on
strict-nonrepeat train requests with clicks and did not read dev labels during
training or scoring.

The witness does not establish recoverability. It produces repeat gains, but on
strict-nonrepeat its direction and true-minus-null/true-minus-wrong utility are not
reliable, and full-true does not beat the nested base. This negative result is
inconclusive because the base is weaker than BM25 and the full-null path itself
degrades. It narrows what can honestly be claimed: the label-oracle intervention
shows direction headroom, not proven learnable headroom.

The first scoring implementation invoked the tree separately for every request and
was stopped after producing five incomplete, metadata-free runs. Batch prediction
created the complete `*_v2` bundle. No incomplete run was passed to the evaluator.
The next motivation replication should change Transformer family or dataset rather
than add another v2-m3 objective. KuaiSearch Full remains independent and pending.

## Qwen3 decoder-family replication

`Qwen/Qwen3-Reranker-0.6B` was added as a decoder-only causal-LM reranker through
its official SentenceTransformers CrossEncoder interface. No KuaiSearch Full data
was downloaded. A tokenizer-only length audit was replaced by an audit of the
model's real instruction/chat preprocessing: history budget six has a small dev
overflow tail, whereas budget five has zero measured overflow on both Lite train
and dev. The already scored budget-six zero-shot bundle was never evaluated.

The clean budget-five zero-shot bundle showed the candidate strict-nonrepeat
direction symptom, but repeat direction and utility were not reliable. It was
therefore excluded from cross-family evidence: an unadapted decoder cannot serve
as proof that a task-trained Transformer has the claimed failure.

Matched QC and FULL Qwen3 models were then adapted using only train labels and an
ordinary pointwise binary-cross-entropy objective. They share initialization,
sampled positive/negative pairs, seed, candidate presentations and optimizer
updates. The FULL input alone contains the five-event causal history. Both
checkpoints have finite parameters and outputs. The true/null/wrong bundle has
complete and equal candidate keys, and all 8,649 no-history candidate scores are
exactly equal across conditions.

The supervised decoder supplies the missing positive control. It gains reliably
from true history overall and on exact-repeat requests; repeat pair direction is
also reliably above chance. On strict-nonrepeat, however, almost every request and
candidate pair changes while direction remains statistically indistinguishable
from chance. True history has no stable advantage over null, wrong-user history,
or independently trained QC there. Reassigning the exact observed delta multiset
to random candidates performs statistically alongside the actual attribution,
whereas label-aligned reassignment exposes large diagnostic headroom.

This matches both pairwise- and pointwise-trained v2-m3 encoders. The scoped Lite
motivation is therefore supported across an encoder and a decoder-only Transformer:
ordinary LLM4Rec rankers can read history and learn recurrence, yet fail to convert
nonrepeat history response into reliable query-conditioned candidate-relative
direction. The result does not show significant absolute strict-nonrepeat
degradation; it shows failure to realize stable history gain and a large oracle
allocation gap. It also does not establish that the gap is learnable or caused by
an unavoidable Transformer primitive.

Qwen QC did not improve over its zero-shot point estimate, although the paired
interval crosses zero. This limits any strong-baseline claim but does not erase the
repeat positive control or the matched FULL-null/QC comparison. The next useful
probe is independent-population replication, not architecture construction.

The first decoder training implementation accumulated loss but omitted its mean
from the saved metadata. Sample counts, optimizer updates, hashes and checkpoint
integrity remain recorded; the source now writes the mean for future runs. The
completed pair was not retrained or selected from dev outcomes.

## Independent KuaiSearch Full-source population replication

The user-prepared KuaiSearch Full files were not downloaded by this run. Their
official-source manifest, bounded schema and slate-integrity audit, and recall-log
history opportunity were checked locally. An independent latest-window 10k scout
was then materialized from the Full files rather than copied from Lite. It has
complete item joins, causal pre-target history, a session-disjoint time split,
isolated dev labels, and substantially more strict-nonrepeat requests than the
Lite scout. This remains an exploratory population sample, not whole-Full
admission or confirmation.

Before any Full dev result was opened, the Lite-selected v2-m3 pairwise recipe was
copied unchanged. Matched QC and FULL training used identical examples,
initialization, optimizer budget and seed; the history budget passed zero-overflow
train/dev tokenizer audits. True, null and wrong score bundles have identical
candidate keys, and every no-history candidate score is exactly invariant.

The important qualitative split replicates. Repeat requests provide a strong
positive control: true history improves direction and ranking utility over null,
wrong-user history, random delta attribution and QC. Strict-nonrepeat requests
also respond almost universally, but directional accuracy remains near chance and
true history has no stable advantage over null, wrong-user history, or QC. A
same-query other-user subset reaches the same inconclusive provenance result.

One Lite formulation does **not** replicate literally. On the Full-source scout,
actual strict-nonrepeat delta attribution is slightly but significantly better
than random reassignment of the same deltas. Its effect is nevertheless small:
the diagnostic converts only about 3.6% of the label-aligned fixed-response
headroom, while the remaining oracle allocation gap is large. The primary
cross-population claim is therefore revised to:

> Ordinary LLM4Rec Transformers reliably react to history and exploit recurrence,
> but on nonrepeat personalization they convert only a small and
> population-unstable fraction of that response into query-conditioned correct
> candidate-relative ranking utility.

This is stronger than a single-Lite symptom and more honest than calling every
strict-nonrepeat response random. It still does not prove an unavoidable
Transformer limitation, learnable oracle headroom, significant absolute
degradation, whole-Full generality, or a valid architecture entry. The next
discriminating probe is independent-dataset replication, using Amazon-C4 as an
English strict-nonrepeat stress test while explicitly recording that it lacks a
natural repeat positive-control surface.

## Full-source history-query restoration

An information-object probe added the original issued query to every retained
KuaiSearch Full history event while holding request queries, candidates, qrels,
non-query history fields, recipe and seed fixed. A label-free token audit required
max length 768 to retain all six events. The enriched history did not improve
strict-nonrepeat direction conversion or true-over-null utility; repeat remained
the positive control. Missing prior queries are therefore rejected as a
sufficient adapter explanation.

## Amazon-C4 positive boundary and length tradeoff

A fresh Amazon-C4 plus Reviews-2023 materialization provided constructed queries,
roughly one hundred BM25 candidates, and temporal purchase histories. At two
events, the ordinary v2-m3 FULL model produced strongly correct direction and
large true-over-null/wrong utility, refuting universal direction failure. Its
true result was only at QC parity because FULL-null had a large base deficit.

A pre-outcome label-free coverage probe increased the budget to eight events and
max length to 1024. Direction and counterfactual history utility improved, but
base erosion grew even more. FULL-true became significantly worse than QC and
the two-event FULL model. More observable/useful history therefore did not imply
safer end ranking.

An Amazon QC/FULL by pure-query/structured-empty factorial found a
checkpoint-dependent formatting interaction, but FULL weights remained far
below QC weights under both formats. As on JDsearch, the large base deficit is
not an empty-history-string artifact.

## JDsearch independent functional replication

The official JDsearch files and mirror were verified against documented source
counts and samples. The first materialization was invalidated before admitted
model outcomes because source candidate order encoded labels perfectly. A
seeded, label-free hash order replaced it. A second pre-outcome correction
canonicalized uppercase behavior codes; only `hash_scout10k_v3` is active.

Matched QC/FULL v2-m3 pairwise models used the same examples, seed and optimizer
budget. QC beat BM25; FULL-true beat QC and recent behavior overall. The apparent
success was recurrence-dominated. Strict-nonrepeat true history had real utility
and direction above chance, but fixed-response direction conversion remained near
four percent and far below repeat. More importantly, FULL-null's loss relative
to QC was larger than true history's gain relative to FULL-null, leaving
FULL-true no better than QC on strict-nonrepeat traffic.

The first wrong-user score attempt exposed one donor whose first sequence could
not fit `only_second` max length. No incomplete run was evaluated. A label-free
context guard removed the oldest effective event for that one request only; all
other effective wrong inputs were unchanged. The completed wrong bundle supports
a broad true-over-wrong effect, but the smaller same-query strict-nonrepeat subset
does not establish provenance.

A two-by-two QC/FULL weight by query-only/structured-empty input control rejects
the empty-history marker as the main base-loss explanation. Format effects are
small and unstable; the FULL weight deficit is large and significant under both
serializations.

## Motivation decision

The universal response-direction failure claim is closed. The surviving
cross-dataset problem is controlled history composition:

> Ordinary full-token LLM4Rec rankers can read history, but do not reliably
> preserve query-candidate base capability while adding a high-efficiency,
> candidate-relative history update. Positive true-over-null utility can repay a
> self-created base deficit, and easy recurrence can hide the failure overall.

KuaiSearch exposes the direction-allocation side, Amazon exposes the base/history
tradeoff despite excellent direction, and JDsearch connects both through exact
base-retention plus history-utility accounting. This is sufficient for an
exploratory paper motivation and insufficient for architecture authorization.
Simple standard repairs, train-only recoverability, an independent family and
frozen confirmation remain the next gate.
