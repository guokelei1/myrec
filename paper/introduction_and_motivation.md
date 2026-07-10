# Introduction (front half) and Motivating Observations

Date: 2026-07-10

Status: **motivation complete; ready to formulate a bounded proposed-system
design, with implementation/training gated by a new pre-implementation
falsifier**.

Scope: Section 1 through the transition into design formulation. No concrete
model architecture is specified or empirically validated here. All reported numbers are KuaiSearch dev
diagnostics copied from registered artifacts; citation markers remain
placeholders pending the bibliography.

The current argument is closed by the pre-outcome component protocol in
`doc/23_c5r3_candidate_history_alignment_protocol.md`. The temporal C5-R2,
train-frozen matched-history, and M3/M4 oracle arguments are retained as
historical/failed diagnostics and are not used to authorize system design.

---

## 1. Introduction

Product search combines two forms of evidence that are individually familiar
but jointly difficult to use. A query expresses what a user wants now, whereas
past clicks and purchases describe preferences accumulated before the current
request. A personalized ranker should exploit both: it should preserve the
explicit intent in the query while using the target user's history to
distinguish among products that remain plausible for that intent. We study this
problem as **query-conditioned personalized product ranking under a fixed
candidate set**, where every method receives the same query, candidate products,
and strictly-prior prequential user history and must reorder the same candidates.

This setting is not equivalent to either open-catalog retrieval or conventional
sequence recommendation. The first-stage recall system has already conditioned
the candidate pool on the query, so reranking must often separate products that
are all superficially relevant. At the same time, behavioral evidence is not a
generic context feature: it is useful only insofar as it belongs to the target
user and bears on the current candidates. Histories may be absent, short, or
dominated by earlier intents. Applying an unrelated user's history is therefore
not a neutral replacement for personalization, and forcing a history-dependent
model to operate without valid history is not a coherent fallback.

These properties suggest a more precise question than whether query and history
scores can be combined: **what behavior inside a successful history heuristic
actually produces its gain?** A global mixture may look personalized while
mostly rewarding candidates that exactly recur in the observed history; that is
scientifically different from transferring a semantic preference across items
or learning query-dependent event use. Conversely, a per-request oracle over
fixed scorers is not sufficient evidence for any of these mechanisms because
selecting and evaluating a winner on the same labels can manufacture apparent
headroom from ranking noise.

The fixed-candidate protocol makes these distinctions measurable. All methods
use one candidate manifest and one evaluator, histories contain only events
strictly before each request, and scoring code cannot read dev or test labels.
The released `recently_*` fields are not used: they failed a log-internal
future-event cross-check, so history is rebuilt from same-user recall events
whose timestamps strictly precede the request. This construction guarantees
temporal direction within the observed recall window, at the cost of shorter
sequences and empty histories near the beginning of the window.
On KuaiSearch, this setup yields three observations. First, the recall candidate
pools are already strongly query-conditioned. Fine-tuning a compact text tower
improves the zero-shot semantic scorer, and adding a legal train-popularity
prior improves it further, but this non-personalized ranker still falls short
of a static combination with correct-user history. Second, that combination
loses substantially under freshness-matched wrong-user histories in aggregate,
but its same-query identity effect is not significant in two of three seeds.
Third, an exact pre-registered decomposition shows that the stable history gain
is concentrated in exact repeat-item memory: category-only history adds no
significant value, and adding it makes the item-only ranker worse in every seed.
The motivation therefore exposes a strong repeat-item shortcut and motivates a
bounded design question: how to preserve reliable recurrence while preventing
unsupported history transfer from contaminating the query-conditioned ranking.

## 2. Motivating Observations

### 2.1 The candidate pool is already query-conditioned

Query-only rankers are the necessary first control. If an ordinary relevance
model already separated clicked from unclicked candidates, there would be
little reason to introduce personalization. The KuaiSearch diagnostics show
that the query field and scoring pipeline are functioning: for 98.5% of dev
requests, the original query gives the candidate pool a higher mean BM25 score
than a shuffled query. Moreover, for 98.8% of requests, recalled candidates
score above a random catalog reservoir. The fixed pool has therefore already
undergone substantial query filtering before reranking begins.

Within this pool, the off-the-shelf query-only controls cluster. BM25 obtains
0.305 NDCG@10 and is statistically indistinguishable from popularity at 0.301
(difference CI [-0.001, +0.010]). A zero-shot BGE bi-encoder reaches 0.306, and
a zero-shot cross-encoder reaches 0.307 without a significant improvement over
the bi-encoder.

We additionally trained a stronger non-personalized control without using user
identity or history. D2t fine-tunes all four layers of
`bge-small-zh-v1.5` as a query tower against frozen candidate-title embeddings
and obtains a three-seed mean of 0.3141. At the preselected seed it improves
over zero-shot BGE by +0.0083 (95% CI [+0.0044, +0.0123]). Combining D2t with a
train-only item-popularity prior using a train-selected global weight gives
D2p at 0.3240. D2p is substantially stronger than either text or popularity
alone, but it remains non-personalized. The bounded conclusion is therefore no
longer based only on zero-shot models: supervised text adaptation and a legal
item prior help, yet they do not explain the gain obtained from the target
user's history. This does not rule out every possible non-personalized ranker;
it shows that the tested lexical, semantic, fine-tuned dual-encoder, and
text-plus-popularity controls leave unresolved signal inside an already
query-filtered pool.

### 2.2 A strong static rule hides an exact-repeat effect

A recent-behavior score alone reaches 0.3139 NDCG@10, while the strongest
non-personalized control D2p reaches 0.3240. We combine these two registered
sources with one global weight selected on an internal train split: 0.3 on
standardized D2p and 0.7 on standardized recent behavior. The resulting D2s
static ranker reaches a three-seed mean of 0.3416. At the preselected seed, D2s
exceeds D2p by +0.0177 (95% CI [+0.0147, +0.0207]), the history score by +0.0276
(95% CI [+0.0231, +0.0321]), and the earlier B7-bge static waterline by +0.0110
(95% CI [+0.0072, +0.0148]). It also exceeds an interim D2h control that mixed
D2t and history but omitted popularity by +0.0064 (95% CI [+0.0037, +0.0090]).
D2s therefore became the complete static reference at that stage. Neither the
stronger non-personalized score nor the bundled history score subsumed the other
at the aggregate level, but this comparison did not reveal which part of the
history heuristic carried the gain.

Complementarity alone does not establish personalization: a history heuristic
might encode generic category, popularity, recency, or catalog-period effects.
The first wrong-user control used only earlier train donors while true dev
history rolled forward. Its positive +0.0354/+0.0276 results are numerically
reproducible but temporally confounded and are retained only as historical
evidence.

We therefore froze C5-R2 before evaluating a replacement. True history keeps
the standardized strictly-prior rolling snapshot. Wrong history comes from a
different user's train or earlier-dev request, is inserted only after the
complete same-timestamp group, and must satisfy a per-request factor-four bound
on latest-event age to enter the gate. Target query and candidates never
change; materialization and scoring do not read qrels.

The repaired wrong D2s variants average 0.3172 NDCG@10. On the 7,614 requests
freshness-balanced under all seeds, true-minus-wrong is +0.0374, +0.0379, and
+0.0362; all paired CIs are positive. The stricter same-query plus
freshness-balanced subset contains 1,063 requests. Its mean difference is only
+0.0095: seed 20260710 is significant, while the 20260708 and 20260709 CIs
cross zero. The locked rule required at least two significant seeds, so C5-R2
fails despite the positive direction.

We then froze C5-R3 before producing any component score. It decomposes the
executable B0b history heuristic exactly into (i) exact candidate-item matches
and (ii) deepest-exclusive category matches, while retaining the same D2p base,
global weight, within-request standardization, seeds, candidates, and 8,119
history-present requests. Across all 575,609 candidate rows, the two components
sum back to both the public scorer and the actual upstream B0b score file with a
maximum absolute error of `7.1e-15` and zero `1e-12` tolerance violations.

The item-only ranker reaches a three-seed overall mean NDCG@10 of 0.3454. On
history-present requests it exceeds D2p by +0.03204, +0.03214, and +0.03263;
all confidence intervals are positive. Category-only minus D2p is +0.00059,
+0.00053, and -0.00003, with every interval crossing zero. More importantly,
full D2s minus item-only is -0.00538, -0.00521, and -0.00634, with every
interval strictly below zero. Thus category alignment is not an independently
useful semantic signal in this frozen heuristic and actively dilutes the
stronger repeat-item component.

The complementary boundary is equally important. History is absent on 4,110
requests, or 33.6% of the evaluated population. On every one of these requests,
both component rankers and their seed-matched D2p base have identical rankings,
NDCG@10, MRR, Recall@10, and purchase NDCG where defined. The supported
conclusion is therefore narrow: **the tested static history gain is dominated
by exact candidate recurrence**, while semantic category transfer and stable
same-query identity specificity are not established.

### 2.3 Useful history is structurally sparse and difficult to learn

History availability alone understates the evidence problem. Among requests
with history, the median sequence contains only six events. The median Jaccard
overlap between the deepest product categories in history and in the candidate
set is 0.111, and it is exactly zero for 38.4% of history-present requests.
These are label-free structural facts, not oracle-derived slices. They make
equal relevance of all past events an unsupported assumption, but do not by
themselves establish query/candidate-conditioned event use.

The item-only C5-R3 control is consequently the demanding static waterline;
full D2s remains an important decomposition reference.
Representative full-dev learned methods tested under the same candidate
protocol do not exceed it. Official RecBole SASRec has a three-seed mean of
0.2972. The official-code
KuaiSearch DNN and DCNv2 rankers have means of 0.3063 and 0.3054 under the
declared proxy last-time split. Provisional official-code adapter traces for
the personalized product-search neighbors ZAM and TEM obtain means of 0.2986
and 0.2940. Separately, a history-aware 7B LLM reranker fails to improve the
earlier B7 on its frozen 2,000-request top-20 subset (difference -0.0019, CI
[-0.0089, +0.0050]); this is ancillary subset evidence, not a direct comparison
with full-dev D2s. These are results for representative methods, not a universal
claim about all personalized rankers. B5 remains proxy-aligned. ZAM/TEM are reported
only as supplementary benchmark context, not as load-bearing evidence: they are
not externally aligned, have only 11.71% unique dev-candidate coverage in the
train-target vocabulary, and their final human-review provenance remains
pending.

We also trained direct residual diagnostics against a supervised
non-personalized base. A recency-mean history residual obtains a three-seed mean
of 0.3145, and a query-attentive residual obtains 0.3148, compared with 0.3147
for their frozen base. Both residuals improve in two seeds and degrade in one;
their paired intervals cross zero, and query attention does not consistently
beat the unconditioned mean. Matched wrong-history rescoring reveals a small
identity effect at only one seed. These negative results prevent us from
claiming that query-conditioned event selection is already established.

The combined pattern is informative but different from the initial design
story. A simple exact-repeat feature is stronger than the complete history
heuristic, while sequence, CTR, semantic-category, and train-fitted residual
controls do not reproduce or improve it. A model can therefore appear to use
personalized history while exploiting a narrow recurrence shortcut. Any future
semantic-history claim must first demonstrate value beyond this item-memory
baseline rather than merely exceed a query-only ranker.

### 2.4 Why we do not motivate a channel router

An earlier analysis selected the best of query-only, history-only, and static
scores separately for each request and reported 0.4232 NDCG@10, or +28.0% over
B7. That statistic is not usable headroom evidence. Replacing the history
channel with a fixed Random ranking produces an even higher oracle of 0.4325
(+30.9%). Likewise, cheap request features predict the original oracle label at
0.6688 macro OvR AUC but predict Random-oracle labels at 0.6952. On 4,110
requests with no history, the original oracle still assigns 1,377 requests to
the nominal history channel.

This failure is caused by selecting and evaluating the per-request maximum on
the same clicks. Split-half stability and bootstrap intervals around that
maximum do not remove the bias. A learned logistic router subsequently scores
only 0.3072, below the earlier B7 by 0.0234, and the preregistered entropy proxy does not
stratify channel utility. We retain these results as negative methodological
evidence: the current data support aggregate correct-history use, but not a
completed identity-specific premise, an oracle-shaped routing architecture, or
a claim that cheap features can choose a winning fixed channel.

### 2.5 Bounded design insight: calibrate history evidence by fidelity

The completed motivation supports the following diagnostic observation:

> **Observation.** Inside the tested query-conditioned candidate pools, the
> reproducible history gain is concentrated in whether the current candidate
> exactly occurred in the user's strictly-prior history. Coarse category
> affinity does not improve the strong non-personalized base and weakens the
> repeat-item ranker when the two are combined.

This exposes a benchmark and modeling risk: aggregate improvements over
query-only controls can be mistaken for semantic personalization even when
exact recurrence is the load-bearing feature. It also yields a concrete design
insight: history should not enter the ranker as uniformly trustworthy context.
The system should calibrate candidate-history evidence by its empirical
fidelity—preserving reliable recurrence while allowing a transferable residual
only when the joint query, candidate, and history support it. The item-only
control, with a mean NDCG@10 of 0.3453755, remains the numeric waterline.

This consequence is a **design hypothesis**, not a post-hoc claim that C5-R3
passed. C5-R3 still falsifies multi-granular additivity and coarse-category
fallback. The next permitted stage is to formulate one end-to-end
candidate-conditioned evidence-fidelity calibration primitive and freeze its
cheap falsifier. Before full training, that falsifier must show no degradation
on repeat-present requests, stable positive value over D2p on the 4,677
history-present requests without exact-repeat candidates, failure of
coarse-only/wrong-user/shuffled-event/query-masked evidence to reproduce the
gain, and exact D2p fallback without history. Only then may the implementation
claim transferable personalization. Test remains unavailable for choosing or
rescuing the premise.

---

### Traceability of numbers

All figures are dev-set diagnostics. No method training, scoring, model
selection, or metric computation used the test split; C1 only created, hashed,
and structurally audited held-out files.

| Number | Source |
|---|---|
| BM25 0.305, popularity 0.301, BGE 0.306, cross-encoder 0.307 | `experiments/pps_results.md` |
| Shuffled-query 98.5%; candidates above random catalog on 98.8% | `reports/pps_c2_b1_diagnostics.json` |
| D2t mean 0.3141; D2p mean 0.3240; D2t vs B2z +0.0083 | `reports/pps_d2_d2h_summary.json` |
| D2s mean 0.3416; +0.0064 vs D2h and +0.0177 vs D2p | `reports/pps_d2s_summary.json` |
| Historical train-frozen wrong-history D2s +0.0354/+0.0276 | `reports/pps_d2s_summary.json` (superseded identity interpretation) |
| C5-R2 balanced true-minus-wrong +0.0362--0.0379; same-query mean +0.0095, only 1/3 significant | `reports/pps_c5r2_temporal_symmetric_identity.json` |
| C5-R3 item-only mean 0.3454; item-minus-D2p +0.0320--0.0326; category-minus-D2p nonsignificant in 3/3; full-minus-item negative in 3/3 | `reports/pps_c5r3_candidate_history_alignment.json` |
| History absent 33.6%; median non-empty length 6; deepest-category overlap zero 38.4% | `reports/pps_c3r_history_identity_control.json` |
| Raw `recently_*` rejected; history rebuilt from recall events with `event_time < request_time` | `reports/pps_c0_data_audit.json` |
| D1 mean-history/query-attentive residual means 0.3145/0.3148; neither stably beats D1q | `reports/pps_supervised_diagnostics_summary.json` |
| SASRec 0.2972; DNN/DCNv2 0.3063/0.3054; provisional ZAM/TEM 0.2986/0.2940 | `experiments/pps_results.md` |
| Random oracle 0.4325 exceeds M3 0.4232; Random-label M4 AUC 0.6952 exceeds 0.6688 | `reports/pps_m3_m4_random_canary_audit.json` |
| R1b 0.3072; -0.0234 vs B7 | `reports/pps_r1_router_summary.json` |

Claim boundaries: the paper says "tested query-only/non-personalized scorers,"
"representative learned methods," and "exact-repeat concentration in the tested
B0b/D2s history bundle." It does not claim that all personalization is
repetition, established semantic transfer, established same-query identity
specificity, a validated system architecture, universal query saturation,
a deployed causal effect, usable M3 oracle headroom, validated per-request
channel routing, or established superiority of query-attentive event selection.
The provisional ZAM/TEM traces are not required for any of these conclusions.
Because this repair is an internal gate on an already analyzed dev split, test
must remain untouched and cannot be used to rescue it. Any future confirmatory
evaluation requires a separately frozen, valid insight and system configuration.
