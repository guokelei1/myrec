# Intro-to-Motivation Repository Audit

> **Current supersession (2026-07-13).** This is a historical audit. C01--C80
> later closed without a validated architecture. Current work is R0 problem
> discovery under [`doc/31`](../doc/31_problem_discovery_and_architecture_iteration_protocol.md);
> no C81 or architecture training is authorized before a passed Failure Card.

> **Terminal supersession / 当前解释（2026-07-11）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
>、[`terminal closure`](../doc/dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已关闭，未得到经过验证的架构
> primitive，也未授权 proposed-system dev/full/test evaluation。C5-R3 FAIL
> 及全部数字不变。下文所有
> `current` / `final` / Go-No-Go 标签均为该审计时点的历史快照。

Date: 2026-07-10

Scope: review the KuaiSearch data/protocol, baseline evidence, motivation
experiments, paper draft, and proposed-system design boundary. No large model was
trained, no shared dev evaluator was invoked, and no test metric was computed.

Historical resolution addendum (C5-R3 gate-local authority at that time):
`doc/23` froze an exact
B0b item/category decomposition and a finite primary/fallback ladder before
component outcomes. Item-only beats D2p in all three seeds and reaches mean
NDCG@10 0.3453755; category-only is nonsignificant in all seeds; full D2s is
significantly worse than item-only in all seeds. Both architecture paths fail
as frozen, while all integrity checks pass. Motivation is complete as an
exact-repeat-concentration benchmark/analysis result; no system design is
authorized. See `reports/pps_c5r3_candidate_history_alignment.json`. All older
Go/waterline statements below are chronological history.

Resolution addendum (later on 2026-07-10): the blocker identified below was not
explained away. A different claim was locked in `doc/17`, then tested with six
shared-evaluator wrong-history controls. C3-R/C5-R passed and now supplies the
positive motivation for a **query-anchored personalized residual**. See
`reports/pps_c3r_history_identity_control.json` and
`reports/pps_c5_insight_audit.json`. The original M3/M4 conclusions remain
invalid. The Go/No-Go section below is the decision at audit time and is
superseded by this addendum.

Historical third resolution addendum (C5-R2 stage): the preceding C3-R/C5-R authorization is
itself superseded. Its wrong histories were train-frozen while true histories
rolled through dev. The frozen `doc/22` temporal repair passed integrity and
aggregate comparisons but failed the same-query rule (1/3 significant seeds;
2/3 required). At that stage there was no formal system authorization; see
`reports/pps_c5r2_temporal_symmetric_identity.json`.

Second resolution addendum (later on 2026-07-10): the authorized D1/D2/D2h
strengthening protocols were executed after this audit. D1m/D1a did not stably
beat their supervised base. A fully fine-tuned text tower plus train popularity
(D2p) reached 0.3240 mean, while a train-calibrated static mix with correct-user
history (D2h) reached 0.3352 and significantly exceeded B7 by +0.0046, CI
[+0.0012, +0.0080]. At that intermediate point D2h became the binding
baseline-to-beat. Its matched
wrong-history controls remain significant for every seed on history-present
and same-query subsets. See `reports/pps_d2_d2h_summary.json`; all claims below
about B7 being the current waterline are historical and superseded.

Third resolution addendum (later on 2026-07-10): the final fairness audit found
that D2h omitted the train-popularity component already validated by D2p. Doc
21 was locked before a train-only beta calibration and six fixed dev
evaluations. The complete D2s = D2p + B0b static control reaches 0.3416 mean and
significantly exceeds D2h by +0.0064, CI [+0.0037, +0.0090]. D2s was the binding
baseline at that intermediate point; see `reports/pps_d2s_summary.json`. The D2h addendum above is
retained as the chronological intermediate decision.

## Historical Executive Decision (at audit time)

**The baseline data and registered model metrics are correct and usable. The
current adaptive-personalization motivation is not yet usable.**

The surviving evidence supports a bounded paper setup:

1. KuaiSearch candidates are already query-conditioned by the recall stage.
2. The tested BM25, zero-shot bi-encoder, and zero-shot cross-encoder provide
   little additional click-ranking separation inside that pool.
3. B7-bge is a strong static, history-derived waterline at 0.3305 NDCG@10.
4. The representative learned baselines tested so far do not exceed B7-bge
   under their explicitly stated adaptation/cold-start boundaries.

The evidence does **not** currently support these claims:

1. M3 provides +28% usable or learnable personalized headroom.
2. M4 shows that useful evidence-channel selection is predictable.
3. The M3 selected slices establish under/over-personalized request prevalence.
4. C3 currently authorizes protocol-valid proposed-system implementation.

Architecture ideation may continue as hypothesis development. System training
should remain paused until the M3/M4 construct and the still-open C5 ordering are
adjudicated.

## Mechanical Verification

### Data and protocol

- C0 correctly rejected raw `recently_*` histories: future-only observed rate
  was 3.79%, above the 0.1% gate, and coverage was below the power floor.
- Standardized histories use only prior recall-window events with
  `event_time < request_time`; raw ranking histories are not used.
- The selected time window was 200,000/25,000/25,000 train/dev/test before
  filtering. The final evaluation scope is 12,229 dev requests after removing
  4,304 requests with fewer than five candidates and 8,467 with no clicked
  positive. All traffic/prevalence statements must be scoped to this filtered,
  click-observed population.
- All standardized history timestamps pass `history.ts < request.ts`.
- Candidate manifest SHA256 is
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- No method training, scoring, model selection, or metric computation used test
  records/qrels. C1 did create, hash, and structurally inspect held-out files, so
  the old global phrase "test split never read" was corrected.

### Metrics and provenance

- All 24 metric-bearing runs currently registered in
  `experiments/pps_results.md` were checked against their local artifacts.
- Every metric file records `generated_by = myrec.eval.evaluator`.
- All 24 runs use the same candidate-manifest hash and dev-qrels hash.
- NDCG@10, MRR, Recall@10, purchase NDCG, and coverage recomputed from each
  `per_request_metrics.jsonl` agree with `metrics.json` up to floating-point
  summation noise (maximum observed difference about `6e-15`).
- Every one of the 24 score-file SHA256 values matches `metrics.json`.
- The 91 dev-evaluator log entries reconcile with cards and budgets. The only
  duplicate run IDs are R1a/R1b, each with the documented initial invocation and
  one allowed low-recovery recheck.
- B7-bge vs B2z was missing a significance artifact. The read-only frozen
  comparison is now recorded at `reports/compare_b7_bge_vs_b2z.json`:
  delta `+0.024881`, 95% CI `[+0.021590, +0.028211]`.

The observed baseline numbers themselves are therefore not the problem.

One Tier-2 boundary remains: B7's gain is evidence that the history-derived
score is predictive, not yet a causal proof that the correct user's history is
responsible. The frozen wrong-query/wrong-history/no-history controls have not
been run and remain necessary before a causal personalization claim.

## Construct-Validity Blocker

The new reproducible audit is
`reports/pps_m3_m4_random_canary_audit.json`, generated by
`scripts/audit_m3_m4_random_canary.py`. It reads only existing per-request
metrics and the frozen M4 feature frame.

| Diagnostic | Actual M3 channels | Replace history with Random |
|---|---:|---:|
| Oracle NDCG@10 | 0.4232 | 0.4325 |
| Relative headroom over B7-bge | +28.0% | +30.9% |
| M4-style 5-fold macro OvR AUC | 0.6688 | 0.6952 |

Additional decisive check:

- On 4,110 requests with empty history, the original M3 oracle still reports
  +27.7% relative headroom and assigns 1,377 requests to `history_b0b`.
- With no history, B0b is a non-informative all-tie ranking resolved by the
  fixed candidate tie-break. Per-request label maximization selects it whenever
  that arbitrary ordering happens to beat query scoring.
- Restricting M4 to history-present requests lowers actual macro AUC to 0.6281,
  while the Random-oracle version remains 0.6528.

This establishes that the current `per-request argmax -> evaluate on the same
click labels` protocol measures selection over metric noise. Split-half
stability and paired bootstrap around that selected maximum do not remove the
bias. The label-shuffle M4 canary detects ordinary leakage, but not this
construct failure.

Consequences:

- M3's +28.0%, M4's 0.6688, and the +65.4% many-channel oracle remain accurate
  frozen computations, but they are failed diagnostics rather than positive
  motivation evidence.
- The 35.1% history-assigned slice is contaminated by 1,377 no-history requests.
  It cannot support the current direct E1 personalization claim.
- R1b's negative recovery is consistent with the absence of validated
  exploitable headroom; it cannot be explained away as merely a weak router.
- `reports/pps_c3_motivation.json` retains its historical threshold/adjudication
  status but is now marked paused for current claim use.

## Baseline Usability

| Evidence | Usability | Required wording |
|---|---|---|
| B0a/B0b/B1/B2z/B3/B7 | usable | Dev results on the filtered click-observed KuaiSearch split |
| Candidate-vs-random and shuffled-query diagnostics | usable | Candidate pool is query-conditioned; BM25/query field works |
| Query-only conclusion | usable with scope | The tested lexical/zero-shot scorers add little; do not claim universal query saturation |
| B4o SASRec | usable with caveat | 3-seed mean 0.2972; 77.8% candidate rows are cold to the train-item vocabulary |
| B5o DNN/DCNv2 | usable with caveat | Means 0.3063/0.3054; official-code, proxy-aligned last-time 10% split |
| B6o | not a KuaiSearch result | Alignment-not-verifiable; excluded from the formal KuaiSearch table |
| B8 | usable as subset control | History-aware B7-seeded top-20 reranker, not query-only or an independent full-dev model |
| B9 ZAM/TEM | numerically usable, protocol provisional | Means 0.2986/0.2940; official-code adapters, not externally aligned, 11.71% unique candidate coverage; author top-5 confirmation pending |
| M3/M4/M5 selected slices | not positive evidence | Preserve as failed/noise-contaminated diagnostics |

For trainable methods, paper prose now uses multi-seed means and variability.
Highest observed seed runs remain only for traceability and conservative
per-request comparisons, consistent with doc/07 Section 11.

## Logic Review

The introduction's first half remains coherent through this chain:

```text
fixed-candidate personalized product ranking
  -> recall candidates are already query-conditioned
  -> tested off-the-shelf query scorers cluster
  -> a static history-derived mixture is a strong waterline
  -> representative adapted baselines do not clear it
```

The next arrow is currently missing:

```text
static waterline
  -X-> validated request-level adaptive headroom
  -X-> query-conditioned evidence selection as an evidence-backed primitive
```

The proposed primitive is still a reasonable research hypothesis, but the
current experiments do not derive it. The paper must not present architecture
choice as the measured consequence of M3/M4 until this gap is repaired.

## Decisions Required

### 1. C3 repair or claim withdrawal

Recommended route: write an explicit post-hoc C3 amendment that preserves the
failed Random null and replaces same-label oracle selection with independent
selection/evaluation. Candidate low-cost protocols include:

- train/cross-fit a selector on train requests and evaluate only once on dev
  (the existing R1b is a negative instance, not a pass);
- on multi-click requests, select a channel using one disjoint positive-label
  partition and evaluate on another, with a frozen coverage/power rule;
- use an explicit Random-channel null and require actual-channel improvement to
  exceed it by a pre-registered margin before calling the difference headroom.

Do not simply subtract the observed Random oracle after seeing the result and
declare a corrected pass. The revised statistic and threshold must be written
before recomputation.

### 2. C5 sequence

Doc/11 still requires C5 before modeling. Insight-2/Consensus Law is already
falsified (`rho = -0.0110` for entropy versus oracle-query gain), and Insight-1
has not been tested. Choose one explicitly:

- keep the original route and run the remaining cheap Insight-1 falsification;
- formally retire/redefine C5 after preserving the original Insight-2 failure;
- continue system work only as exploratory engineering, not protocol-valid
  paper development.

### 3. B9 human review

The 20-request top-5 sheet has preliminary decisions but no reviewer or
authorization field. Numerical runs are valid. To close the internal-validity
suite, an author must review the sheet and record name/role, date, and decision.

### 4. Git promotion

The complete M4/R1/B9/paper/readiness reproducibility core is still untracked in
the working tree. This does not change local numerical correctness, but another
checkout cannot reproduce the current state. After reviewing this audit, the
intended source/config/report files should be committed as one or more explicit
protocol/evidence commits. Raw `data/`, `artifacts/`, `runs/`, and `models/`
remain correctly ignored.

## Corrections Applied

- Added the reproducible M3/M4 Random canary and linked it from M3, M4, C3,
  results, paper, design, and readiness records.
- Paused the old architecture-readiness claim without deleting historical
  threshold/adjudication evidence.
- Corrected the M3 query-assigned slice from a 60.6% implied failure rate to the
  strict 18.6% within-slice / 12.53% all-request rate.
- Narrowed "query saturation" to the tested lexical/zero-shot scorer scope.
- Removed B8 from the query-only evidence chain.
- Switched paper-facing trainable-baseline wording to multi-seed means.
- Corrected global test-isolation wording and R1 directional significance text.
- Fixed stale B9 run metadata from `scored_not_evaluated` to `evaluated`.
- Updated repository README indexes and marked historical summaries as
  superseded where needed.

## Historical Go/No-Go (at audit time)

**Go** for baseline/data reporting and architecture brainstorming.

**No-Go** for claiming C3 motivation success, citing M3/M4 as adaptive evidence,
or starting protocol-valid proposed-system training until Decisions 1 and 2 are
resolved.
