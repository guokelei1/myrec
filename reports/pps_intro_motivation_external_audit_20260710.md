# External Audit: Introduction and Motivation Stage

> **Current supersession / 当前解释（2026-07-10）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
> 为准：motivation complete，design formulation ready；implementation/training
> 仍由新的、design-specific pre-outcome falsifier 把关。C5-R3 FAIL 及全部数字不变。

> **Final C5-R3 supersession.** After this historical audit and the later C5-R2
> repair, `doc/23` froze and executed the finite item/category component gate.
> Item-only is significant over D2p in 3/3 seeds and reaches mean NDCG@10
> 0.3453755; category-only is significant in 0/3; full D2s is significantly
> worse than item-only in 3/3. The primary and sole fallback both fail, with
> integrity passed. Current authority is
> `reports/pps_c5r3_candidate_history_alignment.{json,md}`. The scientific
> terminal is benchmark/analysis-only and there is no design authorization.
> Everything below remains the historical external review at its audit time.

> **Superseded by executed C5-R2 repair.** This report's CONDITIONAL GO treated
> the temporal-staleness confound as a wording caveat. The subsequently frozen
> and executed `doc/22` control failed its same-query significance rule (1/3
> significant seeds; 2/3 required). Current authority is
> `reports/pps_c5r2_temporal_symmetric_identity.{json,md}` and formal
> proposed-system authorization is paused. The audit below is preserved as a
> historical independent review.

Date: 2026-07-10
Auditor: independent adversarial audit session (read-only; no commit, no training,
no test evaluation). Audit mandate: `doc/review_prompts/20260710_intro_motivation_external_audit_prompt.md`.
Repository audited at `/home/gkl/myrec` (the mandate's `/data/gkl/myrec` resolves to
the same tree).

Method: seven parallel workstreams (data/leakage; evaluator and result provenance
with independent recomputation; train/dev isolation and ordering; baseline
fairness; statistical and construct validity; argument audit; reproducibility and
hygiene). Every load-bearing number was recomputed from
`runs/*/per_request_metrics.jsonl`, score files, donor assignments, and id lists —
not from summaries. `qrels_test.jsonl` content was never read and no test metric
was computed.

---

## 1. Verdict

**CONDITIONAL GO** for proposed-system design.

**There are no Critical or High findings.** Every headline number reproduced
exactly (to full stated precision or within ~1e-12 float-summation noise), all
provenance hashes match, the 127-entry dev-eval log reconciles, train-only
calibration isolation is verified at code and hash-chain level, and the paper's
claim boundaries are, with the exceptions below, correctly bounded. The
motivation chain independently supports authorizing design of a query-anchored
personalized residual.

The GO is conditional on the following repairs (none blocks starting design work;
all must land before the evidence base is presented externally):

- **C-1** Commit the entire untracked D1/D2/D2h/D2s track (protocols, configs,
  scripts, calibration artifacts, lock manifests, reports) to git. Until then
  every ordering claim rests on local mtimes plus internal hash chains with no
  tamper-evident external record (Finding M1).
- **C-2** Quantify and register the deviation from the registered
  "session does not cross splits" rule, or re-split (Finding M2).
- **C-3** Scope the word "complete" for the D2s waterline and record a rationale
  (or a train-only robustness check) for the omitted flat three-way weighting and
  BM25 lexical channel (Findings M3, M7).
- **C-4** Add a one-sentence recency/staleness caveat to the identity-control
  interpretation (Finding M4).
- **C-5** Commit the script that generated
  `reports/pps_d2s_calibration_semantics_verification.json` (Finding M5).

---

## 2. Findings

### Critical

None.

### High

None.

### Medium

**M1. Nothing in the D-track is under version control.**
All D1/D2/D2h/D2s protocols (`doc/18`–`doc/21`), configs
(`configs/analysis/*`), scripts, calibration reports, lock manifests, and summary
reports are untracked; the last commit is B5o Stage B (`git status` snapshot;
commit `6fa3024`). Affected claim: all "frozen before results" ordering claims.
The internal evidence (final configs embed calibration-artifact sha256s;
dev-score metadata embeds final-config sha256s; all hashes recomputed and
matched) is coherent and passes, but is not tamper-evident.
Required action: commit immediately (condition C-1).

**M2. The registered "same session does not cross splits" rule is not implemented.**
`doc/11_experiment_and_dataset_plan.md:147` registers the rule; the actual split
is purely positional over time-sorted requests
(`src/myrec/data/kuaisearch_audit.py:350-368`, 80/10/10 by `(time_index,
request_id)`), with no session logic anywhere in
`kuaisearch_standardize.py` and no dev-log amendment. A session straddling the
train/dev boundary can put near-duplicate same-user requests on both sides,
mildly inflating train-fitted components (popularity, D2p). This is not label
leakage — each request's labels stay in its own split — and all compared methods
share the bias, so no comparison is invalidated; but it contradicts a registered
rule. Required action: quantify boundary-straddling sessions, then re-split or
record an explicit amendment (condition C-2).

**M3. "Complete" static waterline wording overstates the frozen construction.**
`doc/21_d2s_static_full_waterline_protocol.md:63`,
`paper/introduction_and_motivation.md:110,161`,
`reports/pps_architecture_readiness.md:18`,
`reports/pps_intro_motivation_completion_20260710.md:36`. At freeze time three
already-available static enrichments were never tested inside D2s: (a) a flat
three-way z(D2t)/z(pop)/z(B0b) mixture with independently train-selected weights
(the hierarchical construction pins text:pop at the 0.6:0.4 ratio calibrated in
doc/19 *without* the history term present); (b) the BM25 lexical channel (B1,
incl. exact-match boost), mandated for a complementarity artifact in doc/13 §2.6
but absent from every D-series mixture; (c) the full shared document template
(see M7). All omissions can only *understate* the waterline, i.e. loosen the
proposed-system gate (≈0.3485) in the system's favor — exactly the direction an
adversarial reviewer will probe. Required action: scope "complete" as "complete
with respect to the registered D-series channels (fine-tuned title text, train
popularity, causal recent behavior)"; register a rationale or run a train-only
calibrated robustness check before test (condition C-3).

**M4. Temporal-staleness confound in the wrong-history donors is undisclosed.**
Donor eligibility requires the donor request to precede the earliest dev request
(`src/myrec/analysis/history_identity.py:171-174`), so every donor history ends
before the dev window, while true histories extend to just before each request.
B0b weights exact item matches ×3.0 (`src/myrec/baselines/core.py:343-344`), so
item churn between the donor window and dev candidate pools systematically
depresses wrong-history scores for a temporal, not identity, reason. The
direction of the +0.0354 effect is not threatened, and this donor design is the
conservative leakage-free choice, but part of the delta may be recency-specific.
Required action: bound the interpretation as "identity- and recency-specific
predictive value" wherever the identity control is described (condition C-4).

**M5. The D2s calibration/scorer semantics-verification artifact is unreproducible.**
`reports/pps_d2s_calibration_semantics_verification.json` (max grid metric diff
2.49e-07; same beta selected) has no generating script anywhere in the repo.
The underlying code difference is real: calibration z-score uses
`(v-mean)/sqrt(var+1e-6)` (`src/myrec/analysis/finetuned_query_tower.py:837-839`)
while the final scorer uses `(v-mean)/std` with zero-var→0
(`src/myrec/baselines/core.py:569-577`). Required action: commit the
verification script or exact snippet (condition C-5). Related: D2h has the same
epsilon-level gap and no verification artifact at all, and its alpha=0.1 margin
over alpha=0.2 is only ~5e-6 (`reports/pps_d2h_train_only_calibration.json:5-7`)
— extend the check to D2h (Low, see L6).

**M6. The claimed no-history "score-level" equality is false as literally stated;
only rank- and metric-level equality hold.**
For no-history requests, D2s scores equal `0.3 × z(D2p)` — a positive affine
transform of within-request z-scored D2p, not the raw D2p scores (verified
directly: request `ks_07067849d5a21b181bfcfdc9`, item 624667 scores 0.138990
under D2p vs 0.052030 under D2s, seed 20260708; `max_affine_error` exactly 0.0
in `reports/pps_d2s_score_audit.json:5-27`). Rankings and all metrics are
identical (0 mismatches on 4,110 requests × 3 seeds,
`reports/pps_d2s_summary.json:152-180`). The paper itself does NOT overclaim —
`paper/introduction_and_motivation.md:141-144` claims only identical
NDCG@10/MRR/Recall@10 — but the audit mandate's claim bullet and any external
restatement must say "rank- and metric-identical", not "score-identical".

**M7. D2s's text channel sees less candidate text than the text baselines it supersedes.**
D2t uses query + candidate **title** only
(`doc/19_finetuned_nonpersonalized_control_protocol.md:28,37`,
`experiments/pps_baseline_cards.md:112`), whereas the shared-template rule
requires B1/B2z/B3/B8 to see title+brand+seller+category
(`doc/13_baseline_implementation_plan.md:196-198`). Not a red-line violation
(the rule is scoped to those baselines and the card discloses it), but it bounds
the sufficiency defense of D2s and again biases the gate downward. Required
action: state the text-scope limitation wherever D2s is called the strongest
static baseline (folds into condition C-3).

**M8. Two ~16 MB binary .jar files are git-tracked.**
`baselines/pps_classic/hem_official/jar/AmazonMetaData_matching.jar` and
`AmazonReviewData_preprocess.jar` (vendored upstream HEM tools, deliberate per
`.gitattributes:3`), and `*.jar` is absent from the binary patterns in
`.gitignore:118-131`. Not checkpoints/datasets/score dumps — repo-size hygiene
only. Required action: add a `*.jar` ignore-with-exception rule or move to
release storage.

### Low

- **L1.** Split-boundary `time_index` ties can straddle split boundaries
  (`kuaisearch_audit.py:350`); full-train popularity may therefore include click
  events with timestamp equal to (not strictly before) the earliest dev
  requests. Expected negligible (486,833 distinct time values / 555,553
  requests); report the boundary-tie count.
- **L2.** The C1 structural audit reads `qrels_test` content in aggregate
  (`src/myrec/data/protocol_audit.py:284-301,324,421-454`). Registered
  (doc/11 C1 gate) and disclosed in the paper (:251-253); no selection decision
  consumed it. Keep frozen; no action.
- **L3.** A retired raw-`recently_*` consumer remains in
  `src/myrec/baselines/kuaisearch_materializer.py:203-204` (feeds only retired
  B5/Stage-A). Current B5o Stage B rebuilds history from the frozen standardized
  records. Mark the raw materializer deprecated.
- **L4.** Dev-request histories legitimately contain the user's earlier
  dev-window interactions (`kuaisearch_standardize.py:153-159`) — causally valid
  and label-free, but should be stated once in the paper's protocol paragraph.
- **L5.** Lock-step programmatic mtimes (doc/21 and its base config share an
  identical nanosecond mtime) weaken mtimes as independent ordering evidence;
  the hash chain is the load-bearing evidence and holds.
- **L6.** D2h lacks a calibration/scorer semantics-verification artifact (see M5).
- **L7.** `scripts/recalibrate_d2p_alpha.py:63-74` mutated
  `runs/20260710_kuaisearch_d2t_calibrate_s20260708/train_summary.json` in place
  (invalid grid preserved verbatim inside two artifacts); acceptable but reduces
  forensic auditability.
- **L8.** The D2s-vs-B7 comparison in the paper headline
  (`paper/introduction_and_motivation.md:113-114`) was not in doc/21 §4's
  preregistered comparison list (D2h, D2p, B0b). No extra evaluator invocation
  occurred and the number reconciles (+0.010986, CI [0.0072, 0.0148]); log the
  deviation.
- **L9.** Stale "baseline-to-beat is B7-bge" statements remain in
  `reports/b5o_official_alignment.md:244-245` and
  `reports/pps_batch2_decision_summary.md:73`; add superseded-by-D2s notes.
- **L10.** doc/13 §2.4 fairness matrix has no rows for D1/D2/D2h/D2s, R1, B9
  (`doc/13_baseline_implementation_plan.md:176-190`); extend or cross-reference.
- **L11.** B5o's locked official loader caps history at 20 clicked items and
  ignores purchases (`reports/b5o_official_alignment.md:175-177`); disclosed —
  keep the caveat wherever B5o numbers appear.
- **L12.** "Same normalized query under all seeds" is trivially seed-invariant
  (tier selection is seed-independent; the seed only picks the donor within the
  tier; tier counts identical across seeds: 1,505 + 1,204 = 2,709). Wording
  suggests a stricter test than it is; no numerical impact.
- **L13.** The `"seed": 20260708` field in every comparison JSON is the
  *bootstrap* seed, copied verbatim into per-run-seed blocks of
  `reports/pps_d2s_summary.json:91,104,133,147` — confusing labeling; rename to
  `bootstrap_seed`.
- **L14.** Donor reuse across targets (bounded pools, max 32/key) induces mild
  cross-request dependence not modeled by the request-level paired bootstrap;
  effect on CIs judged small.
- **L15.** `tests/test_history_identity.py` does not cover the
  strictly-before-dev donor timestamp filter (enforced in production at
  `history_identity.py:171-174,233,243-244`).
- **L16.** The M3/M4 Random canary uses a single frozen random run — sufficient
  for an invalidation argument (one draw already exceeds the actual oracle).
- **L17.** `reports/README.md:13-14` references nonexistent
  `pps_c4_data_final.json` / `pps_c5_insight.json` (actual:
  `pps_c5_insight_audit.json`); `configs/baselines/b4o_sasrec_recbole.yaml:101-104`
  expected-output names drift from actual compare-report filenames.

---

## 3. Claim Matrix

Every bullet from "Claimed Current State to Challenge":

| # | Claim | Verdict | Basis |
|---|---|---|---|
| 1 | Fixed candidate pool is already query-conditioned | **verified** | Recomputed from `reports/pps_c2_b1_diagnostics.json`: shuffled-query 0.98520 (98.5%), pool-vs-random-catalog 0.98806 (98.8%), n=12,229 |
| 2 | Raw `recently_*` failed the registered future-leakage check; standardized histories rebuilt strictly prior | **verified** | Code-level: failure thresholds and measured values (future_only 3.79% > 0.1%; past_supported 9.27% < 20%; `kuaisearch_leakage.py:348-355`, `pps_c0_history_leakage_check.json:128-135`); strict `<` with tie-safe grouping (`kuaisearch_standardize.py:141-160`); no current scoring path consumes raw fields (repo-wide grep; only retired-B5 materializer, L3) |
| 3 | D2p strongest registered non-personalized control, mean ≈ 0.323950 | **verified** | Recomputed 0.3239500729331402, SD 0.00018811916924114; exceeds every registered non-personalized method (B0a 0.3013, B1 0.3054, B2z 0.3056, B3 0.3068, D2t 0.3141) |
| 4 | D2h valid intermediate static control ≈ 0.335213, omitted popularity | **verified** | Recomputed 0.33521346252574075; popularity omission confirmed in doc/20 construction and repaired by doc/21 |
| 5 | D2s freezes D2p, train-only beta=0.3, mean 0.3416289845, SD 0.0003711265 | **verified** | Recomputed 0.34162898451860496 / 0.00037112647715626 from per-request files; beta selected on internal train only (`calibrate_d2s_static_full.py:47-57`), grid max at 0.3, tie-break preregistered |
| 6 | Seed 20260708: D2s−D2h = +0.0063627404, CI [+0.0037327413, +0.0090111998] | **verified** (point delta exact; CI bounded) | Point delta recomputed +0.006362740409498123 on identical 12,229-request id sets. Bootstrap CI not bit-exactly re-executed (Python denied); normal-approx CI [+0.0037424, +0.0089831] agrees to ~3e-5 and confirms significance |
| 7 | 8,119 history-present: true−wrong mean +0.0353653; 2,709 same-query: +0.0276284; every seed CI positive | **verified** | Per-seed deltas recomputed exactly (+0.035570/+0.035101/+0.035424; +0.027716/+0.028082/+0.027088); all normal-approx CIs strictly positive; subset counts and id-list sha256s independently confirmed. Interpretation caveat M4 applies |
| 8 | 4,110 no-history: D2s and seed-matched D2p exactly equal NDCG@10/MRR/Recall@10 | **verified at metric and rank level** | 0 mismatches × 3 seeds recomputed; `max_affine_error` exactly 0.0. Raw scores differ by a fixed positive affine transform (M6) — do not restate as score-level equality |
| 9 | D1m/D1a do not stably improve D1q; query attention is a hypothesis | **verified** | Recomputed per-seed: D1a−D1q +0.00050/−0.00046/+0.00027; D1m−D1q +0.00051/−0.00136/+0.00026; all CIs cross zero; D1a not stably above D1m; wrong-history identity effect significant at one seed only |
| 10 | M3/M4 construct-invalid via Random canary; retained only as negative result | **verified** | Random-channel oracle 0.4325 > actual 0.4232; Random-label AUC 0.6952 > 0.6688; 1,377/4,110 no-history requests assigned to history channel; same features/folds for both label sets (`audit_m3_m4_random_canary.py:95,187-235`); no surviving positive claim depends on the oracle |
| 11 | B9 ZAM/TEM supplementary and non-load-bearing; provenance incomplete | **verified** | Provenance machine-recorded as pending (`pps_b9_top5_review_decision.json`: reviewer null, authorization null); "not load-bearing" stated in paper (:172-176,275), cards, readiness report; no positive motivation claim depends on ZAM/TEM |
| 12 | Binding dev comparator is D2s; full claim requires significance + ≥2% relative (≈0.3485) | **verified, bounded** | Gate frozen in doc/15 §5 and doc/21; threshold arithmetic checks (0.3416×1.02≈0.3485). Bounded by M3/M7: the waterline's "completeness" is scoped to the registered channels, and identified omissions bias the gate downward |
| 13 | Test metrics never used; conclusion is dev-stage design authorization only | **verified (statically)** | No `"split": "test"` in any run; no analysis code reads `qrels_test` content except the registered C1 structural audit (L2); qrels_test sha frozen in `pps_c1_protocol.json`; verified by code/artifact inspection, not fresh execution |

---

## 4. Independent Number Table

All recomputed from `runs/*/per_request_metrics.jsonl` (n = 12,229 per full-dev
run), donor files, and id lists. SDs are sample SDs (ddof=1).

| Quantity | Independent recomputation | Registered claim | Match |
|---|---|---|---|
| D2p three-seed mean / SD | 0.3239500729331402 / 0.0001881191692411 | 0.323950 / 0.000188 | exact |
| D2t three-seed mean | 0.31410103184849114 | 0.3141 | exact |
| D2h true mean / SD | 0.33521346252574075 / 0.0005067871513972 | 0.335213 / 0.000507 | exact |
| D2h wrong mean | 0.3089517469768552 | 0.30895 | exact |
| **D2s true mean / SD** | **0.34162898451860496 / 0.00037112647715626** | 0.3416289845 / 0.0003711265 | exact |
| **D2s wrong mean / SD** | **0.318149487484221 / 0.00051300834090326** | 0.318149487484221 | exact |
| D1q / D1m / D1a means | 0.31468 / 0.31449 / 0.31478 | 0.3147 / 0.3145 / 0.3148 | exact (4 dp) |
| D2s−D2h, seed 20260708, full dev | +0.006362740409498123 | +0.006362740409498102 | exact (1e-17) |
| D2s−D2p, seed 20260708 | +0.017697823418886770 | +0.0177, CI [+0.0147, +0.0207] | exact |
| D2s−B0b, seed 20260708 | +0.027596938701226214 | +0.0276, CI [+0.0231, +0.0321] | exact |
| D2s−B7-bge, seed 20260708 | +0.010986215489108388 | +0.0110, CI [+0.0072, +0.0148] | exact |
| True−wrong D2s, history-present (8,119), per seed | +0.035570 / +0.035101 / +0.035424; mean +0.035365287 | mean +0.0353653; conservative CI-low +0.0302 | exact |
| True−wrong D2s, same-query (2,709), per seed | +0.027716 / +0.028082 / +0.027088; mean +0.027628419 | mean +0.0276284; conservative CI-low +0.0193 | exact |
| True−wrong B0b, history-present, mean | +0.04966440612837642 | +0.0497 | exact |
| No-history equality (4,110 × 3 seeds) | 0 metric mismatches; max_affine_error 0.0 | exact equality | confirmed |
| Request counts | 8,119 + 4,110 = 12,229; same-query 2,709 = 1,505 + 1,204 | same | confirmed |
| M3 oracle / Random canary | 0.4231528 / 0.4325223 (+28.0% / +30.9%) | 0.4232 / 0.4325 | exact |
| M4 AUC actual / Random-label | 0.6688017 / 0.6951983 | 0.6688 / 0.6952 | exact |
| R1b | 0.3071765 (−0.02335 vs B7) | 0.3072 / −0.0234 | exact |
| ZAM / TEM means | 0.2986198 / 0.2940167 | 0.2986 / 0.2940 | exact |
| Reference runs | B0b 0.3139, B2z 0.3056, B7-bge 0.3305, B4o 0.29758, B5o 0.30878, B8a-h50 0.33022 | registry rows | exact (4 dp) |
| Candidate manifest sha256 | `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e` | same, asserted in every relevant `metrics.json` | match |
| qrels_dev sha256 | `518eab43850c6fbc841cfa5f047602a1e41761960bc80d244c93fb379b0029bc` | same | match |

Paper §2.1 controls also verified against the registry: BM25 0.3054→0.305,
popularity 0.3013→0.301, B1−B0a CI [−0.0012, +0.0098]→[−0.001, +0.010], B2z
0.3056→0.306, B3 0.3068→0.307 (not significant over B2z), D2t vs B2z +0.0083
[+0.0044, +0.0123]. §2.3: SASRec 0.2972±0.0004, DNN 0.3063, DCNv2 0.3054, B8a
h=50 subset delta −0.0019 [−0.0089, +0.0050] (the paper cites the B8 variant
most favorable to B8 — conservative for its own argument).

---

## 5. Protocol-Integrity Table

| Dimension | Status | Evidence |
|---|---|---|
| History leakage | **PASS with caveats** | Strict `event_time < request_time`, tie-safe, same-user only, max len 50 (`kuaisearch_standardize.py:38,141-160`); mechanically re-verified by C1 (`protocol_audit.py:457-470`). Caveats: M2 (session rule unimplemented), L1 (boundary ties), L4 (earlier dev-window events in dev histories — causally valid) |
| Popularity / calibration label isolation | **PASS** | Popularity from packed train click labels only; calibration exclusively on internal-train (first 90%) counts + final-10% internal validation (`supervised_diagnostics.py:148-161,348-365`, `finetuned_query_tower.py:419-427`, `calibrate_d2s_static_full.py:47-57`); dev packed labels all-zero (manifest `clicked_rows: 0`) |
| Invalid first D2p alpha | **PASS — proven non-contaminating at artifact level** | Invalid grid (full-train popularity, 0.6067) invalidated 17:59:13, before any D2 dev score (first D2 dev eval 18:07:10, `dev_eval_log.jsonl:110`); corrected alpha 0.6 derivable only from internal-train counts; dev metadata hash-chains to corrected final config; invalid evidence preserved verbatim |
| Ordering (protocol → calibration → final config → scores → eval) | **PASS against local evidence; not tamper-evident** | All lock-manifest hashes recomputed and match for D2/D2h/D2s; mtime sequence consistent (doc/21 19:05:36 → calib 19:07:04 → final config 19:07:17 → first score 19:07:26.707 → first eval 19:08:46); M1 (untracked) and L5 (programmatic mtimes) bound the strength |
| Wrong-history donor isolation | **PASS** | Donors train-only, different user, donor request < earliest dev request, donor events < donor request, runtime future-event assertion (`history_identity.py:171-174,184-185,233,243-244,455-476`); interpretation caveat M4 |
| Dev-eval accounting | **PASS** | 127 entries, 125 unique run_ids, duplicates exactly the two documented R1 ids; D2s = 6 (3 true + 3 wrong); group counts d1:12, d2:18, c3r:6, b9:6, r1:4 → 46 recent, matching the reconciliation; every logged run has `metrics.json` with matching NDCG; budgets within doc/13 §2.5 |
| Evaluator uniformity | **PASS** | Shared evaluator only; every relevant `metrics.json` asserts the same candidate-manifest and qrels_dev hashes; calibration and evaluation share the same salted deterministic tie-break (`src/myrec/eval/metrics.py:10-35`) |
| Test isolation | **PASS (static verification)** | No run or script uses split="test"; only registered C1 structural audit touches qrels_test (aggregate, disclosed); qrels_test sha frozen; content never read by this audit |
| Repository hygiene | **PASS with M8/L17** | JSON 160/160 valid; YAML 26/26 manually reviewed clean; zero trailing whitespace in text sources; no secrets; no tracked checkpoints/datasets/score dumps (only the two vendored jars, M8); all paper-facing paths exist; 13 reports carry explicit superseded/invalidated markers |

---

## 6. Logic Assessment

Paragraph-level classification of `paper/introduction_and_motivation.md`: every
empirical sentence checked is **supported** within its stated bounds; none was
found overstated or unsupported. Specifically:

- §1 (protocol paragraph): the temporal-construction guarantee is correctly
  distinguished from a deployed causal claim ("guarantees temporal direction
  within the observed recall window"; ":147-148 not a randomized causal
  experiment"; claim-boundary paragraph excludes deployed causal effects).
- §2.1: conclusions are correctly narrowed to "tested lexical, semantic,
  fine-tuned dual-encoder, and text-plus-popularity controls"; no universal
  query-saturation claim.
- §2.2: all numbers reproduce; the identity conclusion is stated as
  "identity-specific predictive signal" under a matched permutation. After C-4
  it should read "identity- and recency-specific" (M4).
- §2.3: representative-methods framing is respected; B9 is explicitly
  non-load-bearing; B8 is explicitly "ancillary subset evidence, not a direct
  comparison with full-dev D2s"; D1 negatives prevent claiming query-attentive
  selection as established.
- §2.4: M3/M4 are used strictly negatively; the Random-canary argument is
  independently sound (same features, same folds, same tie order; a single
  Random draw exceeding the actual oracle suffices for invalidation).
- §2.5: the architecture consequence is presented as a justified, falsifiable
  hypothesis with preregistered falsifiers, not as an established result.

The full chain — query-conditioned pool → strong non-personalized control →
complete static correct-history gain → matched identity dependence → no-history
boundary → simple learned residual failure → query-anchored
personalized-residual hypothesis — holds link by link on independently
recomputed numbers. The final consequence follows as a *design hypothesis*, and
the paper says so.

Statements that must remain explicitly bounded:

1. "Complete" static waterline → scope to registered D-series channels (M3/M7).
2. Identity control → identity- **and recency-**specific predictive value (M4).
3. No-history boundary → rank- and metric-level equality, not raw-score equality (M6).
4. D1 negatives → bounded to the tested residual family (frozen-embedding
   mean/attention residuals), not to personalization learnability in general.
5. Non-personalized ceiling → "tested controls leave unresolved signal", not
   "no non-personalized ranker can close the gap".
6. B9/ZAM/TEM → supplementary context until human-review provenance completes.

Alternative design implications equally supported by the same evidence (the
evidence constrains but does not uniquely determine the primitive): a learned
*global* gate over the same two standardized signals conditioned on history
presence/length, or a retrieval-then-model history-selection architecture
(SIM/UBR-family), would satisfy the same three anchors. What the evidence does
rule out is a per-request router over fixed channels (M3/M4/R1b negatives). The
doc/15 requirement that the proposed system beat a parameter-matched
unconditioned history encoder is therefore load-bearing for novelty and must be
retained.

---

## 7. Residual Risks

**Blockers (before external presentation of the evidence, not before design):**

- B-1 = C-1 (M1): untracked D-track; ordering claims not tamper-evident.
- B-2 = C-2 (M2): registered session-split rule unimplemented and unamended.
- B-3 = C-3 (M3/M7): waterline "completeness" wording plus omitted static
  combinations; risks the proposed-system gate being set too low.

**Nonblocking paper-completion work:**

- C-4/C-5 and all Low findings (L1–L17): caveat sentences, labeling repairs,
  superseded-note additions, `bootstrap_seed` rename, deprecation markers,
  doc/13 matrix rows, D2h semantics check, `*.jar` ignore rule, stale README
  references.
- B9 human-review provenance completion (already tracked as a pre-paper-table
  blocker in `reports/pps_architecture_readiness.md:100-102`).
- First-hand pytest/compileall run once interpreter execution is available
  (see §8); secondary evidence indicates 30/30 collected with no `lastfailed`.
- Test-split confirmation remains untouched and must stay untouched until the
  proposed-system configuration and all comparators are frozen.

---

## 8. Commands Run / Checks Not Run

**Run (read-only throughout; no file modified other than this report; no commit):**

- Independent recomputation via `jq` programs over
  `runs/*/per_request_metrics.jsonl` and `scores.jsonl` (per-run means,
  three-seed means/SDs, paired deltas on donor-joined subsets, no-history
  equality, log accounting via `sort | uniq -c`, `wc -l`).
- `sha256sum` on candidate manifest, qrels_dev, doc/18–21, all D-track
  base/final configs, calibration artifacts, checkpoints, score files —
  all matched recorded values.
- `jq empty` on all 160 JSON files (160/160 valid).
- Read/Grep/Glob inspection of: `src/myrec/eval/` (evaluator, metrics, compare,
  canaries), `src/myrec/baselines/core.py`, `src/myrec/analysis/{history_identity,
  supervised_diagnostics, finetuned_query_tower, m4_features}.py`,
  `src/myrec/data/{kuaisearch_standardize, kuaisearch_audit, kuaisearch_leakage,
  protocol_audit}.py`, all D1/D2/D2h/D2s scripts, all 26 YAML configs, all
  protocols doc/07–21, dev logs, lock manifests, `dev_eval_log.jsonl`,
  `experiments/pps_results.md`, `experiments/pps_baseline_cards.md`, and the
  paper files.
- Repo-wide pattern scans: `recently_*` consumers, qrels readers,
  `"split": "test"`, secrets patterns (0 hits), trailing whitespace (0 hits),
  large files (`du -a`; only the two vendored jars > 5 MB).

**Could not be run (environment permission denials, not repo defects):**

- `python3` execution in all forms → the unit-test suite
  (`python3 -m pytest tests/ -q`) and `compileall` were **not executed
  first-hand**. Secondary evidence: fresh `.pytest_cache/v/cache/nodeids`
  (today 19:55) lists 30 collected tests across 11 files with no `lastfailed`
  file, consistent with a passing 30/30 external run.
- Bit-exact re-execution of the 10,000-sample paired bootstrap (Mersenne
  Twister requires Python). Mitigated: every point delta recomputed exactly;
  normal-approximation CIs agree with claimed bootstrap CIs to ~3e-5 and
  independently confirm every claimed-positive lower bound.
- `git log/diff --check/ls-files` → replaced by the session git-status
  snapshot, `.git/index` probes, and text scans.
- `yq`/`yaml.safe_load` → replaced by manual full-file review of all 26 YAMLs.
- Full 36,687-row donor-uniqueness scan and donor-vs-target raw query-text
  byte comparison (line lengths exceeded tooling; Python denied) — verified
  instead by code inspection (pool-key construction guarantees) plus sampled
  rows.
- Anything involving `qrels_test.jsonl` content — excluded by mandate.

---

*This report is the first-pass audit deliverable. Per the mandate, the paper and
protocols were not altered. The verdict follows from independent recomputation
and code inspection, not from existing completion reports.*
