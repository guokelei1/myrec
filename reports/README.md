# Reports

Curated, paper-ready results. Tracked selectively (small files only).

## Checkpoint audit reports

Each phase gate (C0-C5) produces a JSON audit report here:

```text
reports/pps_c0_data_audit.json
reports/pps_c1_protocol.json
reports/pps_c3_motivation.json
reports/pps_c4_data_final.json
reports/pps_c5_insight.json
```

A positive claim may advance only after its gate passes. C01--C80 are closed;
C80 failed a pre-label mechanical contract, its utility is unknown, and there
is no C81. Current report promotion follows doc/31: mechanics, learnability,
utility, specificity, attribution, numerical safety, and confirmation are
separate report states. R0 strong-baseline and Failure-Card reports must not be
described as proposed-architecture results.

## Current Decision Reports

| File | Role |
|---|---|
| `pps_intro_motivation_completion_20260710.md` | Motivation completion and bounded design-stage transition decision |
| `pps_intro_motivation_repository_audit_20260710.md` | Repository-wide audit plus C3-R resolution pointer |
| `pps_m3_m4_random_canary_audit.json` | Permanent construct-validity failure for original M3/M4 |
| `pps_c3_motivation.json` | Historical C3 record; positive use superseded by C3-R |
| `pps_c3r_history_identity_control.json` | Historical train-frozen matched wrong-user result |
| `pps_c5r2_temporal_symmetric_identity.json` | Historical temporal-symmetric control and failed identity gate |
| `pps_c5r3_candidate_history_alignment.json` | Frozen item/category decomposition and `TERMINAL_FAIL` for the doc/23 recovery ladder |
| `pps_c5r3_consistency_audit.json` | Independent raw-metric recompute, hash/log/registry/test/repository audit for C5-R3 |
| `pps_c5_insight_audit.json` | Historical design-entry status; active-round wording is superseded by the C05 G2a gate report |
| `pps_c05_g2a_signal_gate.json` | C05 clean G0 pass and terminal G2a failure: zero ranking delta, common-mode `+1` score collapse, no G2b/CCEB/dev authorization |
| `pps_c02_mechanical_continuation_gate.json` | C02 valid one-shot continuation failure: five matched GPU fits, 2/6 internal checks pass, saturated candidate-common CHHT write, closed before dev |
| `pps_c06_synthetic_mechanism_probe.json` | C06 locked bidirectional synthetic pass: local Hodge trust is conditionally useful under planted coupling and harmful when coupling is absent/reversed; this report alone does not establish a real-data result |
| `pps_c06_real_mechanism_gate.json` | C06 terminal train-internal A0 failure: 0/1200 order and Top-10 changes, zero nontrivial delta-range fraction, A labels never opened |
| `pps_c10_synthetic_mechanism_gate.json` | C10 locked GPU synthetic failure: predictive write gives no non-repeat gain and loses to centered attention; includes the post-outcome generator-shortcut caveat and no-real authorization |
| `pps_c18_synthetic_gate.json` | C18 locked three-seed GPU failure: recurrence/order constraints pass exactly, but non-repeat transfer remains base-equivalent; no repository-data/dev/test authorization |
| `pps_c19_synthetic_gate.json` | C19 locked three-seed GPU failure: oriented lag is trainable but not stably better than forward/structured controls and is counterfactually unstable; no real-data authorization |
| `pps_c20_synthetic_gate.json` | C20 locked three-seed GPU failure: transition-cone solver/sign witness is active, but listwise transfer loses to pooled/span controls and shuffle retention is high; no real-data authorization |
| `pps_c21_train_signal_gate.json` | C21 locked train-only real-signal failure: active directed multi-step closure is base-equivalent, significantly worse than one-step, and not corruption-specific; no Transformer/dev/test promotion |
| `pps_c22_synthetic_gate.json` | C22 locked synthetic failure: exact layerwise filtration contracts and absolute utility pass, but dense/parallel/final-projection controls match or win; no repository-data authorization |
| `pps_c23_train_gate.json` | C23 locked label-free A0 failure: active recurrence-reset write ignores nontrivially shuffled post-anchor suffixes; internal-A/delayed/escrow/dev/test remain unopened |
| `pps_c24_train_gate.json` | C24 locked label-free A0 failure: cross-candidate edges change numerical corrections on every request but change 0/600 rankings in all three seeds; internal-A/escrow/dev/test remain unopened |
| `pps_c25_train_gate.json` | C25 locked label-free A0 failure: pure three-way pooled-state write changes only 2/1,200 top-10 sets and wrong histories change no top-10 sets; internal-A/delayed-B/escrow/dev/test remain unopened |
| `pps_c31_train_gate.json` | C31 train-only A1 failure: all seeds positive but overall CI and one fixed fold cross zero; delayed-B/escrow/dev/test remain unopened |
| `pps_c32_train_gate.json` | C32 first overall-positive tangent result: +0.004268 with positive CI and all seeds positive, but one frozen fold fails; controls/delayed-B/escrow/dev/test remain closed |
| `pps_c33_train_gate.json` | C33 fresh paired confirmation: tangent-D2p is +0.002988 with every seed/fold positive but CI crosses zero; +0.000583 over matched unprojected also crosses zero, so tangent closes at A1 |
| `pps_c34_train_gate.json` | C34 label-free A0 failure: candidate-specific cone writes are active and distinct, but only 2--5/15,424 candidate rows abstain per seed; the absolute half-space is nearly always-on, and A/B/dev/test remain closed |
| `pps_c39_train_gate.json` | C39 Amazon train-internal A1 failure: all 31 structural checks pass and the common trainable history Transformer strongly beats frozen BGE, but eventwise halfspace loses nominally to every matched control and true history ties wrong-user history; reserve/dev/test remain closed |
| `pps_c40_design_gate.json` | C40 data-free D1 failure: metric coupling is learnable and corruption-specific, but one seed misses the shifted-loop margin and the simpler semantic-value-preserving selection-only reduction wins every seed; no repository data/dev/test read |
| `pps_c41_design_gate.json` | C41 inherited design pass: exact C40 selection-only equivalence, raw semantic carrier, structural contracts, and conditional corruption evidence; boundary-only novelty status |
| `pps_c41_train_gate.json` | C41 A1 failure: routing-only primary strongly beats base/fixed attention and is true/wrong specific, but loses to C38 and coupled content; the pre-outcome coupled control triggers separately locked C42 confirmation |
| `pps_c42_confirmation.json` | C42 weights-preserving confirmation: frozen coupled checkpoints beat base and C38 with positive CIs and all seed/fold signs and retain true/wrong specificity, but advantages over semantic/asymmetric routing have CIs crossing zero; terminal gate fails, dev/test remain closed |
| `pps_c43_train_gate.json` | C43 KuaiSearch cross-domain A1 failure: exact metric-coupled structure weakly beats D2p with positive CI, but loses to shifted-loop, ties single-wide, lacks stable selection-only rent, and true history ties wrong-user history; dev/test remain closed |
| `pps_c44_design_gate.json` | C44 data-free D1 failure: partial candidate-plus-null logit flow is structurally valid and solves the planted task, but forced flow, partial vector write, and global pooling all tie it at perfect clean NDCG; no repository data read |
| `pps_c45_design_gate.json` | C45 data-free failure: factual-minus-NULL event tokens learn the task but lose to raw events and retain too much shuffled-event gain; a D0 gradient-check scope defect is recorded, and no repository data was read |
| `pps_c46_signal_gate.json` | C46 leakage-safe early-source signal failure: true user transitions beat shuffled pairing, but the content Transformer ties frozen semantic mean and lacks stable true/wrong specificity |
| `pps_c47_prelock_label_scope_incident.json` | C47 prelock train-label availability audit incident: 2,370 Kuai indices are conservatively fit-only and excluded from every C47 outcome role; no dev/test/qrels access |
| `pps_c47_signal_gate.json` | C47 fresh two-domain terminal failure: posterior-supported ridge beats query base and wrong history on Amazon, but ties softmax; on Kuai it loses to plain ridge in every fold and lacks stable specificity, so no trainable PSRT is authorized |
| `pps_c48_formulation_gate.json` | C48 exposed-cohort terminal formulation: signed KRR influence consensus is positive over base but fails to beat plain KRR stably on Kuai and loses to softmax on Amazon; fresh reserve remains closed |
| `pps_c49_learnability_gate.json` | C49 exposed dual-domain GPU failure: prequential innovation memory is active and trainable but ties raw KRR/DeltaNet on Kuai, collapses below base on Amazon, and has significantly wrong history specificity; reserve remains closed |
| `pps_c50_formulation_gate.json` | C50 zero-step terminal formulation: exact orthogonal protection of raw semantic memory holds numerically but still loses to raw KRR on both domains, so no training or fresh reserve is authorized |
| `pps_c51_formulation_gate.json` | C51 exposed two-domain terminal formulation: centered query/candidate event-affinity covariance gains slightly over base on Kuai but loses to uncentered and KRR controls, while its Amazon base gain is unstable; no training or fresh reserve is authorized |
| `pps_c52_formulation_gate.json` | C52 exposed dual-domain terminal formulation: token-level KRR query-concept bias is strongly rank-active but loses to its linearized reduction and pooled controls; no training or fresh reserve is authorized |
| `pps_c53_foundation_gate.json` | C53 dual-domain label-free A0 failure: ordinary joint list/history Transformer learns a mostly history-invariant candidate-list reranker; Kuai fails convergence/history load-bearing, A labels remain closed |
| `pps_c54_mechanism_gate.json` | C54 mechanics-only failure: history-only attention values reduce the C53 shortcut but do not make null contrast, candidate edges, or correct history Top-10-load-bearing under strong D2p; labels remain closed |
| `pps_c55_residual_signal_gate.json` | C55 fit-internal residual-signal failure: common score units remove Amazon weak-anchor overwrite, but history carrier ties wrong history on Kuai and loses to the history-free raw control on Amazon |
| `pps_c56_token_competition_signal_gate.json` | C56 label-free A0 failure: contextual token query-complement/factual-null carriers collapse to zero or candidate-common writes; raw candidate-list control remains strongly active, and holdout labels stay closed |
| `pps_c57_candidate_budget_attention_gate.json` | C57 label-free A0 failure: candidate-axis evidence allocation is ensemble-load-bearing, but corrections range from exact zero to overactive across seeds and loss/each-seed gates fail; holdout labels stay closed |
| `pps_c58_semantic_candidate_budget_gate.json` | C58 label-free numerical A0 terminal: fixed semantic candidate budgets pass all activity and fallback checks, but GPU candidate reductions exceed the frozen permutation tolerance; labels stay closed and utility is unknown |
| `pps_c59_exact_semantic_candidate_budget_gate.json` | C59 exact-reduction A1 failure: all mechanics pass and true history beats wrong history, but the semantic branch sharply degrades the strong base and candidate+NULL ties simpler axis/pooled controls |
| `pps_c60_base_order_edge_transport_gate.json` | C60 exposed formulation failure: conservative adjacent edge transport recovers nearly all C59 damage and beats wrong/raw/signed controls, but remains tied-slightly-below base and does not beat history-axis evidence |
| `pps_c61_counterfactual_edge_likelihood_gate.json` | C61 fresh label-free A0 failure: trained factual-minus-NULL edge likelihood has gradients and small fit-loss gains but produces no ranking change in any seed; fresh-A labels stay closed |
| `pps_c62_write_once_preference_memory_gate.json` | C62 data-free G0 failure: the two-phase history-write/query-candidate-read graph passes all structural and gradient contracts, but standard latent slots tie the pooled reduction and remain insensitive to wrong history; repository data stay closed |
| `pps_c63_finite_evidence_memory_gate.json` | C63 data-free G0 failure: event-wise finite write conservation and NULL mass are exact, but primary remains near four-interest chance, ties Slot Attention/OT/standard/pooled controls, and cannot identify nuisance or wrong history |
| `pps_c64_end_to_end_lm_representation_probe.json` | C64 exposed-fit label-free A0 failure: adapting the final two BGE layers makes rankings strongly active, but two seeds fail wrong-history Top-10 activity and bf16 listwise scoring fails permutation tolerance; validation labels stay closed |
| `pps_c65_counterfactual_residual_state_gate.json` | C65 label-free G0 terminal: factual-minus-NULL hidden states and gradients are active, but residual normalization amplifies caller-order roundoff above the frozen tolerance; no labels opened |
| `pps_c66_canonical_counterfactual_residual_state_gate.json` | C66 exact numerical continuation and label-free A0 failure: canonicalization makes every permutation check bit-exact, but wrong history changes only 1/10/7 Top-10 sets and validation labels stay closed |
| `pps_c67_cross_validated_fast_weight_gate.json` | C67 data-free failure: exact held-out fast-weight validation is mechanically sound but assigns uniform useful/nuisance mass, ties ordinary TTT/first-order controls, and retains about 96% accuracy under wrong history; repository data stay closed |
| `pps_c68_population_relative_interaction_free_energy_gate.json` | C68 data-free failure: four-way population-relative free energy satisfies exact cancellations but misses accuracy/wrong-history gates, reacts strongly to a fixed carrier, and ties mean/pooled reductions; repository data stay closed |
| `pps_c69_semantic_null_behavior_relation_gate.json` | C69 dual-domain signal failure: semantic-matched adjacent-event training is mechanically clean but trails semantic attention on both domains, collapses below its random-negative control on Amazon, and lacks stable true/wrong specificity; reserve/dev/test stay closed |
| `pps_c70_logged_choice_episode_coverage_gate.json` | C70 preimplementation data gate: 96.56% of Kuai historical choice episodes are recoverable but Amazon coverage is zero and JDsearch lacks historical slates, so a signed logged-choice Transformer is not authorized for GPU execution |
| `pps_c71_logged_choice_gradient_signal_gate.json` | C71 invalid fresh-role attempt: all label-free logged-choice mechanics pass, but all 600 unpacked targets have zero click and purchase positives, so utility is undefined and no signal conclusion is made |
| `pps_c72_exposed_logged_choice_gradient_diagnostic.json` | C72 exposed-fit formulation failure: logged-choice gradients are rank-active and nominally improve query cosine, but trail positive-only history, tie uniform slate centering, and lack stable true/wrong specificity |
| `pps_c73_counterfactual_query_relay_design_gate.json` | C73 data-free failure: query-mediated factual-minus-NULL token attention beats late and factual-only reductions, but one seed collapses and its all-seed margin over pooled query relay misses the frozen architecture-rent threshold; repository data stay closed |
| `pps_c74_semantic_conservative_query_relay_design_gate.json` | C74 data-free pass: learning only two-hop routing/chronology while conserving semantic values robustly beats coupled-value, pooled, and factual reductions in every seed |
| `pps_c74_pretrained_lm_probe_a0.json` | C74 real label-free A0 failure: token relay and true/wrong history are strongly Top-10-load-bearing, but two seeds fail the all-mode loss-trend gate; validation labels remain closed |
| `pps_amazon_token_history_observability_v1.json` | Ordinary full-token Amazon positive control: true-null +0.025298 and true-wrong +0.035944 with positive user-cluster CIs |
| `pps_amazon_token_edge_attribution_v1.json` | Frozen full-token edge attribution: Q--H, C--H, and history-read-context paths are load-bearing; same reserve, not fresh confirmation |
| `pps_c80_amazon_real_gate.json` | Terminal C80 pre-label mechanical failure: all 15 fits completed, event-permutation contract failed, 365 fresh labels unopened, utility unknown, no C81 |
| `pps_supervised_diagnostics_summary.json` | D1 supervised base/residual negative result |
| `pps_d2_d2h_summary.json` | Fine-tuned controls and the valid but interim D2h waterline |
| `pps_d2_score_audit.json` | Label-free D2 candidate/metadata integrity audit |
| `pps_d2h_score_audit.json` | Label-free D2h coverage and no-history fallback audit |
| `pps_d2s_summary.json` | Historical complete D2p + bundled-history reference at 0.3416 |
| `pps_d2s_score_audit.json` | Label-free D2s coverage and no-history fallback audit |
| `pps_d2s_protocol_lock_manifest.json` | D2s protocol/calibration/config/scoring/evaluation ordering proof |
| `pps_d2s_calibration_semantics_verification.json` | Exact scorer-z-score verification of the frozen D2s beta selection |
| `pps_intro_motivation_dev_eval_reconciliation.json` | Reconciliation of all 55 R1/B9/C3-R/D1/D2/D2h/D2s/C5-R2/C5-R3 evaluator invocations |
| `pps_architecture_readiness.md` | Current NOT-READY decision and R0/Failure-Card architecture-entry checklist |
| `pps_b9_neighbor_summary.md` | ZAM/TEM multi-seed evidence and review status |

Frozen computations are not deleted when a later audit invalidates their
interpretation. The later report must be linked from the original artifact and
from `experiments/pps_results.md`.

The C5-R3 `TERMINAL_FAIL` is scoped to the preregistered item/category recovery
ladder. The later full-token observability result changes the current priority:
first establish and tune the ordinary full-token strong baseline, then identify
a reproducible defect. No concrete architecture is validated and no new
architecture run is authorized before a doc/31 Failure Card passes.

C02 subsequently received one strictly mechanical continuation because its
original outcome was invalidated by an empty-mask NaN.  All five fixed variants
completed, but CHHT failed four of six train-internal conditions and produced
no dev scores or evaluator call.  C06 subsequently passed a CPU, data-free mechanism contract and received
explicit authorization for the smallest train-internal real mechanism gate.
After an independently reviewed numeric-equivalence repair, all four fixed fits
completed, but label-free A0 found `0/1200` order changes, `0/1200` Top-10
changes, and a zero nontrivial delta-range fraction. It therefore stopped with
`a1=null` before internal-A labels, B, escrow, dev, or test. C10 later failed
its locked synthetic GPU gate; C11--C17 were rejected before execution by
algebraic, complexity, fidelity, nearest-neighbor, or energy-reduction audits.
C18 subsequently passed its recurrence-safety, feasibility, load-bearing and
no-history contracts but failed all three non-repeat utility gates, so it also
closed before repository data. C19 and C20 then failed their locked GPU
synthetic gates: oriented lag paid no stable rent over forward controls, while
the active transition cone lost to simpler pooled/span controls and retained
too much shuffled-history gain. C21 subsequently tested directed contiguous
paths on a label-isolated C06-fit split; the write was active but base-equivalent,
lost significantly to one-step, and remained nonspecific under wrong/shuffled
history. C22 then implemented reliability-ordered one-way residual streams;
the invariant worked, but simpler matched controls solved the same synthetic
task as well or better. C23's reset write changed rankings but was essentially
invariant to the suffix it claimed to model, so it stopped before internal-A
labels. No proposed candidate currently has further
data-bearing authorization.

## What goes here vs. elsewhere

| Directory | Content | Tracked? |
|---|---|---|
| `runs/` | raw experiment output (logs, score dumps, raw metrics) | no |
| `artifacts/` | generated intermediate plots, tables, predictions | no |
| `reports/` | curated final results, audit JSONs, paper-ready tables | yes (small) |
| `paper/` | manuscript source and final selected assets | yes |

The `.gitignore` explicitly allows `reports/**/*.csv`, `*.tsv`, `*.json`,
`*.jsonl` so curated tables can be tracked.
