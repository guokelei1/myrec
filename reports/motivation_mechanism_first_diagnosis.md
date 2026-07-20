<!-- machine_report_json_sha256: f4e225256c461fed012c5953733cd8c0652c59cd7fceeef89658fcbf6ee38383 -->
# Motivation mechanism first diagnosis

Machine-readable report SHA-256: `f4e225256c461fed012c5953733cd8c0652c59cd7fceeef89658fcbf6ee38383`.

Status: `first_mechanism_diagnosis_complete`. Scope: KuaiSearch train-only diagnostics and internal-dev evaluation; the held-out evidence boundary remains closed.

## Integrated diagnosis

The first diagnosis does not support a single routing or sampling fix: visible-field signal and a localized Q2 preference representation exist, the final history-dependent readout state is causally used but carries harmful margin, block-13 movement is not universally request-specific, and simple surface balancing is opposed. The primary design opportunity is a unified ID-free factorized preference state plus an explicit signed candidate-residual path.

Claim boundary: M1 weakens relevant-history filtering as a sufficient routing explanation. M2 weakens universal abstraction absence because Q2 has localized decodability, and weakens simple readout non-use because the final mixed state causally reproduces harmful full-history margin; however, cross-request controls and Q2/Q3 heterogeneity prevent a preference-specific or model-universal mediation claim. M3 supports only a narrow final-state other-overlap gradient conflict and rejects simple complete surface balancing as the immediate remedy. All conclusions remain exploratory or boundary-level at one seed and one internal-dev population; the held-out evidence boundary remains closed and no transfer architecture has been implemented.

Next direction (requires new user authorization): pending user authorization

## LLM4Rec system architecture coverage

This matrix distinguishes partially probed and currently untested layers of the full ranking system. Untested rows are explicit gaps, not negative findings.

| Component ID | Component | Code paths | Parameter surfaces | Coverage | Evidence | Hypotheses | Limitations | Next preregistered probe | Negative controls |
|---|---|---|---|---|---|---|---|---|---|
| serialization_tokenization | Input-field whitelist, prompt serialization, history/candidate delimiters, token budgets, truncation, and tokenizer mapping | `src/myrec/baselines/motivation_v12_contracts.py`<br>`src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/history_interventions.py`<br>`src/myrec/mechanism/token_length_audit.py` | serialized query, history title/brand/category/event/prior-query, and candidate title/brand/category; raw item IDs and timestamps remain identity/provenance fields rather than prompt text<br>last-6-event history budget serialized newest first<br>2048-token model limit, with Q1 reserving 96 response tokens and a fixed 512-token context budget and Q3 reserving the complete answer suffix<br>condition-specific prompt tokens, truncation boundaries, and delimiter positions | partial | E_M1_INPUT_INTERVENTIONS<br>E_M1_TOKEN_LENGTH_CONTROL | H0<br>H1<br>H2<br>H5 | The token audit measures length and truncation but does not causally isolate delimiter, field-order, token-identity, or relative-position effects; M1 histories are only approximately length matched.<br>Q1 uses one multi-candidate instruction prompt and a marked-candidate response target, whereas Q0/Q2/Q3 use separate pointwise candidate prompts. | With new authorization, neutralize one field span at a time using fixed token count and fixed candidate/query text, re-encode identity controls, and test delimiter/field-order effects without selecting a condition after outcomes. | Byte- and token-identical re-encoding identity control.<br>Delimiter-only mask with equal token count.<br>Field-order permutation followed by exact inverse restoration. |
| token_embedding_position | Token embeddings, lexical embedding-state baselines, causal positions, and rotary-position exposure | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/representation_probe.py`<br>`src/myrec/mechanism/representation_evaluator.py` | tied input embedding table and output-head weight<br>hidden state index 0 before any Transformer block<br>no learned absolute position embedding; Qwen3 RoPE on normalized Q/K with theta 1000000 and no rope scaling<br>query/history/candidate token positions | partial | E_M2_Q2_REPRESENTATION<br>E_M2_Q3_REPRESENTATION | H2<br>H3<br>H5 | State 0 is the lexical input-embedding hook before Qwen3 constructs or applies RoPE; at history_summary_end it can expose terminal-token identity and is a baseline, not evidence of a preference representation or position mechanism.<br>Full and null prompts change token identity, sequence length, and query-history/candidate distances together; no causal position-ID or RoPE intervention is registered. | With new authorization, hold token IDs fixed while swapping registered history position IDs or RoPE offsets, and separately patch state-0 vectors with same-request identity and cross-request donors. | Fixed token IDs and unchanged position IDs.<br>State-0 full-to-full identity patch.<br>Random-label and cross-request state-0 controls. |
| attention_qkv | Transformer self-attention Q/K/V projections, Q/K normalization, head routing, and attention-logit construction | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/gradient_diagnostic.py`<br>`configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml`<br>`configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml` | 16 query heads and 8 key/value heads with head dimension 128 and two query heads sharing each key/value head<br>q_proj, k_proj, and v_proj<br>per-head q_norm and k_norm before RoPE<br>causal per-head attention logits, weights, and value contributions | partial | E_M3_Q2_GRADIENT_RESULTS<br>E_M3_Q3_GRADIENT_RESULTS | H1<br>H2<br>H3<br>H4 | The only current Q/V coverage is raw gradients: Q2 full q_proj/v_proj weights at zero-based blocks 0, 6, 13, 20, and 27, and Q3 LoRA A/B tensors for q_proj/v_proj in all 28 blocks. K, q_norm/k_norm, post-RoPE Q/K, logits, weights, value edges, and individual head routes are untested.<br>A parameter gradient does not establish that a projection or attention edge causally mediates strict-transfer behavior; the repository ranker loads the pinned Transformers Qwen3 implementation rather than implementing these branches itself.<br>No causal-mask, attention-backend, or KV-cache mechanism comparison is registered; any backend change needed to expose logits must first pass an end-to-end numerical identity audit. | With new authorization, patch pre-q_norm, post-q_norm, post-RoPE Q/K, and V token/head nodes at fixed layers; use attention-logit or single-edge value-contribution interventions for edge claims, report every registered head, and keep each KV head grouped with its two paired query heads. | Same-request full-to-full Q/K/V identity patches.<br>Cross-request same-layer patches.<br>GQA-group-preserving head permutation with exact inverse and equal norm. |
| attention_output | Attention-head concatenation and output projection before residual addition | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/patch_scorer.py` | 16-by-128 per-head attention output concatenated to width 2048<br>o_proj from width 2048 to residual width 1024<br>post-o_proj attention residual increment | untested |  | H1<br>H3 | Registered hidden-state capture and post-block patching occur after the attention output, first residual addition, pre-MLP RMSNorm, MLP, and second residual addition have mixed.<br>No current evidence isolates o_proj, the attention increment, or individual head contributions; these internal branches live in the pinned Transformers Qwen3 implementation rather than motivation_v12_ranker.py. | With new authorization, patch pre-o_proj head outputs, post-o_proj attention increments, and matched post-residual states at fixed layers to decompose attention-output mediation. | Pre/post-o_proj identity patch.<br>Head permutation paired with the exact inverse permutation of the corresponding o_proj input columns as a functional identity control.<br>Cross-request same-layer attention-output patch. |
| mlp | SwiGLU feed-forward gate, up projection, activation product, and down projection | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/patch_scorer.py` | gate_proj and up_proj from residual width 1024 to intermediate width 3072<br>SiLU(gate_proj(x)) multiplied elementwise by up_proj(x)<br>down_proj from width 3072 to residual width 1024<br>MLP residual increment | untested |  | H2<br>H3 | Post-block states do not distinguish feature transformation in the MLP from attention, normalization, or residual transport.<br>No registered neuron, gate, activation-product, or projection intervention exists; the branch definition is supplied by the pinned Transformers Qwen3 implementation. | With new authorization, patch gate/up products and down-projection outputs separately at fixed layers, preserving residual input and reporting the full registered layer set. | MLP identity patch.<br>Joint gate/up neuron permutation with the exact inverse permutation of down_proj input columns.<br>Cross-request same-layer MLP-output patch. |
| residual_norm | Pre-attention and pre-MLP RMSNorm, residual streams, and final normalization | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/patch_scorer.py` | input_layernorm and post_attention_layernorm RMSNorm with epsilon 1e-6<br>attention residual stream<br>MLP residual stream<br>final RMSNorm | partial | E_M2_Q2_REPRESENTATION<br>E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_REPRESENTATION<br>E_M2_Q3_PATCH_MEDIATION | H3<br>H5 | The shared Q2/Q3 sign reversal between block 13 and block 27 establishes a mixed residual-state transition but does not identify where within the intervening attention, MLP, residual, or normalization operations the harmful margin is created.<br>M2 captures post-block states after the second residual addition at blocks 6, 13, 20, and 27, but patches only blocks 13 and 27 at candidate_readout; state 28 is before final RMSNorm.<br>This is partial residual-stream coverage only: it does not isolate the attention increment, first residual addition, pre-MLP normalization, MLP increment, second residual addition, or final RMSNorm, and no norm-preserving intervention is registered. | With new authorization, decompose pre-attention state, attention increment, pre-MLP state, MLP increment, and pre/post-final-norm readout states at fixed layers. | RMS-norm-preserving random-direction control.<br>Pre/post-normalization identity patch.<br>Residual increment zero-and-restore control. |
| layerwise_representation | Embedding and mixed post-block hidden states at query end, history summary end, and candidate readout | `src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/representation_probe.py`<br>`src/myrec/mechanism/representation_evaluator.py` | hidden state indices 0, 7, 14, 21, and 28<br>query_end<br>history_summary_end<br>candidate_readout<br>linear readouts for the current best-gain candidate's brand and deepest category | partial | E_M2_Q2_REPRESENTATION<br>E_M2_Q3_REPRESENTATION | H2<br>H3<br>H5 | Coverage is limited to state 0 and post-block states 7/14/21/28 for Q2/Q3; no all-layer curve or Q0/Q1 internal analysis is authorized.<br>The probe label is the current highest-gain candidate's brand/deepest category (first candidate on a gain tie), not a directly observed user-preference factor; because history_summary_end also sees the query, decodability can come from query semantics.<br>Decodability is not causality, brand/category do not exhaust transferable preference, and full-versus-null states also differ in terminal token and relative-position layout. | With new authorization, preregister an all-layer curve using the identical train/query split and report every layer with no best-layer selection, followed only by fixed-layer causal confirmation. | Random labels.<br>Embedding state index 0.<br>Matched null/query-only and cross-request controls at every reported state.<br>Identity representation patch. |
| history_routing | Behavioral sensitivity to externally selected query-relevant histories and putative internal transport into model state | `src/myrec/mechanism/history_interventions.py`<br>`src/myrec/mechanism/recoverability_probe.py`<br>`src/myrec/mechanism/scorer.py`<br>`src/myrec/baselines/motivation_v12_ranker.py` | relevant_6 versus irrelevant_6 event sets<br>history order<br>external BGE query-to-history relevance ordering, not a learned Qwen router parameter<br>null-history response | partial | E_M0_RECOVERABILITY_CONTROLS<br>E_M1_INPUT_INTERVENTIONS<br>E_M1_TOKEN_LENGTH_CONTROL | H0<br>H1<br>H5 | There is no explicit router module in Q0-Q3: M0 recoverability and M1 relevance selection are external behavioral interventions and do not show that Qwen attention performed the selection.<br>Relevant and irrelevant histories can differ semantically and in exact token count even when event count and approximate length are controlled.<br>Current probes do not separate query-to-history formation at history-row/query-column edges from later history-to-candidate/readout transport at readout-row/history-column edges. | With new authorization, use a token- and position-matched fixed-slot control and separately intervene on registered query-to-history edges (history rows, query columns) and history-to-readout edges (readout row, history columns) using attention logits or single-edge value contributions; keep a whole-history attention-null ablation distinct from the frozen [NO_HISTORY] marker condition. | Equal-count random history removal.<br>Query-shuffled routing.<br>Attention-edge identity mask. |
| candidate_conditioned_interaction | Candidate-specific query/history interaction and downstream cross-candidate comparison | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`src/myrec/mechanism/patch_scorer.py`<br>`src/myrec/mechanism/patch_evaluator.py` | candidate readout positions<br>same-request target and competitor scores<br>candidate-overlap semantic swap<br>full-to-null candidate-conditioned readout-state patch | partial | E_M0_RECOVERABILITY_CONTROLS<br>E_M1_INPUT_INTERVENTIONS<br>E_M1_TOKEN_LENGTH_CONTROL<br>E_M2_Q2_REPRESENTATION<br>E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_REPRESENTATION<br>E_M2_Q3_PATCH_MEDIATION | H0<br>H2<br>H3 | Q2 block-13 cross-request movement is larger than its correct same-request movement, while Q3 has a descriptive same-versus-cross separation without stable preference decodability; the current state patch therefore does not establish a universal candidate-conditioned preference interaction.<br>Current patching replaces one mixed post-block state at the last prompt token for each candidate; it is neither a candidate-token span nor an isolated preference state and cannot separate attention, MLP, residual, or final-normalization effects.<br>Only Q1 encodes the full candidate slate in one forward pass. Q0/Q2/Q3 score separate pointwise candidate prompts; Q2 couples those scores in its training loss, while final cross-candidate comparison otherwise occurs in downstream scoring/evaluation.<br>The full donor and null recipient prompts differ in history-token count and relative positions, so a same-request patch can still mix content mediation with length/position effects despite identity and cross-request controls. | With new authorization, inject a preregistered history_summary_end full-minus-null history-contribution delta as null_state plus delta at a fixed layer, absolute position, and length; call it preference only if irrelevant/random-history controls and signed target-versus-competitor residuals pass under fixed query and slate. | Full-to-full identity patch.<br>Cross-request same-layer patch.<br>Candidate-slate permutation and inverse.<br>Query-only and common-score-offset controls. |
| readout_calibration | Native final decision positions, yes/no logit rows, generation likelihood, score margin, and rank calibration | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/patch_scorer.py`<br>`src/myrec/mechanism/patch_evaluator.py`<br>`src/myrec/mechanism/gradient_diagnostic.py` | Q0/Q2 single-next-token lowercase yes-minus-no raw-logit difference<br>Q3 teacher-forced complete Yes/No plus end-token mean log-likelihood difference<br>Q1 mean token log-likelihood of the exact ordinal-marked visible candidate line plus end token<br>tied output embedding rows<br>target-versus-best-lower-gain-competitor margin<br>graded NDCG@10 | partial | E_M0_RECOVERABILITY_CONTROLS<br>E_M1_INPUT_INTERVENTIONS<br>E_M2_Q2_REPRESENTATION<br>E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_REPRESENTATION<br>E_M2_Q3_PATCH_MEDIATION<br>E_M3_Q2_MATCHED_CONTROL_RESULTS | H3<br>H5 | Correct block-27 restoration recreates the negative full-minus-null target margin in both models, and wrong block-27 donors amplify strict-transfer harm; this establishes final-state use but not a beneficial or calibrated preference readout.<br>M0 and M1 endpoint disagreements show calibration sensitivity but do not identify the native Qwen readout path.<br>Registered M2 patches cover Q2/Q3 candidate_readout states at blocks 13 and 27 only. For Q3, a block-13 patch can propagate through later layers to subsequent answer positions, but a block-27 patch changes the first-answer-token logit only and cannot alter later answer-position states already computed in that block.<br>Q1 multi-token likelihood mediation is untested. | With new authorization, patch pre- and post-final-norm states at every native decision position, including each Q3 teacher-forced answer-prediction position and Q1 token-by-token response likelihood, under identity and all-vocabulary common-offset controls. | Exact readout-weight identity patch.<br>All-vocabulary common logit-offset invariance control.<br>Full-to-full native-readout identity patch.<br>Cross-request same-position patch. |
| loss_gradient | Native ranking/alignment losses, per-surface gradients, directions, and normalized squared raw-gradient-mass proxies | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/gradient_diagnostic.py`<br>`src/myrec/mechanism/matched_training_control.py` | Q2 combined 0.5 RankNet plus 0.5 tie-aware ListNet objective<br>Q3 mean answer-token NLL for complete Yes/No plus end-token sequences<br>recurrence, strict-transfer, and other-overlap gradients<br>Q2 full q_proj/v_proj weights at blocks 0/6/13/20/27 and the monitored yes/no rows of the tied embedding/output matrix<br>Q3 all trainable LoRA Q/V parameters | partial | E_M3_Q2_GRADIENT_RESULTS<br>E_M3_Q3_GRADIENT_RESULTS<br>E_M3_Q2_MATCHED_CONTROL_RESULTS | H4<br>H5 | M3 has no gradient coverage for Q0 pointwise BCE or Q1 candidate-response NLL, and neither Q2 nor Q3 diagnostics cover every backbone tensor family.<br>Q2 records RankNet and ListNet scalar components but differentiates only their fixed weighted sum; component-specific gradients are not isolated.<br>The normalized shares are sums of per-request squared raw-gradient norms; they omit clipping, AdamW moments, weight decay, and scheduler scaling and are not effective-update shares.<br>Reported surface cosines compare sums of request gradients over the registered coordinates, not mean pairwise request-gradient cosines; Q2 and Q3 parameterizations are not directly norm-comparable.<br>Q2 yes/no row gradients mix tied input-embedding and output-readout roles rather than isolating a pure lm_head. | With new authorization, use fixed train request IDs to decompose native objectives across Q0-Q3, separately report Q2 RankNet/ListNet gradients, and register all tensors or fixed tensor families before outcomes. | Within-request label shuffle is the registered statistical negative control; equal counts, fixed RNG, and restore checks are sampling/reproducibility/integrity controls rather than mechanism negatives. |
| optimizer_scheduler | Gradient accumulation, clipping, AdamW moments, weight decay, warmup/linear schedule, checkpoint resume, and effective parameter updates | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/matched_training_control.py`<br>`configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml` | Q2-only accumulation of 16 one-request microbatches per optimizer step<br>global gradient clipping at 1.0<br>AdamW moments and weight decay<br>linear warmup and decay<br>4096-group, 256-step matched diagnostic retraining from a common base initialization | untested |  | H4<br>H5 | M3 gradient diagnostics execute no optimizer step; the Q2 matched arms hold AdamW, scheduler, clipping, and step count fixed while changing the sampling mixture, so neither identifies an optimizer/scheduler mechanism.<br>The matched arms start from the same base-model initialization with fresh optimizer state; they are not a replay from the frozen final checkpoint or its AdamW moments.<br>Gradient accumulation differs across frozen Q0/Q1/Q2/Q3 configs (2/8/16/8), while the matched training control is Q2-only. | With new authorization, restore the identical parameter, AdamW-moment, variance, scheduler, and RNG state; compare preregistered one-surface-at-a-time or order-averaged microgradient counterfactuals, report the combined update, and separate weight decay from data-gradient effects. | Exact checkpoint-plus-optimizer-state no-step identity reload.<br>Zero-data-gradient replay with weight decay separately enabled and disabled.<br>Microbatch-order permutation of the identical multiset with numerical-tolerance reporting.<br>Resume-at-boundary identity replay. |
| adapter_lora | Q3 low-rank adaptation path and its restriction to attention Q/V projections | `src/myrec/baselines/motivation_v12_ranker.py`<br>`src/myrec/mechanism/gradient_diagnostic.py`<br>`src/myrec/mechanism/representation_runtime.py`<br>`configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml` | q_proj LoRA A/B matrices<br>v_proj LoRA A/B matrices<br>rank 8, alpha 16, and alpha/r effective scale 2<br>LoRA dropout 0.05<br>all 28 Transformer blocks | partial | E_M2_Q3_REPRESENTATION<br>E_M2_Q3_PATCH_MEDIATION<br>E_M3_Q3_GRADIENT_RESULTS | H2<br>H3<br>H4<br>H5 | M3 observes raw gradients for every trainable Q3 LoRA q_proj/v_proj A/B tensor. M2 observes mixed hidden states of the adapted whole model, not an isolated LoRA contribution; neither causally separates q-only, v-only, layer groups, or low-rank directions.<br>A/B-coordinate gradient norms, especially at the zero-delta base initialization, are parameterization dependent and are not equivalent to function-space or merged-delta effects.<br>Q3 cannot update K/O, MLP, norm, embedding, or output-head parameters under the frozen adapter recipe. | With new authorization, scale Q3 LoRA deltas by fixed q-only, v-only, and preregistered layer groups at inference, reporting the complete group grid without outcome-based layer selection. | LoRA scale 1 plus merge/unmerge numerical identity; scale 0 reported separately as the base-model ablation rather than an identity negative.<br>Projection- and shape-preserving cross-layer delta permutation reported as a destructive control.<br>Equal-Frobenius-norm low-rank direction shuffle.<br>Dropout-disabled deterministic replay. |

## Stage summaries

### M0

M0 establishes a nontrivial but narrow visible-field signal boundary; it does not establish a positive strict-transfer recovery ceiling.

- The internal-dev partition reconstructs all 8000 requests and contains 2195 strict-transfer requests in 2074 normalized-query clusters; the two-sided alpha-0.05, power-0.80 MDE ranges from 0.00900 to 0.01505 across Q0-Q3.
- Visible-field alignment is present but incomplete: 50.07% of strict-transfer targets share a history brand, 43.37% share a category prefix, and 15.26% share the deepest category; the mean history-semantic target-minus-best-competitor margin is -0.05782.
- For the preregistered raw-item-ID-free linear probe, full versus null strict-transfer NDCG is +0.003122 with its registered interval crossing zero and opposite fold directions, while the target margin is -0.039827 with a negative interval and both folds negative.
- Correct history beats the frozen wrong-history assignment control on strict-transfer NDCG by +0.012261 with both folds positive; the corresponding target-margin estimate is -0.001398 with an interval crossing zero and opposite fold directions. Label shuffle is negative on both registered endpoints, and eight of the twelve family hypotheses reject after Benjamini-Hochberg correction.

Limitations:

- M0 uses one KuaiSearch internal-dev population and request/query-cluster uncertainty; it does not cover training-seed uncertainty or forward-temporal generalization.
- The recoverability control is a fixed train-only linear probe over frozen visible-field semantic features, not a Transformer architecture attribution or a new ranking method.
- Endpoint disagreement and fold disagreement prevent a positive recoverability-ceiling claim and must be carried into M1-M3 interpretation.
- The registered history-shuffle condition loads frozen other-user history assignments, with a global-other-user fallback when an exact-query donor is unavailable; it is not a within-history order shuffle or a provenance-matched causal user-specificity control.

### M1

M1 finds no positive FDR-controlled strict-transfer NDCG benefit from any registered history intervention; the few surviving effects are endpoint- and model-specific, while the token audit bounds but does not eliminate serialization confounding.

- All registered Q0-Q3 comparisons and both endpoints form one family; five of forty-eight reject after Benjamini-Hochberg correction, and none is a positive NDCG intervention benefit.
- Relevant-only versus irrelevant-only NDCG is positive for Q0 (+0.006752) and Q2 (+0.007918) with both query folds positive, but neither survives family correction (q=0.242 and q=0.132); Q0 target margin instead decreases by -0.009600 (q=0.006399). Relevant-only does not significantly beat frozen full history for any model.
- The registered category-path/brand-prioritized different-ID replacement establishes no positive FDR-controlled endpoint over semantic breaking or frozen full history: preserving-versus-breaking target margins are negative in all four models, with Q0 (-0.009785) and Q3 (-0.006979) rejecting; Q1 preserving versus full also reduces NDCG by -0.007892 (q=0.038392). Uncorrected positive estimates remain, including Q2 preserving-versus-full margin +0.007236 (q=0.281544, both folds positive), so this is not a no-effect claim.
- The only positive surviving effect is Q1 order-shuffle versus frozen-full target margin (+0.000151, q=0.033593), a tiny isolated margin result; its NDCG estimate is -0.000570 with an interval crossing zero, q=1.000000, and opposite fold directions. Candidate-overlap semantic swap is exactly zero on the candidate-disjoint strict-transfer surface for every model, endpoint, and fold.
- Q0, Q2, and Q3 have no request truncation in any audited condition. Q1 frozen-full truncates 22.675% of requests, but intervention-specific rate changes are at most 0.1625 percentage points; order shuffle is exactly total-token-count matched.

Limitations:

- Input interventions are end-to-end behavioral probes and cannot by themselves identify a specific Transformer submodule or distinguish absent preference representation from unused representation.
- Relevant and irrelevant histories still differ in semantic content and have small token-length differences, so their contrast is not a pure semantic-causality claim.
- The semantic-preserving intervention prioritizes same-brand plus exact-category-path donors but allows an exact-category-path-only fallback and replaces other visible donor-event fields; it is not an all-visible-field semantic-invariance intervention.
- The token summary is an aggregate over all 8000 internal-dev requests rather than a strict-transfer-stratified audit. Except for order shuffle, it bounds but does not eliminate condition-specific serialization confounding.
- The exact-zero overlap-swap result is an isolation control induced by the strict-transfer surface contract; it is not evidence that overlap is behaviorally irrelevant on overlap-bearing requests.

### M2

M2 finds a localized Q2 preference representation but no stable Q3 analogue; causal patches then reveal a cross-layer sign reversal in both models, where the final candidate-readout state reproduces harmful full-history margin while block-13 restoration moves margin in the opposite direction with model-dependent donor specificity.

- All ten registered activation bundles are complete with exactly 8000 requests and 160753 candidates on dev plus the complete registered train-probe population; the representation evaluator reports the full preregistered condition, state, surface, task, label-control, and fold grids rather than selecting a layer after outcomes.
- For Q2 strict-transfer category at state 28, real-label versus random-label balanced accuracy is 0.448148 versus 0.256481 for full, 0.311111 versus 0.224537 for null, 0.506019 versus 0.267130 for relevant-only, and 0.398148 versus 0.245833 for irrelevant-only. The full-versus-null excess-decoding direction is positive in both folds, whereas the relevant-versus-irrelevant direction disagrees across folds. This category cell contains only 165 labelled requests.
- For Q2 strict-transfer brand at state 7, full real-label versus random-label balanced accuracy is 0.539379 versus 0.497523, while both null controls are 0.500000; the full-versus-null excess-decoding direction is positive in both folds. Q2 full-minus-null candidate-readout state distance at state 28 is 3.784604 L2 per square-root hidden dimension over all 2195 strict-transfer requests, so history-dependent state change reaches the registered readout position but is not yet causally tied to ranking.
- Q3 does not reproduce the Q2 pattern: strict-transfer category state-28 full real-label versus random-label balanced accuracy is 0.253704 versus 0.308796, while null is 0.313889 versus 0.281481; the derived full-versus-null excess-decoding direction is negative and folds disagree. Q3 still has a nonzero full-minus-null candidate-readout state distance of 1.263191, which shows state sensitivity but not stable preference decodability.
- Both patch evaluators pass all seven pre-qrels integrity checks. Q2 identity controls have zero maximum score error; Q3 identity maximum error is 0.000000029802322387695312, below the 0.000010 tolerance. All six registered patch cells, both blocks, both surfaces, and both query folds are retained for each model.
- On 2194 margin-eligible strict-transfer requests, Q2 full-minus-null margin is -0.029227. The same-request block-13 patch is +0.004415 above null, giving a ratio of -0.151072 with a registered interval of -0.402493 to +0.007165; block 27 exactly reproduces the harmful full response with ratio 1.000000. The block-13 cross-request control is larger in the opposite-to-harm direction (ratio -0.641326, interval -1.091089 to -0.388761), while the block-27 cross control amplifies harm (ratio 4.395712, interval 3.065657 to 6.817771).
- For Q3 strict transfer, full-minus-null margin is -0.014785. The same-request block-13 patch is +0.002849 above null, ratio -0.192679 with interval -0.385673 to -0.046015; block 27 reproduces the harmful full response with ratio 0.999999. The block-13 cross control is near zero with an interval crossing zero (ratio -0.042390, interval -0.199664 to +0.088611), while the block-27 cross control amplifies harm (ratio 4.023133, interval 3.222642 to 5.147349).
- The observed-positive all-request surface shows the same layer sign reversal: Q2 and Q3 same-request block-13 ratios are -0.143598 and -0.026653 with intervals below zero, while block 27 is approximately 1. Cross-request block-13 is also negative for Q2 (-0.210658) but near zero for Q3 (+0.003731), so request specificity is heterogeneous and no model-universal preference-mediation claim is available.

Limitations:

- The block-13 same-versus-cross donor contrast is descriptive; no separate paired inference was registered for their difference, and Q2's larger cross-request effect specifically prevents interpreting its same-request block-13 response as user-specific preference mediation.
- Q2 category labels cover only 165 strict-transfer requests, some routing contrasts disagree across folds, and the Q2/Q3 disagreement prevents a model-universal representation claim.
- The denominator is negative in both models: a positive ratio at block 27 means mediation of harmful full-history margin, not preference recovery. Ratios are not constrained to the unit interval.
- The registered anchors are Q2 and Q3 only, and post-block hidden states mix attention, MLP, residual, and normalization effects; Q3 block 27 directly changes only the first answer-token prediction.

### M3

M3 finds checkpoint- and anchor/scope-dependent gradient allocation plus a narrow final-state conflict with other-overlap gradients, while the fixed-exposure Q2 surface-balance control fails to improve strict-transfer NDCG and reliably worsens target margin.

- Q2 observed raw-gradient mass moves from recurrence/strict-transfer/other-overlap = 31.53%/20.77%/47.70% at base to 22.55%/40.62%/36.82% at final; observed recurrence-strict cosine changes from -0.279 to +0.389, while final recurrence-other and strict-other cosines are -0.358 and -0.150.
- Q3 observed raw-gradient mass is approximately balanced at base (33.00%/33.85%/33.15%) with all surface cosines near +0.997, but final recurrence/strict-transfer/other-overlap mass becomes 55.69%/22.68%/21.62%; final recurrence-strict cosine remains positive (+0.127), while recurrence-other and strict-other are -0.540 and -0.320.
- For both final anchors, observed other-overlap gradients oppose recurrence and strict-transfer, whereas every final within-request label-shuffle pairwise cosine is nonnegative. However, label shuffle concentrates recurrence mass even more strongly (Q2 41.80%; Q3 64.75%), so recurrence mass alone is not a clean label-sensitive shortcut diagnostic.
- Both Q2 diagnostic arms complete the exact same registered recipe with 4096 group exposures, 256 optimizer steps, no resume, a common base initialization, and distinct preregistered sampling selections; fixed-recipe and score-bundle admission pass before shared evaluation.
- On 2195 strict-transfer requests, original-mixture full-minus-null NDCG is +0.008119 and surface-balanced is +0.012444, but the preregistered joined difference-in-differences is only +0.004324 (registered query-cluster interval -0.006232 to +0.015086, q=0.421516) with opposite fold directions.
- The strict-transfer target-margin difference-in-differences is -0.081230 (registered interval -0.089781 to -0.072596, q=0.000800) with both folds negative. Thus simple surface-balanced exposure does not provide a credible strict-transfer ranking gain and produces a coherent adverse calibration/separation result.

Limitations:

- Normalized shares are sums of per-request squared raw-gradient norms over registered parameter subsets with no optimizer step; they are not effective parameter updates and omit clipping, AdamW moments, weight decay, and scheduler scaling.
- Q2 and Q3 use different registered trainable-parameter scopes, so cross-anchor mass values are diagnostic contrasts rather than directly interchangeable full-model update budgets or an isolated anchor effect.
- Each gradient cell contains 96 requests, and the reported shares and surface-mean cosines are point aggregates without bootstrap intervals, fold estimates, p-values, or FDR inference.
- At Q3 base initialization, all 56 registered LoRA-A scope-gradient means are exactly zero in every cell while LoRA-B gradients are nonzero; the near-collinear base geometry and base-to-final change therefore also reflect zero-B LoRA initialization and a change in active gradient coordinates.
- The matched control changes the complete train sampling mixture rather than isolating recurrence, nonpersonalized relevance, one loss component, or an optimizer coordinate; it is Q2-only, short-horizon, and diagnostic-only, so it cannot be presented as the paper method.

## M2 registered strict-transfer patch grid

Every registered model, patch kind, block, and query fold is shown. Interpret the full-minus-null denominator sign before the ratio; the ratio is not constrained to [0, 1], and post-block patching does not isolate attention, MLP, residual, or normalization branches.

| Model | Patch kind | Block | Fold | Requests | Mean patch-null margin | Mean full-null margin | Mediated fraction | 95% cluster CI | Negative control |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| q2 | same_request_full_to_null | 13 | all | 2194 | 0.004415451230628988 | -0.029227438468550592 | -0.15107212475633527 | -0.40249339249277627<br>0.0071647304506527445 | false |
| q2 | same_request_full_to_null | 13 | 0 | 1106 | 0.0019213381555153706 | -0.03610985533453888 | -0.053208137715179966 | -0.3333333333333333<br>0.1174939134532317 | false |
| q2 | same_request_full_to_null | 13 | 1 | 1088 | 0.0069508272058823525 | -0.022231158088235295 | -0.3126614987080103 | -1.2353534687417569<br>-0.005238883183080528 | false |
| q2 | same_request_full_to_null | 27 | all | 2194 | -0.029227438468550592 | -0.029227438468550592 | 1.0 | 1.0<br>1.0 | false |
| q2 | same_request_full_to_null | 27 | 0 | 1106 | -0.03610985533453888 | -0.03610985533453888 | 1.0 | 1.0<br>1.0 | false |
| q2 | same_request_full_to_null | 27 | 1 | 1088 | -0.022231158088235295 | -0.022231158088235295 | 1.0 | 1.0<br>1.0 | false |
| q2 | full_to_full_identity | 13 | all | 2194 | -0.029227438468550592 | -0.029227438468550592 | 1.0 | 1.0<br>1.0 | true |
| q2 | full_to_full_identity | 13 | 0 | 1106 | -0.03610985533453888 | -0.03610985533453888 | 1.0 | 1.0<br>1.0 | true |
| q2 | full_to_full_identity | 13 | 1 | 1088 | -0.022231158088235295 | -0.022231158088235295 | 1.0 | 1.0<br>1.0 | true |
| q2 | full_to_full_identity | 27 | all | 2194 | -0.029227438468550592 | -0.029227438468550592 | 1.0 | 1.0<br>1.0 | true |
| q2 | full_to_full_identity | 27 | 0 | 1106 | -0.03610985533453888 | -0.03610985533453888 | 1.0 | 1.0<br>1.0 | true |
| q2 | full_to_full_identity | 27 | 1 | 1088 | -0.022231158088235295 | -0.022231158088235295 | 1.0 | 1.0<br>1.0 | true |
| q2 | cross_request_same_layer | 13 | all | 2194 | 0.018744302643573383 | -0.029227438468550592 | -0.6413255360623782 | -1.0910889181543264<br>-0.3887605836306513 | true |
| q2 | cross_request_same_layer | 13 | 0 | 1106 | 0.019326401446654613 | -0.03610985533453888 | -0.5352112676056339 | -1.0315885604422015<br>-0.27952022690800327 | true |
| q2 | cross_request_same_layer | 13 | 1 | 1088 | 0.018152573529411766 | -0.022231158088235295 | -0.8165374677002585 | -2.3846666666666665<br>-0.3649597864287646 | true |
| q2 | cross_request_same_layer | 27 | all | 2194 | -0.128475387420237 | -0.029227438468550592 | 4.395711500974659 | 3.065657355187051<br>6.817771172561829 | true |
| q2 | cross_request_same_layer | 27 | 0 | 1106 | -0.12019665461121157 | -0.03610985533453888 | 3.328638497652582 | 2.048847000017996<br>5.8723151579472335 | true |
| q2 | cross_request_same_layer | 27 | 1 | 1088 | -0.13689108455882354 | -0.022231158088235295 | 6.157622739018088 | 3.5099426851491855<br>16.162864576641397 | true |
| q3 | same_request_full_to_null | 13 | all | 2194 | 0.002848677608840772 | -0.01478459307738837 | -0.19267879703754265 | -0.38567285765062603<br>-0.046015485326034264 | false |
| q3 | same_request_full_to_null | 13 | 0 | 1106 | 0.0029385345322745188 | -0.01661385919150996 | -0.1768724832925136 | -0.42974744773956863<br>0.005976807120782144 | false |
| q3 | same_request_full_to_null | 13 | 1 | 1088 | 0.002757334081894335 | -0.012925063369466978 | -0.2133323453104312 | -0.5505121728506183<br>0.023529133477324667 | false |
| q3 | same_request_full_to_null | 27 | all | 2194 | -0.01478458007792505 | -0.01478459307738837 | 0.9999991207425695 | 0.9999964236912978<br>1.0000018524466747 | false |
| q3 | same_request_full_to_null | 27 | 0 | 1106 | -0.016613868555259533 | -0.01661385919150996 | 1.000000563610746 | 0.9999971685483061<br>1.0000043617100816 | false |
| q3 | same_request_full_to_null | 27 | 1 | 1088 | -0.012925027636811137 | -0.012925063369466978 | 0.9999972353980154 | 0.9999926195890758<br>1.0000017173325708 | false |
| q3 | full_to_full_identity | 13 | all | 2194 | -0.014784592832884358 | -0.01478459307738837 | 0.9999999834622427 | 0.9999999607040332<br>1.0000000053400844 | true |
| q3 | full_to_full_identity | 13 | 0 | 1106 | -0.016613859124144852 | -0.01661385919150996 | 0.9999999959452464 | 0.9999999674874248<br>1.0000000242614369 | true |
| q3 | full_to_full_identity | 13 | 1 | 1088 | -0.012925062944893451 | -0.012925063369466978 | 0.9999999671511454 | 0.999999925237954<br>1.0000000009138383 | true |
| q3 | full_to_full_identity | 27 | all | 2194 | -0.014784592832884358 | -0.01478459307738837 | 0.9999999834622427 | 0.9999999607040332<br>1.0000000053400844 | true |
| q3 | full_to_full_identity | 27 | 0 | 1106 | -0.016613859124144852 | -0.01661385919150996 | 0.9999999959452464 | 0.9999999674874248<br>1.0000000242614369 | true |
| q3 | full_to_full_identity | 27 | 1 | 1088 | -0.012925062944893451 | -0.012925063369466978 | 0.9999999671511454 | 0.999999925237954<br>1.0000000009138383 | true |
| q3 | cross_request_same_layer | 13 | all | 2194 | 0.0006267149120216057 | -0.01478459307738837 | -0.042389730223965826 | -0.19966448280654273<br>0.08861089813553356 | true |
| q3 | cross_request_same_layer | 13 | 0 | 1106 | 5.654277950255823e-05 | -0.01661385919150996 | -0.0034033501097356605 | -0.20189156250657747<br>0.15803649225702757 | true |
| q3 | cross_request_same_layer | 13 | 1 | 1088 | 0.0012063200393801227 | -0.012925063369466978 | -0.09333184719463937 | -0.3869023433426191<br>0.11877656133126095 | true |
| q3 | cross_request_same_layer | 27 | all | 2194 | -0.05948038621989835 | -0.01478459307738837 | 4.02313312977602 | 3.222641773210669<br>5.147349053768759 | true |
| q3 | cross_request_same_layer | 27 | 0 | 1106 | -0.05645342334215723 | -0.01661385919150996 | 3.3979716988937856 | 2.514666463161297<br>4.732567469117493 | true |
| q3 | cross_request_same_layer | 27 | 1 | 1088 | -0.0625574275276021 | -0.012925063369466978 | 4.840009347681978 | 3.5016898623882833<br>7.197886866139089 | true |

## H0-H5 evidence matrix

| ID | Status | Claim level | Supporting evidence | Opposing evidence | Rationale |
|---|---|---|---|---|---|
| H0 | unresolved | boundary | E_M0_RECOVERABILITY_CONTROLS | E_M0_DATA_SIGNAL_POWER<br>E_M2_Q2_REPRESENTATION | M0 shows incomplete semantic alignment and no positive full-versus-null recovery ceiling, but it also shows nonzero brand/category alignability, 2195 requests, and adequate power for effects near 0.02. Q2 independently decodes a localized real-label brand/category preference proxy beyond random labels and shows history-dependent candidate-readout state change, arguing against complete visible-field signal absence; Q3 instability and the lack of causal mediation still prevent separating weak signal from model inability or effects below the registered MDE. |
| H1 | weakened | exploratory | E_M0_RECOVERABILITY_CONTROLS | E_M1_INPUT_INTERVENTIONS | M0 shows that correct rather than frozen wrong-history assignments can matter for NDCG, but M1 does not establish that filtering to registered relevant events recovers a coherent strict-transfer benefit: relevant-only never significantly beats frozen full history, and Q0/Q2 relevant-versus-irrelevant NDCG signals fail family correction while Q0 margin reverses significantly. This weakens query-aware selection failure as a sufficient behavioral explanation, without testing attention routing or an explicit null gate. The all-population token audit only bounds, and does not pass as, a mechanism-level negative control for this contrast. |
| H2 | weakened | exploratory | E_M1_INPUT_INTERVENTIONS<br>E_M2_Q3_REPRESENTATION<br>E_M2_Q2_PATCH_MEDIATION | E_M2_Q2_REPRESENTATION | M1 establishes no FDR-controlled positive different-ID behavioral benefit and preserves endpoint reversals, which is compatible with H2. Q2 nevertheless contains a localized brand/category preference signal beyond random labels, so the universal absence-of-abstraction claim is weakened. The patch evidence does not convert that decodability into request-specific preference use: Q2 block-13 cross-request restoration moves strict-transfer margin farther opposite the full-history harm than the correct same-request donor, while Q3 does not reproduce stable preference decodability. Usable abstraction remains weak, but exact recurrence reliance itself is not established. |
| H3 | weakened | exploratory | E_M2_Q2_REPRESENTATION | E_M2_Q3_REPRESENTATION<br>E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_PATCH_MEDIATION | Q2 supplies the representation-side precondition for H3, but the literal unused-readout explanation is weakened. In both models the correct block-27 full state recreates the negative full-minus-null target margin with mediated fraction approximately 1, proving that the final mixed state is used, but used harmfully. Block 13 moves margin in the opposite direction; Q2 fails request specificity because its cross-request control is larger, while Q3 has a clearer descriptive same-versus-cross separation but lacks stable preference decodability. The active readout problem is therefore sign, calibration, and preference specificity rather than simple non-use. |
| H4 | weakened | exploratory | E_M3_Q2_GRADIENT_RESULTS<br>E_M3_Q3_GRADIENT_RESULTS | E_M3_Q2_MATCHED_CONTROL_RESULTS | Both final anchors show an observed label-sensitive directional-conflict point pattern between other-overlap gradients and recurrence/strict-transfer gradients: the observed cosines are negative while all within-artifact label-shuffle cosines are nonnegative. Recurrence mass dominance appears only in Q3, recurrence and strict-transfer are aligned at both final anchors, and label shuffle increases recurrence mass further. The independent fixed-exposure Q2 training control then fails its preregistered improvement prediction: its strict-transfer NDCG DID is inconclusive with opposite folds, while its target-margin DID is reliably adverse. H4 is therefore weakened: a narrow final-state surface-conflict point pattern remains supported, but the broad easy-surface-domination story and simple sampling-balance remedy are not supported. |
| H5 | unresolved | boundary | E_M0_RECOVERABILITY_CONTROLS<br>E_M1_INPUT_INTERVENTIONS<br>E_M2_Q3_REPRESENTATION<br>E_M3_Q2_MATCHED_CONTROL_RESULTS | E_M0_DATA_SIGNAL_POWER<br>E_M1_TOKEN_LENGTH_CONTROL<br>E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_PATCH_MEDIATION<br>E_M3_Q2_GRADIENT_RESULTS<br>E_M3_Q3_GRADIENT_RESULTS | The primary M0 full-versus-null NDCG folds disagree, M1 leaves only five of forty-eight corrected effects with mixed endpoint and fold directions, Q2/Q3 representation patterns differ, and the matched-control NDCG DID has opposite folds; these observations keep measurement/model instability live. Against a pure instability explanation, both models share the same patch geometry: correct block 27 reproduces the harmful full response and wrong block-27 donors amplify it to more than four times the strict-transfer denominator; both final M3 anchors also share an other-overlap conflict direction, and the matched-control adverse margin DID is nearly identical across folds. Block-13 donor specificity still differs by model, and one seed plus one internal-dev population cannot resolve H5. |

### Hypothesis details

#### H0 — unresolved

Statement: 当前可见历史与商品文本缺少足够、可学习的跨商品偏好信号，或 strict-transfer 测量功效不足

Component statuses:

- `complete_visible_field_signal_absence`: `weakened`
- `recoverability_ceiling`: `unresolved`
- `measurement_power`: `unresolved`

Triangulation roles:

- Reversible behavior: none
- Independent source: E_M2_Q2_REPRESENTATION
- Negative control: none
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Whether a nonlinear model-internal preference representation can exploit the visible-field signal that the fixed linear probe misses.
- Whether realistic strict-transfer effects are closer to 0.005, 0.01, or 0.02 and therefore below some model-specific MDEs.

Scope limitations:

- KuaiSearch internal dev only.
- Single frozen training seed and no forward-temporal claim.
- M1 behavior, heterogeneous M2 representation, and the adverse Q2 surface-balance control do not resolve H0 until the registered mediation chain is complete.

#### H1 — weakened

Statement: query-aware history selection 失败，相关历史被噪声淹没

Component statuses:

- `behavioral_routing`: `weakened`
- `internal_attention_routing`: `unresolved`
- `abstention_or_null_gate`: `unresolved`

Triangulation roles:

- Reversible behavior: E_M1_INPUT_INTERVENTIONS
- Independent source: none
- Negative control: none
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Whether a learned router with an explicit abstention path could outperform the frozen relevance heuristic without changing the evaluation contract.
- Whether the endpoint-divergent behavioral effects are mediated by attention edges, positional exposure, residual transport, or downstream readout nonlinearities.

Scope limitations:

- M0 is a separate linear recoverability control and M1 is an input intervention; neither is an internal Qwen routing attribution.
- No attention-logit, attention-weight, head-routing, or query-key causal intervention is complete.

#### H2 — weakened

Statement: 模型依赖 item/text recurrence，没有形成不同-ID偏好抽象

Component statuses:

- `usable_different_id_behavior`: `weakened`
- `localized_q2_attribute_preference_decodability`: `supported`
- `category_brand_semantic_invariance`: `weakened`

Triangulation roles:

- Reversible behavior: E_M1_INPUT_INTERVENTIONS
- Independent source: E_M2_Q2_REPRESENTATION<br>E_M2_Q3_REPRESENTATION
- Negative control: E_M2_Q2_PATCH_MEDIATION
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Whether the localized Q2 proxy signal generalizes beyond brand/category and survives a seed or architecture-anchor confirmation.
- Whether an explicitly factorized, ID-free preference bottleneck can make Q2's localized proxy signal request-specific and behaviorally usable rather than merely decodable.

Scope limitations:

- Registered deep representation anchors are Q2 and Q3 only.
- Brand and category are partial preference proxies and do not exhaust attributes, price, style, or intent.

#### H3 — weakened

Statement: 偏好已进入表示，但未被候选比较或最终 readout 使用

Component statuses:

- `localized_q2_preference_representation`: `supported`
- `request_specific_midlayer_preference_mediation`: `unresolved`
- `final_history_state_readout_use`: `supported`
- `beneficial_final_readout_use`: `rejected`

Triangulation roles:

- Reversible behavior: E_M2_Q2_PATCH_MEDIATION
- Independent source: E_M2_Q2_REPRESENTATION<br>E_M2_Q3_REPRESENTATION
- Negative control: E_M2_Q3_PATCH_MEDIATION
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Whether the Q2 preference-decoding component, rather than unrelated history/position state change, causally affects target-versus-competitor margin.
- Which attention, MLP, residual, or final-normalization operation converts the block-13 opposite-to-harm response into the harmful block-27 readout state.

Scope limitations:

- Post-block patching does not isolate attention output, MLP, residual, or RMSNorm branches.
- Only blocks 13 and 27 and the candidate readout position are preregistered for causal patching.

#### H4 — weakened

Statement: 训练目标被容易的 recurrence 样本或非个性化相关性捷径主导

Component statuses:

- `surface_gradient_allocation`: `unresolved`
- `observed_final_surface_gradient_conflict`: `supported`
- `simple_surface_balancing_remedy`: `rejected`

Triangulation roles:

- Reversible behavior: none
- Independent source: E_M3_Q2_MATCHED_CONTROL_RESULTS
- Negative control: E_M3_Q2_GRADIENT_RESULTS<br>E_M3_Q3_GRADIENT_RESULTS
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Whether the observed squared raw-gradient-mass and cosine patterns survive clipping, AdamW moments, weight decay, scheduler scaling, and full effective-update accounting.
- Whether a mechanism-specific loss or optimizer intervention can resolve the observed other-overlap conflict without reproducing the adverse margin effect of complete surface balancing.

Scope limitations:

- Gradient diagnostics cover registered parameter subsets rather than every model tensor and execute no optimizer step.
- The Q2 surface-balanced run is diagnostic-only and cannot be named as the method.

#### H5 — unresolved

Statement: 观察主要来自 seed、人口漂移或测量不稳定，而非共同机制

Component statuses:

- `query_fold_stability`: `unresolved`
- `training_seed_stability`: `unresolved`
- `population_stability`: `unresolved`

Triangulation roles:

- Reversible behavior: none
- Independent source: E_M2_Q3_REPRESENTATION<br>E_M3_Q2_MATCHED_CONTROL_RESULTS
- Negative control: E_M2_Q2_PATCH_MEDIATION<br>E_M2_Q3_PATCH_MEDIATION
- Two-fold direction consistent: false
- Major-claim gate met: false

Remaining uncertainty:

- Training-seed variability is outside the current bootstrap and has not been estimated.
- The retrospective first-round cohorts are descriptive only and cannot be reused for probe selection.

Scope limitations:

- No second training seed is authorized or run for this first diagnosis.
- The held-out evidence boundary remains closed and no new dataset is introduced.


## Valid-result contradictions

- `C_M0_FULL_NULL_ENDPOINT_DIVERGENCE`: For full versus null, strict-transfer NDCG is weakly positive (+0.003122), its interval crosses zero, and folds disagree, whereas target-versus-best-competitor margin is reliably negative (-0.039827) with both folds negative. Interpretation: The probe can alter within-slate ordering without moving the registered target margin in the same direction; neither endpoint may be silently preferred, and the cell does not establish a positive recoverability ceiling.
- `C_M0_CONTROL_ENDPOINT_DIVERGENCE`: Full beats the frozen wrong-history assignment control on NDCG (+0.012261, both folds positive), while its margin estimate is -0.001398 with an interval crossing zero and opposite folds. Full versus routing-query shuffle has an NDCG estimate of +0.000593 with an interval crossing zero and opposite folds, but a corrected positive margin of +0.002442 with both folds positive. Interpretation: History provenance/content and query-conditioned routing leave different signatures on rank position and score margin; this is a valid unresolved endpoint split, not evidence for selecting one control or metric after observing outcomes. The wrong-history assignment is not a within-history order intervention or a provenance-matched causal user-specificity control.
- `C_M1_RELEVANCE_ENDPOINT_DIVERGENCE`: Relevant-only versus irrelevant-only history raises strict-transfer NDCG for Q0 (+0.006752) and Q2 (+0.007918) with both folds positive but neither survives 48-endpoint correction, while Q0 target margin reliably falls by -0.009600 (q=0.006399). Interpretation: Focused relevant history can move some candidates' rank positions without coherently improving the target-versus-best-competitor score separation; small length differences remain, so this cannot be elevated to either a routing success or a pure semantic-routing failure.
- `C_M1_SEMANTIC_PRESERVATION_REVERSAL`: Semantic-preserving different-ID history produces no FDR-controlled positive endpoint over semantic breaking or frozen full history. Preserving-versus-breaking NDCG point estimates are positive with both folds positive for Q0, Q1, and Q2, while target-margin estimates are negative with both folds negative in all four models; only the Q0 and Q3 margin endpoints reject. Q1 preserving-versus-full NDCG is also corrected negative. Interpretation: The category-path/brand-prioritized intervention does not establish a corrected behavioral advantage, but uncorrected positive cells mean it also does not establish a zero or universally negative effect. Category-only fallback, replacement of other visible donor-event fields, and residual token differences prevent a pure semantic-invariance claim; M2 is still needed to distinguish H2 representation failure from H3 readout failure.
- `C_M1_ISOLATED_ORDER_EFFECT`: Q1 order shuffle versus frozen full has a positive corrected target-margin effect of only +0.000151 (q=0.033593), while its NDCG estimate is -0.000570 with an interval crossing zero, q=1.000000, and opposite folds; no corresponding corrected order effect appears for Q0, Q2, or Q3. Interpretation: Because order shuffle is exactly total-token-count matched, the corrected margin cell is not explained by total token count. Token sequence, position, and which content remains exposed under Q1 clipping can still change, and the tiny single-model, single-endpoint result is insufficient for a general order-sensitive mechanism claim.
- `C_M2_CROSS_MODEL_REPRESENTATION_HETEROGENEITY`: Q2 has a localized strict-transfer preference-decoding signal beyond random labels and a positive full-minus-null category contrast at state 28 in both folds, while Q3 does not reproduce it: its corresponding full-minus-null category contrast is negative with opposite folds. Both models nevertheless show nonzero full-minus-null candidate-readout state distance. Interpretation: The Q2 result argues against a universal absence of different-ID attribute representation, but Q3 prevents a model-general claim. State distance alone is not preference content or causal readout use; the registered patch controls remain necessary.
- `C_M2_CROSS_LAYER_MARGIN_SIGN_REVERSAL`: Full-minus-null strict-transfer target margin is negative for Q2 (-0.029227) and Q3 (-0.014785). Correct same-request block-13 restoration instead moves margin above null (+0.004415 and +0.002849), while block 27 reproduces the negative full response with mediated fraction approximately 1 in both models. Interpretation: The full-history state is not uniformly helpful or unused across depth: by the final readout it causally carries a harmful margin response, whereas the block-13 mixed state moves in the opposite direction. This is a layerwise transformation/calibration finding, not attribution to attention, MLP, residual, normalization, or a preference subspace.
- `C_M2_DONOR_SPECIFICITY_AND_DEPTH_HETEROGENEITY`: At block 13, Q2 cross-request patching has a larger opposite-to-harm strict-transfer ratio (-0.641326, interval below zero) than same-request patching (-0.151072, interval crosses zero), whereas Q3 cross-request is near zero (-0.042390, interval crosses zero) while same-request is -0.192679 with an interval below zero. At block 27, cross-request ratios amplify harm to 4.395712 for Q2 and 4.023133 for Q3. Interpretation: Q2 block-13 movement is not request-specific under the registered donor control, while Q3 has a descriptive same-versus-cross separation but no stable preference decoder or registered inference for that difference. The large block-27 cross effects show donor identity matters at the final state but also forbid treating any nonzero patch as preference mediation.
- `C_M3_ANCHOR_SCOPE_DEPENDENT_RECURRENCE_AND_SHUFFLE`: At final checkpoints Q3 allocates 55.69% of observed squared raw-gradient mass to recurrence, whereas Q2 allocates only 22.55% and instead allocates 40.62% to strict transfer; within-request label shuffle raises recurrence mass further in both anchors (41.80% and 64.75%) and leaves all final surface cosines nonnegative. Interpretation: Recurrence-mass dominance is model/registered-scope dependent rather than universal, and its amplification under label shuffle means mass alone is not a clean label-sensitive shortcut signal. The shared final conflict with other-overlap gradients remains a narrower point diagnostic, not an isolated anchor effect, sampling-inference result, or optimizer-update attribution.
- `C_M3_MATCHED_CONTROL_ENDPOINT_REVERSAL`: For Q2 surface-balanced versus original-mixture training, the preregistered strict-transfer full-minus-null NDCG DID is +0.004324 but its interval crosses zero, q=0.421516, and folds disagree; the target-margin DID is -0.081230 with a wholly negative interval, q=0.000800, and both folds negative. Interpretation: Surface balance does not provide a credible ranking-response improvement and coherently worsens score separation. The small positive NDCG point estimate must be retained, but it cannot rescue the proposed balance control or be selected over the adverse margin endpoint.

## Mechanical non-results

- `NR_M0_SUPERSEDED_INITIAL_ANALYSES`: The four initial M0 recoverability analyses were superseded by the frozen v2 synthesis inputs and are retained only for provenance. Reason: They are outside statistics.input_analyses and therefore cannot enter the mechanism evidence set; this row retains exactly the four frozen superseded M0 run roots.
- `NR_M2_OFFSET_UNIQUENESS_MECHANICAL_FAILURE`: Four first-attempt M2 activation bundles stopped at an atomic shard boundary when Qwen byte-fallback produced two or three tokenizer subtokens sharing one Unicode terminal-character offset. Reason: The implementation incorrectly required a unique covering token even though the frozen position contract specifies the last covering subtoken. This row retains exactly the four failed run roots and four bound failure logs; the partial bundles use the superseded implementation digest, were never resumed, cannot overlap a protected formal run root, and cannot enter representation or patch evidence.
- `NR_M2_REPRESENTATION_RAW_QUERY_HASH_MECHANICAL_FAILURE`: The first Q2 representation-evaluation attempt stopped during the pre-qrels request-manifest audit because three raw standardized queries contain surrounding whitespace. Reason: The first evaluator implementation compared the manifest hash of the raw standardized query with the hash of the prompt-sanitized query after whitespace stripping. No pre-qrels completion report, qrels access, metric, or eligible result was produced; the retained failure record binds the empty output, traceback, three exact raw-query mismatches, repaired evaluator identities, and regression tests.
- `NR_M3_GRADIENT_SMOKE_RUNS`: Four capped gradient smoke runs exercised mechanics for Q2/Q3 base/final states. Reason: Smoke coverage is an engineering diagnostic and does not satisfy the registered request, surface, endpoint, or completion grid.
- `NR_M2_ACTIVATION_CPU_SMOKE_RUNS`: Seven capped CPU activation smoke runs exercised Q2/Q3 extraction, position indexing, shard finalization, and a repaired mechanical-failure path. Reason: These request-capped engineering runs do not satisfy the registered model, request-count, shard, condition, or finite-coverage grid and are excluded from representation evidence.
- `NR_M1_Q2_RELEVANT_SCORE_SMOKE`: A capped Q2 relevant-history score smoke exercised the M1 scoring path. Reason: It is not the registered complete score bundle and cannot contribute an intervention effect.
- `NR_M3_Q2_MATCHED_CONTROL_MOCKS`: The Q2 original-mixture and surface-balanced CPU mocks exercised selection, resume, and learning-curve mechanics. Reason: Mock optimization is not a numerical training result and is excluded from the matched diagnostic comparison.
- `NR_M3_MATCHED_CHECKPOINT_DIR_CONTRACT_FAILURE`: The first post-training matched-control supervisor pass failed before scoring because it expected output_root/checkpoint while the shared Qwen ranker writes output_root/checkpoint_latest; its dependent DID waiter stopped on the upstream failed status. Reason: Both training arms had already completed the exact 256-step, 4096-exposure recipe with no resume. No scientific outcome or score was read. The supervisor was repaired to consume the shared checkpoint-directory contract, the completed training artifacts were reused without retraining, and the failure record binds the repair and regression tests; this event is pipeline provenance, not a transfer result.

## Architecture opportunity matrix (design only)

| ID | Bottlenecks | Priority | Requirement | Falsifiable predictions | Status |
|---|---|---|---|---|---|
| OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER | H1 | boundary_only | A router must score query-history relevance before candidate scoring, expose selected-event mass, include an explicit null route, and pass only a bounded set of events into downstream preference formation. | Relevant-only history must outperform irrelevant-only history and frozen full history on both strict-transfer endpoints in both query folds.<br>Removing router-selected events must reverse the gain, while removing equal-count unselected events must not.<br>Null-route mass must rise on no-history or low-relevance requests without degrading recurrence indiscriminately. | not_started_not_authorized |
| OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK | H2 | secondary_candidate | A bounded preference bottleneck must factorize train-visible history into auditable brand, category, attribute, style, price-band, and query-intent factors without serializing raw item IDs, then expose those factors to candidate scoring. | Real-label factors must be decodable above random-label and embedding-state controls across both query folds.<br>Semantic-preserving different-ID swaps must preserve or improve strict-transfer response relative to semantic-breaking swaps.<br>Destroying the bottleneck factors while preserving token count must remove the transfer response without creating recurrence gains. | not_started_not_authorized |
| OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL | H3 | secondary_candidate | Each candidate score must decompose into a query-only relevance term plus a gated signed preference-to-candidate residual whose common offset cancels and whose contribution can be patched or zeroed independently. | A same-request preference-state patch must move target-versus-best-competitor margin in the predicted direction beyond identity and cross-request controls.<br>Zeroing the residual must recover the query-only score ordering, while adding a common residual offset must not change ranks.<br>Candidate-slate permutation followed by inverse permutation must preserve per-candidate residuals. | not_started_not_authorized |
| OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH | H2<br>H3 | primary_candidate | A query-conditioned ID-free factor state must feed a separate signed candidate-residual scorer alongside a query-only relevance backbone; factor slots, residual contributions, and the abstention gate must each be independently zeroable and patchable. | Real factor slots must exceed random-label and state-0 controls in both folds, while factor-preserving history replacement retains slot content without relying on item identity.<br>Restoring the correct same-request factor state must recover target-versus-competitor margin beyond identity and cross-request patches; a wrong factor must move residuals according to candidate compatibility rather than adding a common offset.<br>Zeroing the preference path must exactly expose the query-only ordering, and recurrence gains must not increase when candidate IDs are excluded from the factor encoder.<br>If factor decodability rises but signed residual mediation remains absent, the joint design fails and H3 rather than H2 remains the active bottleneck. | not_started_not_authorized |
| OP_H4_SURFACE_AWARE_GRADIENT_BUDGET | H4 | deprioritized | The training system must expose per-surface losses and gradient directions, allocate a preregistered update budget across surfaces, and preserve the same backbone, optimizer steps, visible fields, and evaluator. | Observed recurrence normalized squared raw-gradient-mass proxy must be disproportionate or conflict with strict-transfer gradients beyond label-shuffle controls before H4 is supported; effective-update attribution additionally requires a restored-state optimizer counterfactual.<br>The completed fixed-exposure balance control fails to establish a strict-transfer NDCG improvement and significantly worsens target margin, so complete surface balancing is falsified as the immediate remedy; any future conflict-specific method must beat this negative control without endpoint reversal.<br>If optimizer-aware one-step replay removes the apparent imbalance, a raw-gradient-only explanation must be rejected. | not_started_not_authorized |

### Opportunity details

#### OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER

Innovation target: Make history use query-conditioned, sparse, auditable, and able to abstain when no history event is useful.

Necessary modules:

- Query-to-history event scorer with sparse top-k or differentiable sparse gating.
- Explicit null or abstention token whose mass is logged per request.
- Routing audit head that reports selected event IDs only for integrity analysis, not as model features.

Train-only data requirements:

- Query-relevant and query-irrelevant history pairs built only from train-visible events.
- Equal-count and approximately token-length-matched relevant and irrelevant histories; frozen null is marker-only and must be reported separately unless a fixed-slot masked-null control is preregistered; any future wrong-user donor must exclude global fallback and report usable coverage.
- Different-ID positives and candidate-excluded donors to prevent exact recurrence leakage.

Training signals:

- Listwise ranking loss on the unchanged candidate slate.
- Counterfactual consistency that rewards retaining useful relevant history and abstaining on irrelevant history.
- Sparsity or entropy control registered before outcomes, with no dev-selected top-k.

Key ablations:

- Dense attention versus sparse routing at matched parameter count.
- Router with versus without explicit null route.
- Query-conditioned versus query-shuffled routing with identical visible tokens.
- Selected-event identity replacement and equal-count random-event controls.

Prior-work differentiation:

- CoPPS: shared ground — Both address noisy histories and use query-related sequence information. Substantive difference — CoPPS centers sequence-view contrast and invariance; this opportunity requires an explicit sparse per-query router with a measurable null route and same-request recurrence-transfer causal tests. Source: https://doi.org/10.1145/3580305.3599287
- BATA: shared ground — Both use query-history relations to locate relevant past behavior. Substantive difference — BATA injects external query/item relations as dense attention bias and auxiliary tasks; this opportunity isolates an inspectable sparse selection decision and abstention path under the current fixed evaluator. Source: https://doi.org/10.1145/3726864
- HMPPS: shared ground — Both filter history using the current query and seek robustness to irrelevant behavior. Substantive difference — HMPPS uses first-stage filtering for an MLLM reranker; this opportunity puts the router inside the same ranker and requires reversible selection and null-route attribution on candidate-disjoint transfer. Source: https://arxiv.org/abs/2509.18682
- MemRerank: shared ground — Both aim to suppress raw-history noise before reranking. Substantive difference — MemRerank learns a downstream-reward preference memory; this opportunity targets per-query sparse event routing with explicit abstention before preference compression. Source: https://arxiv.org/abs/2603.29247
#### OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK

Innovation target: Force the history pathway to express transferable preference factors rather than relying on exact item or surface recurrence.

Necessary modules:

- ID-free history encoder over the existing visible-field whitelist.
- Factorized preference slots or prototypes with query-conditioned mixture weights.
- Preference reconstruction and different-ID semantic-invariance heads used only during training.

Train-only data requirements:

- Attribute-preserving different-ID counterfactual histories built from train-visible donors.
- Attribute-breaking, label-shuffled, and cross-user controls with candidate-item exclusions.
- Train-only factor labels derived from existing brand/category/text fields, with missingness audited.

Training signals:

- Contrastive invariance for semantic-preserving different-ID replacements.
- Factor prediction or reconstruction from history-summary states with random-label controls.
- Candidate ranking loss that requires the factor state to distinguish query-matched hard negatives.

Key ablations:

- Free-form history state versus fixed factorized slots.
- Brand/category-only versus expanded attribute and intent factors.
- Different-ID invariance loss versus ranking loss alone.
- Factor-label shuffle, item-ID exposure, and slot permutation controls.

Prior-work differentiation:

- CoPPS: shared ground — Both use semantically similar different-ID history views to reduce brittle sequence representations. Substantive difference — CoPPS enforces representation invariance through sequence contrast; this opportunity requires an explicit factorized preference bottleneck whose factors are decoded, intervened on, and linked to candidate-disjoint ranking. Source: https://doi.org/10.1145/3580305.3599287
- BATA: shared ground — Both exploit brand/category and query relations as cross-item structure. Substantive difference — BATA supplies external relations as attention bias and auxiliary reconstruction; this opportunity makes the transferable factor state an explicit internal bottleneck rather than a dense relation prior. Source: https://doi.org/10.1145/3726864
- HMPPS: shared ground — Both rely on content semantics to handle low-frequency or unseen products. Substantive difference — HMPPS compresses descriptions and filters histories; this opportunity preregisters auditable preference factors and semantic-preserving counterfactuals within the ranking model. Source: https://arxiv.org/abs/2509.18682
- MemRerank: shared ground — Both compress raw interaction history into a preference representation. Substantive difference — MemRerank learns a reward-driven preference memory; this opportunity uses explicit structured factors with causal slot ablations and a fixed candidate-disjoint transfer contract. Source: https://arxiv.org/abs/2603.29247
#### OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL

Innovation target: Create an explicit, auditable score path that maps a preference state to relative candidate advantages rather than leaving history use implicit in the language-model readout.

Necessary modules:

- Candidate encoder or candidate readout projection shared across the slate.
- Signed bilinear or cross-attention matcher between preference state and each candidate.
- Per-request gate and explicit residual score output logged before final ranking.

Train-only data requirements:

- Same-query candidate pairs differing in preference-compatible attributes.
- Different-ID positive candidates and query-matched hard negatives from train-only records.
- Identity, cross-request, slate-permutation, and query-only controls.

Training signals:

- Pairwise signed residual loss in addition to the unchanged relevance ranking loss.
- Counterfactual preference-state swaps that reverse only the residual term when candidate compatibility reverses.
- Residual sparsity or calibration constraint preventing a common history-dependent score offset.

Key ablations:

- Implicit LM readout versus explicit signed residual path.
- Candidate-independent preference bias versus candidate-conditioned matching.
- Bilinear matcher versus cross-attention matcher at matched capacity.
- Residual gate, common-offset, identity-patch, and cross-request controls.

Prior-work differentiation:

- CoPPS: shared ground — Both connect a history-derived user representation to product ranking. Substantive difference — CoPPS optimizes sequence representations and downstream ranking; this opportunity exposes an additive signed candidate-specific residual whose mediation can be directly patched and falsified. Source: https://doi.org/10.1145/3580305.3599287
- BATA: shared ground — Both model fine-grained relations among query, history, and products. Substantive difference — BATA uses dense relation-biased Transformer interactions and auxiliary tasks; this opportunity isolates the history contribution as a separate relative-score path with a query-only counterfactual. Source: https://doi.org/10.1145/3726864
- HMPPS: shared ground — Both condition product reranking on filtered user history and candidate content. Substantive difference — HMPPS applies a pointwise MLLM reranker after first-stage filtering; this opportunity explicitly compares all candidates through a shared signed preference residual inside one slate. Source: https://arxiv.org/abs/2509.18682
- MemRerank: shared ground — Both train history-derived information for downstream reranking utility. Substantive difference — MemRerank focuses on reward-trained preference memory; this opportunity separates query-only relevance from a patchable candidate-conditioned residual and tests mediation under fixed requests. Source: https://arxiv.org/abs/2603.29247
#### OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH

Innovation target: Join transferable preference formation and candidate-specific use in one auditable path, so a history factor cannot count as learned unless it changes relative candidate scores under different-ID counterfactuals.

Necessary modules:

- Query-conditioned factor-slot encoder over visible history attributes with an explicit empty/abstention slot.
- Shared candidate attribute projection aligned to the same factor basis.
- Signed factor-candidate interaction head whose output is added to, and logged separately from, query-only relevance.
- Causal audit interface for slot zeroing, slot permutation, same-request restoration, and cross-request transplantation.

Train-only data requirements:

- Train-only same-query different-ID positive/negative candidate pairs with candidate IDs excluded from donor histories.
- Attribute-preserving and attribute-breaking history counterfactuals that keep event count, token budget, and candidate slate fixed as closely as possible.
- Missing-factor, wrong-user, label-shuffle, and cross-request donors retained as negative controls rather than discarded examples.

Training signals:

- Native listwise or pairwise relevance loss on the unchanged slate.
- Factor-consistency loss across semantic-preserving different-ID histories and factor-separation loss for semantic-breaking controls.
- Signed residual ranking loss requiring compatible candidates to gain relative to query-matched incompatible candidates, with a zero-mean/common-offset constraint across the slate.
- Gate calibration that selects the query-only path when no reliable preference factor is present.

Key ablations:

- Free-form history state versus explicit factor slots at matched parameter count.
- Implicit LM score versus decomposed query-only plus signed preference residual.
- Factor loss only, residual loss only, and their joint objective.
- Explicit abstention versus always-on preference gate.
- Same-request factor restoration, cross-request factor swap, slot permutation, item-ID exposure, and common-offset controls.

Prior-work differentiation:

- CoPPS: shared ground — Both seek different-ID invariance from history-derived representations. Substantive difference — CoPPS regularizes sequence views; this opportunity couples explicit factor slots to a separately measurable signed candidate residual and requires both representation and mediation tests. Source: https://doi.org/10.1145/3580305.3599287
- BATA: shared ground — Both exploit brand/category/query relations and auxiliary supervision. Substantive difference — BATA injects relation bias into dense Transformer interactions; this opportunity enforces an ID-free bottleneck plus a decomposed relative-score path with exact zeroing and patch controls. Source: https://doi.org/10.1145/3726864
- HMPPS: shared ground — Both use query-filtered history and content-aware hard cases to support unseen products. Substantive difference — HMPPS combines filtering, description compression, and an MLLM reranker; this opportunity tests a single internal factor-to-residual causal path without changing the candidate population. Source: https://arxiv.org/abs/2509.18682
- MemRerank: shared ground — Both compress history into a downstream-useful preference memory. Substantive difference — MemRerank optimizes a reward-trained memory; this opportunity structurally decomposes preference factors and candidate-relative residuals so each stage can be independently falsified under strict transfer. Source: https://arxiv.org/abs/2603.29247
#### OP_H4_SURFACE_AWARE_GRADIENT_BUDGET

Innovation target: Prevent easy recurrence or nonpersonalized relevance gradients from consuming the update budget needed for candidate-disjoint preference transfer.

Necessary modules:

- Surface-stratified train-only sampler with fixed request-group exposure.
- Per-surface loss heads or gradient accounting before optimizer aggregation.
- Gradient normalization or conflict-handling layer whose behavior is logged and ablated.

Train-only data requirements:

- Frozen recurrence, strict-transfer, and other-overlap surface labels computed from train-only qrels gains plus candidate/history item-ID structure.
- Query-matched hard negatives and different-ID preference pairs without changing candidate evaluation slates.
- Original-mixture and surface-balanced selections with identical total optimizer steps.

Training signals:

- Native ranking objectives decomposed by surface and Q2 RankNet/ListNet component.
- Preregistered normalized gradient-budget constraint or surface-balanced aggregation.
- Within-request label shuffle and restored-(theta, moments, variance) one-step optimizer counterfactual controls.

Key ablations:

- Original mixture versus surface-balanced exposure at exactly 256 steps.
- Sampler balance versus loss weighting versus gradient projection.
- Normalized squared raw-gradient-mass proxy versus a restored-state combined one-step update under the identical microbatch multiset.
- Observed labels versus within-request label shuffle and recurrence-removed controls.

Prior-work differentiation:

- CoPPS: shared ground — Both use train-time interventions to make history representations less brittle. Substantive difference — CoPPS uses contrastive sequence augmentation; this opportunity explicitly budgets and audits optimizer updates across recurrence and candidate-disjoint transfer surfaces. Source: https://doi.org/10.1145/3580305.3599287
- BATA: shared ground — Both add training signals beyond a single undifferentiated ranking objective. Substantive difference — BATA adds relation biases and auxiliary reconstruction; this opportunity targets measured surface-gradient allocation under fixed exposure and optimizer dynamics. Source: https://doi.org/10.1145/3726864
- HMPPS: shared ground — Both use hard negatives and history-aware training to address difficult or unseen products. Substantive difference — HMPPS combines first-stage filtering and hard negatives; this opportunity isolates recurrence-versus-transfer update competition without altering the evaluation population or reranker boundary. Source: https://arxiv.org/abs/2509.18682
- MemRerank: shared ground — Both let downstream reranking utility shape how history information is learned. Substantive difference — MemRerank trains a preference-memory extractor with downstream reward; this opportunity is an optimizer-accounted surface budget and remains a diagnostic design until separately authorized. Source: https://arxiv.org/abs/2603.29247

## Aggregate artifact registry

All numerical values in the machine report are copied from these producer aggregates without statistical recomputation. Human result prose passed the producer-token audit for 259 rendered numeric claims.

| Artifact ID | Stage | Run ID | SHA-256 | Path |
|---|---|---|---|---|
| m0.data_power_audit | m0 | 20260717_kuaisearch_mech_m0_data_power | `d49ac8d331b1908ea8a73473fc3172170a9405b48fb05c900b0ca5cdfb39d3de` | `runs/20260717_kuaisearch_mech_m0_data_power/data_power_audit.json` |
| m0.recoverability_statistics | m0 | 20260717_kuaisearch_mech_m0_statistical_synthesis | `b364d5b79d8cb88fb25897e7a0a8b1bbca0a105350b021ace99c9e24b7db7442` | `runs/20260717_kuaisearch_mech_m0_statistical_synthesis/statistics.json` |
| m1.intervention_statistics | m1 | 20260717_kuaisearch_mech_m1_statistical_synthesis | `f207eabc2fd1e5df3461ea00c39c81454a6b6b2229529334f442456cc5acf77a` | `runs/20260717_kuaisearch_mech_m1_statistical_synthesis/statistics.json` |
| m1.token_length_summary | m1 | 20260717_kuaisearch_mech_m1_token_audit | `1eb76aa333717886bfcb173e321761fd33ff087b2e9ba528e77c7b1047bc0188` | `runs/20260717_kuaisearch_mech_m1_token_audit/summary.json` |
| m2.q2.patch_metrics | m2 | 20260717_kuaisearch_mech_m2_q2_patch_eval | `be2c52164165cc7acca01fd375b2c1c7d87cb2fbafef832f4d0e4ee0260ec067` | `runs/20260717_kuaisearch_mech_m2_q2_patch_eval/metrics.json` |
| m2.q2.representation_metrics | m2 | 20260717_kuaisearch_mech_m2_q2_representation_eval | `d43ceef3de1627184ff5ec7a7dde957a09b4ac7e7df5d30c19eae52728a46b69` | `runs/20260717_kuaisearch_mech_m2_q2_representation_eval/metrics.json` |
| m2.q3.patch_metrics | m2 | 20260717_kuaisearch_mech_m2_q3_patch_eval | `f00f3efb71878357476f7ce87d49623aa4fb8f68603cef547c28764059f51833` | `runs/20260717_kuaisearch_mech_m2_q3_patch_eval/metrics.json` |
| m2.q3.representation_metrics | m2 | 20260717_kuaisearch_mech_m2_q3_representation_eval | `96f4618ee29c529399c3ab912cbc40a32f1b4346b984049211f8f6328070cc3c` | `runs/20260717_kuaisearch_mech_m2_q3_representation_eval/metrics.json` |
| m3.q2.base_gradient_diagnostics | m3 | 20260717_kuaisearch_mech_m3_q2_base_gradient | `f48127ac7c6e3cec9072d4802337890e13f49b837638d19621cac71cc78e32c8` | `runs/20260717_kuaisearch_mech_m3_q2_base_gradient/gradient_diagnostics.json` |
| m3.q2.final_gradient_diagnostics | m3 | 20260717_kuaisearch_mech_m3_q2_final_gradient | `ed51c6ebdfe100f9dd3a26136635f1656b035e10fc446719a85233b6840e9825` | `runs/20260717_kuaisearch_mech_m3_q2_final_gradient/gradient_diagnostics.json` |
| m3.q2.matched_balanced_pair_metrics | m3 | 20260717_kuaisearch_mech_m3_q2_matched_control_analysis_surface_balanced_full_vs_null | `344267f5ff206119e8b883a5039c64f4fb43c444bbd8e023ce962306e3d84d6c` | `runs/20260717_kuaisearch_mech_m3_q2_matched_control_analysis_surface_balanced_full_vs_null/metrics.json` |
| m3.q2.matched_control_metrics | m3 | 20260717_kuaisearch_mech_m3_q2_matched_control_analysis | `ce4aeea526dc918587612c65d64d187a4f7e4cb9e0e7ae26c2e89fc1bdabe401` | `runs/20260717_kuaisearch_mech_m3_q2_matched_control_analysis/metrics.json` |
| m3.q2.matched_did_statistics | m3 | 20260717_kuaisearch_mech_m3_q2_matched_did_analysis | `a42cfe245a1e74998bd2da10356881227c0cf18a740add56b482449bdbe67cc6` | `runs/20260717_kuaisearch_mech_m3_q2_matched_did_analysis/metrics.json` |
| m3.q2.matched_original_pair_metrics | m3 | 20260717_kuaisearch_mech_m3_q2_matched_control_analysis_original_mixture_full_vs_null | `0c12d7ab8e59c7359482db101fb39ae682764944d806c4c142825405bf89d577` | `runs/20260717_kuaisearch_mech_m3_q2_matched_control_analysis_original_mixture_full_vs_null/metrics.json` |
| m3.q2.original_mixture_learning_curve | m3 | 20260717_kuaisearch_mech_m3_q2_matched_original_train | `6b87939ed98534b9d2ee06c46a7892eb661fda4111eb9a90fff23e8ede9cd3ae` | `runs/20260717_kuaisearch_mech_m3_q2_matched_original_train/learning_curve.jsonl` |
| m3.q2.surface_balanced_learning_curve | m3 | 20260717_kuaisearch_mech_m3_q2_matched_balanced_train | `ac564e82af61544c17e6004ed9080964857c35d8385237aedb815e4a363e92fb` | `runs/20260717_kuaisearch_mech_m3_q2_matched_balanced_train/learning_curve.jsonl` |
| m3.q3.base_gradient_diagnostics | m3 | 20260717_kuaisearch_mech_m3_q3_base_gradient | `c2d7cf51f272bc2a70e16a9cea0005e030082479a502e7b0895dd216f96c1d7e` | `runs/20260717_kuaisearch_mech_m3_q3_base_gradient/gradient_diagnostics.json` |
| m3.q3.final_gradient_diagnostics | m3 | 20260717_kuaisearch_mech_m3_q3_final_gradient | `b778487546f822d4f3c6e9c4302440558ff812b5a25382d7e1b80d57d00ef3b6` | `runs/20260717_kuaisearch_mech_m3_q3_final_gradient/gradient_diagnostics.json` |

## Stop point

The first mechanism diagnosis is complete. The held-out evidence boundary remains closed; no new dataset or transfer architecture was opened or implemented; the matched training control remains diagnostic rather than a paper method. Further method work requires new user direction.

Forbidden without new direction: `implement_transfer_architecture`, `switch_dataset`, `open_source_test`, `present_diagnostic_training_control_as_paper_method`.
