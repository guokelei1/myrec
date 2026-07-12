# C01 Nearest-Neighbor Audit

Search date: 2026-07-11.  Audit completed before any C01 GPU/model outcome.
Sources were original papers and, when available, author-linked official code.
No sibling prompt, design, code, run, or note was read.

## Verdict

The locked operator is **not reducible** to the neighbors below by variable
renaming.  Its closest neighbors are CFT (paired history/no-history causal loss)
and CARD (leave-one-event-out prediction-error attention).  C01 was tightened
after finding them: its defining operator is instead a shared Transformer's
event energy calibrated against the robust maximum of four counterfactual twin
families, followed by a train-only null quantile that yields a one-pass inference
certificate.  This distinction is falsifiable by the registered CFT/CARD
degenerations and the matched plain-attention control.  Because a literature
search cannot prove global novelty, the pre-outcome paper-level verdict is
`distinct-with-uncertainty`, not “globally novel.”

## Mechanism comparison

| Neighbor | Existing operator | Non-reducible C01 difference | Matched degeneration/ablation |
|---|---|---|---|
| DIN (KDD 2018) | Candidate local-activation weights over behavior embeddings; no counterfactual null. | C01 contextualizes query, candidate, and ordered events jointly, then admits an event only relative to a calibrated multi-twin null. | Remove event-event/query contextualization and CQC; use independent target activation. |
| SIM / UBR family | Candidate-conditioned retrieval followed by exact interest modeling for very long histories. | C01 does not retrieve a top history subset and does not equate similarity with evidence; admission is a false-admission-controlled counterfactual certificate. | Select top events by candidate similarity and remove `Q_cf`/twins. |
| HEM | Static hierarchical query/user/item embeddings and a fixed query-user mixture. | C01 has no static user vector; its candidate-specific event states and rejection threshold are request-local. | Mean-pool history into a user vector and remove event certificates. |
| ZAM (CIKM 2019) | Query attention can allocate mass to a zero vector to reduce personalization. | ZAM's null competes inside ordinary attention but is not calibrated from intervention twins and has no event-level false-admission contract. | Add a learned zero event and softmax all events; remove twin losses/quantile. |
| TEM (SIGIR 2020) | Transformer over query and purchase history, scoring from a contextual query/sequence state. | C01's score is constrained by event-level calibrated admission and protected recurrence, rather than unconstrained contextual attention. | Remove CQC and read the Transformer query state directly. |
| RTM (SIGIR 2021) | Transformer review-level user/item matching with dynamic contextual review effects. | RTM dynamically matches reviews but supplies no counterfactual event null or inference certificate. | Treat event states as review states and score directly without calibration. |
| SASRec / BERT4Rec | Causal or bidirectional self-attention over item-ID sequences with next-item/Cloze objectives. | C01 is query-and-candidate conditioned per fixed candidate and uses multi-twin evidence admission, not an unconditional sequence representation. | Remove query/relation tokens and CQC; train next-item/plain rank loss. |
| LLaRA / CoLLM | Project collaborative/ID embeddings into an LLM input space and tune a recommendation objective. | C01's contribution is neither hybrid prompting nor an external CF embedding; it changes the internal event evidence contract. | Concatenate projected history/ID states and remove event certificates. |
| iLoRA / CoRA | History- or collaborative-conditioned low-rank LLM weight adaptation. | C01 never generates request-specific `Delta W`; all conditioning stays in token states and the shared encoder. | Not an allowed C01 pivot; listed to prevent locus collision. |
| CauseRec / CASR / CaseRec | Synthesize or replace counterfactual sequences/exposures to augment sequential training; contrast observational and synthesized user representations. | C01 twins are negative contract interventions, not additional positive interaction sequences; the deliverable is an inference-time event admission certificate. | Train on synthesized sequences as extra positives and remove `Q_cf`. |
| CounterCLR | Counterfactual contrastive representation learning for non-random missing feedback. | Its construct is missingness/selection bias at representation level, not query-candidate-event fidelity or a per-event inference gate. | Use a global contrastive representation loss only. |
| CoDeR / CoDeR+ | Demand-shift graphs plus backdoor adjustment for sequential confounding. | C01 makes no causal-identification/backdoor claim and uses no demand graph; twins operationalize a falsifier, not identified causal effects. | Replace CQC with demand-shift state and adjustment. |
| CLLMR | Spectral LLM-side-information encoder plus counterfactual removal of LLM propensity bias. | C01 neither aligns spectral side information nor estimates LLM propensity; its null is event-evidence corruption. | Remove event contract and perform side-information debiasing. |
| CFT (2024) | Two LLM passes with/without the complete history; fit labels using their output-distribution difference; inference uses the normal pass. | C01 is event-level, uses four twin families and a calibrated null quantile, and never defines its primitive as a paired prefix/logit delta. | Replace event CQC with the shared model's history/no-history output difference. |
| CARD (ICASSP 2026) | Route low-stability sequences and weight events by leave-one-event-out future prediction-error reduction before diffusion guidance. | C01 has no sequence router, future-target PER, leave-one-out inference, or diffusion model; it is query/candidate conditioned and calibrates against wrong/shuffle/qmask/coarse nulls. | Replace `a_i,Q_cf` by CARD PER weights and remove multi-twin calibration. |
| UFRec (May 2026) | Use prediction uncertainty to weight multi-step future supervision; auxiliary tasks disappear at inference. | UFRec calibrates supervision horizon, not event evidence admission, and has no candidate-specific counterfactual certificate. | Weight an auxiliary future loss by base confidence; remove CQC. |
| Recommendation confidence/conformal work | Calibrate final recommendation confidence or prediction sets, typically post hoc. | C01 calibrates the *internal event admission null* inside the ranking path; the quantile controls counterfactual event false admission, not final-score coverage. | Calibrate only final candidate scores and leave history attention unchanged. |

## Primary sources checked

- DIN: https://arxiv.org/abs/1706.06978
- SIM: https://arxiv.org/abs/2006.05639
- HEM: https://doi.org/10.1145/3077136.3080813 and the repository's pinned
  official source.
- ZAM: https://arxiv.org/abs/1908.11322
- TEM: https://arxiv.org/abs/2005.08936 and the repository's pinned
  `kepingbi/ProdSearch` source.
- RTM: https://arxiv.org/abs/2004.09424
- SASRec: https://arxiv.org/abs/1808.09781
- BERT4Rec: https://arxiv.org/abs/1904.06690
- LLaRA: https://arxiv.org/abs/2312.02445
- CoLLM: https://arxiv.org/abs/2310.19488
- iLoRA: https://arxiv.org/abs/2408.10159
- CauseRec: https://arxiv.org/abs/2109.05261
- CASR: https://doi.org/10.1145/3404835.3462855
- CounterCLR: https://arxiv.org/abs/2402.05740
- CLLMR: https://arxiv.org/abs/2409.20052
- CFT and official code: https://arxiv.org/abs/2410.22809 and
  https://github.com/juntaoyou/CFT
- CaseRec and official code: https://arxiv.org/abs/2504.13482 and
  https://github.com/ZiqiZhao1/CaseRec
- CoDeR: https://doi.org/10.1609/aaai.v39i12.33379
- CoDeR+: https://doi.org/10.1145/3778863
- CARD and official code: https://arxiv.org/abs/2601.15673 and
  https://github.com/yanqilong3321/CARD
- UFRec: https://arxiv.org/abs/2605.28493
- DT4SR uncertainty modeling: https://arxiv.org/abs/2106.06165
- Confidence calibration for recommendation:
  https://arxiv.org/abs/2402.16325
- Conformal group recommendation: https://arxiv.org/abs/2307.12034

## Stop/pivot rule applied

The initial broad idea “use counterfactual differences to weight history” would
have been reducible to CFT/CARD and was not locked.  The proposal pivoted before
outcome to the multi-twin quantile contract above.  If implementation shows that
removing the quantile/multi-twin contract leaves behavior unchanged, C01 is
scientifically `reducible` and stops; no dev-driven rescue is permitted.
