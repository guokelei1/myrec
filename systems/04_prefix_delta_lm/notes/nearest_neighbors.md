# C04 nearest-neighbor and novelty audit

Search date: 2026-07-11. Scope: original papers and author/official code pages
available by this date. No C04 dev outcome was read before this audit.

## Operator-level audit

| Family | Primary source | Closest mechanism | Difference that must pay rent |
|---|---|---|---|
| DIN target attention | [DIN](https://arxiv.org/abs/1706.06978) | candidate attends to behavior features in one CTR pass | C04 uses the same LM under factual/null prefixes and a candidate-set logit contrast; `single_pass` degenerates toward DIN-like candidate conditioning |
| UBR / SIM long-history retrieval | [UBR4CTR](https://arxiv.org/abs/2005.14171), [SIM](https://arxiv.org/abs/2006.05639) | retrieve candidate-relevant behaviors, then rank | C04 neither retrieves nor drops events; corruption controls act on a paired conditional likelihood effect. Equal performance to concat/single-pass removes this distinction |
| ZAM | [ZAM](https://arxiv.org/abs/1908.11322) | query attention includes a zero vector to suppress personalization | C04's null is a complete counterfactual prefix scored by the same LM, not an attention sink. A single-pass/zero-vector tie makes the distinction non-contributory |
| TEM / RTM | [TEM](https://arxiv.org/abs/2005.08936), [RTM](https://arxiv.org/abs/2004.09424), [official family code](https://github.com/kepingbi/ProdSearch) | Transformer jointly contextualizes query/history/item or reviews | they use one factual representation; C04 identifies an order-changing effect against a shared null prefix. The matched single-pass control directly tests reducibility |
| SASRec / BERT4Rec | [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690) | Transformer sequence/Masked-LM next-item scoring | no query-conditioned paired history ablation or null-order anchor; C04 becomes this family if the null branch and tangent do not matter |
| P5 / GPTRec / generative recommendation | [P5](https://arxiv.org/abs/2203.13366), [GPTRec](https://arxiv.org/abs/2306.11114) | represent recommendation as LM generation | C04 is discriminative over the manifest candidates and cannot generate an item ID; free generation is outside its contract |
| Recformer | [Recformer](https://arxiv.org/abs/2305.13731) | text-only bidirectional Transformer encodes item sequences | C04 scores query/history/candidate triples under a paired null intervention; ordinary history concatenation is a frozen degeneration |
| TALLRec / ReLLa / ReLLaX | [TALLRec](https://arxiv.org/abs/2305.00447), [ReLLa](https://arxiv.org/abs/2308.11131), [ReLLaX](https://arxiv.org/abs/2501.13344) | LoRA-tuned LLM recommendation, sometimes after semantic history retrieval | C04 uses ordinary static LoRA in both branches and claims value only for the paired operator. `static_lora` and `single_pass` are mandatory controls |
| Cognitive personalized-search memory | [CoPS](https://arxiv.org/abs/2402.10548) | external sensory/working/long-term memory around an LLM | C04 has no profile generator, refinding branch, or external memory scorer; all personalization passes through one local Transformer rank logit |
| Contextual calibration | [Calibrate Before Use](https://arxiv.org/abs/2102.09690) | score a content-free input to correct label bias | C04's null removes history, not task content, and its projection keeps only candidate-order-changing effects; calibration alone cannot satisfy transfer/repeat controls |
| Counterfactual logit pairing | [CLP](https://arxiv.org/abs/1809.10610) | penalize prediction differences for counterfactual token substitutions | algebraically close training regularizer; C04 additionally uses the factual/null difference as the ranking coordinate. If the tangent/operator ablation ties, C04 is reducible to CLP-style regularization |
| LM classifier-free guidance | [CFG for language models](https://arxiv.org/abs/2306.17806) | same LM, conditional minus unconditional token logits | **closest algebraic neighbor**. Plain `h-n` is not novel. C04 can remain distinct only if the candidate-set tangent projection and ranking-specific no-history/perturbation contract outperform the ordinary-difference degeneration |
| Contrastive decoding | [Contrastive Decoding](https://arxiv.org/abs/2210.15097) | subtract expert/amateur likelihoods | uses two models of different capacity and generation-time plausibility constraints; C04 uses one shared model and fixed candidates |
| Recommendation contrastive decoding | [NAR4Rec](https://arxiv.org/abs/2402.06871), [Why Thinking Hurts](https://arxiv.org/abs/2602.16587) | list-dependency or bias-subtracted decoding | neither source, as inspected, defines a history/null-history paired candidate-logit operator with exact no-history anchor; they make generic “contrastive” naming insufficient for a novelty claim |

## Current verdict

**`uncertain`, not `globally novel`.** The raw same-model conditional/null logit
difference is algebraically reducible to language-model classifier-free
guidance and is also close to counterfactual logit pairing. The only candidate
for operator-level distinctness is the request-level tangent projection that
removes shift/scale effects parallel to the null candidate ordering, together
with its mandatory paired ranking use. No searched recommendation or
personalized-search source instantiated that exact operator, but absence from a
bounded search is not proof of novelty.

The pre-outcome decision is therefore to continue only with the tangent form
and to freeze two decisive degenerations: ordinary factual single-pass/static
LoRA, and paired logits without the tangent. If either reproduces C04, the
verdict becomes `reducible` and the track stops; a different prompt template or
dataset application cannot rescue it.
