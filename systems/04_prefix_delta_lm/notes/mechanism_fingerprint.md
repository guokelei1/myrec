# C04 mechanism fingerprint

Status: frozen candidate description before C04 dev outcome.

| Field | C04 value |
|---|---|
| Primitive | candidate-order-tangent shared-LM prefix delta |
| Factual prefix | `[CLS] [Q] query [H] prior event tokens [C] candidate tokens [SEP]` |
| Null prefix | same sequence and candidate, with history replaced by one `NULL_HISTORY` token |
| Shared parameters | one BGE-small masked Transformer, the same static LoRA Q/V updates, token embeddings, and scalar candidate head |
| Candidate state | `[CLS]` state after joint query/history/candidate self-attention |
| Raw evidence | `d_c = f_theta(q,H,c) - f_theta(q,NULL,c)` |
| Intervention layer | candidate-logit vector over the complete fixed request pool |
| Calibrated evidence | centered `d`, projected orthogonally to centered null logits, bounded by `tanh`, then reprojected to preserve the tangent constraint |
| Final rank logit | null LM logit plus the bounded order-tangent; no external scorer at inference |
| Query-only anchor | frozen D2p order used only as a train-split KL target for null logits |
| Training labels | `records_train.jsonl` clicks only; internal holdout is the frozen last-10% train range |
| Counterfactual contract | empty, wrong-user, shuffled-event, query-masked, and coarse-only deltas are trained toward zero |
| Exact recurrence | same three deterministic item-hash tokens in history and candidate; no separate identity scorer in the main model |
| No-history identity | factual and null token tensors identical; presence mask makes delta exactly zero; empirical rank must match D2p |
| Inference input | label-free standardized query, strictly-prior history, fixed candidates, and evidence masks |
| Candidate generation | none; every output key is copied from the manifest-bound candidate record |
| PEFT scope | static rank-8 LoRA on Q/V of the final two Transformer layers plus one score head |
| Compute reuse | factual/null examples share tokenizer caches, one parameter set, and batched Transformer kernels; no KV reuse claim for bidirectional BERT |
| Online cost | two compact LM passes per candidate plus `O(|C|)` projection; zero online API/large-LLM calls |

## Named components (complexity budget)

1. **Shared paired-prefix LM** — produces both candidate logits.
2. **Candidate-order tangent** — turns the paired logit difference into an
   order-changing evidence coordinate.
3. **Anchor-and-consistency contract** — train-only D2p order distillation plus
   corruption-to-zero constraints.

Every component has a preregistered degeneration. Nothing routes among fixed
scorers or uses a query-type branch.

## Degenerations and matched controls

| Degeneration | What remains | Falsifying interpretation |
|---|---|---|
| `single_pass` | same structured factual LM, LoRA, and head; no null/delta | equal behavior means paired evidence is unnecessary |
| tangent removed | ordinary `h-n`/factual scoring | equal behavior reduces C04 to classifier-free guidance/logit pairing |
| `concat_head` | flat query/history/candidate concatenation and ordinary head | equal behavior means field/prefix structure adds nothing |
| `static_lora` | same local backbone/LoRA/head on the query/candidate null path | equal behavior attributes the result to adapter capacity |
| `identity_shortcut` | null LM plus learned exact-repeat flag | equal behavior means the candidate has no cross-item mechanism |
| all history corrupted | same candidate/null path, invalid history evidence | surviving delta invalidates evidence fidelity |

The main and `single_pass` variants have the same trainable parameter set. The
static-LoRA control isolates ordinary PEFT capacity. The shortcut is a
mechanistic lower-dimensional diagnostic, not a capacity-matched claim control.
