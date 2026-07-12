# C04 proposal — CPDLR

Date: 2026-07-11. Status: pre-outcome design; no C04 dev outcome was read.

## Observation → architecture consequence → falsification

**Observation.** In the frozen KuaiSearch B0b/D2s bundle, exact candidate
recurrence is stable while the tested coarse category transfer is
non-informative and dilutes item-only ranking. This establishes unequal
history-evidence fidelity, not semantic transfer, identity causality, or the
superiority of query attention.

**Architecture consequence.** Use one compact masked Transformer as the
end-to-end candidate ranker under two prefixes that differ only in history:

\[
  h_c=f_\theta([q,H,c]),\qquad n_c=f_\theta([q,\varnothing_H,c]).
\]

The two calls share every backbone, LoRA, token embedding, and score-head
parameter. Across the fixed candidate pool, form `d = h - n`, center `d`, and
remove the component parallel to the centered null-logit vector. With
`C=I-11^T/|C_q|`,

\[
  \Pi_n(x)=Cx-\frac{\langle Cx,Cn\rangle}
                         {\|Cn\|_2^2+\epsilon}Cn,
  \quad
  z = n + \Pi_n\!\left(\tau\tanh(\Pi_n(d)/\tau)\right).
\]

`z` is the final LM candidate logit. The projection makes the load-bearing
delta candidate-order-changing: a history effect that merely shifts or rescales
the query-only ordering is removed. No D2p score is added at inference. A
frozen D2p model supplies only a train-split listwise distillation target for
`n`; dev scoring depends solely on CPDLR parameters and the standardized
record.

**Falsification.** The primitive fails if any of the following occurs:

1. the matched factual single-pass/static-LoRA control reproduces the paired
   behavior;
2. the useful delta is confined to exact identity and has no positive value on
   the 4,677 history-present/non-repeat requests;
3. wrong-user, event-shuffled, query-masked, or coarse-only histories retain
   comparable delta;
4. repeat-present behavior falls below the item-only control;
5. no-history requests do not have identical factual/null prefixes, zero delta,
   and D2p-equivalent ordering;
6. removing the candidate-order tangent projection does not remove the gain,
   reducing the operator to ordinary classifier-free guidance/logit pairing.

## Information flow

The tokenizer allocates fixed budgets to the query, the last four strictly
prior events, and the candidate. Query, event type, item identity, item text,
brand, and category appear as tokens. Three stable item-hash tokens use the
same map for historical and candidate items, so exact recurrence is visible
without a generated item ID or a separate item table. The candidate remains
inside the Transformer input for every branch.

The BGE-small-zh-v1.5 four-layer BERT backbone is local. Its last two layers'
query/value projections receive ordinary static rank-8 LoRA updates; the same
updates serve both prefixes. A single linear head maps the `[CLS]` state to a
candidate logit. Factual and null candidates are concatenated along the batch
dimension for shared-kernel execution. Candidate pools are never expanded or
filtered.

## Training signal

Only train records carry labels. The main loss is the sum of:

- multi-positive listwise click loss on final `z`;
- train-only D2p listwise KL anchor on `n`;
- non-repeat positive-versus-negative delta margin;
- repeat-preservation delta margin;
- corruption consistency, cycling wrong-user, shuffled, query-masked, and
  coarse-only prefixes toward zero delta.

The paired operator, not a new loss alone, is the proposed primitive: removing
its candidate-set projection yields a frozen degeneration tested explicitly.

## No-history identity

For an empty history, the factual builder emits exactly the same token IDs,
attention mask, and segment IDs as the null builder. A presence mask sets the
delta to exact zero. The null ranker's D2p order preservation is still an
empirical hard gate; it is not asserted by naming or by the distillation loss.
Any no-history rank mismatch stops C04.

## Bounded claim

This proposal does not claim that semantic transfer exists, that user identity
is causal, that C5-R3 validated CPDLR, or that the operator is globally novel.
The LLM4Rec family is a project scope choice. A one-call screening can only
reject an obviously broken candidate or nominate it for a separately
authorized full gate.
