# Motivation V1 direction decision (historical)

Status: frozen historical scope after Motivation V1 consolidation. Current
execution is governed by `doc/43_llm_rerank_recurrence_transfer_research_logic_zh.md`
and `experiments/motivation_v1_2/plan.md`. The dataset and authorization order
below is retained only to interpret V1 evidence and must not redirect V1.2.

The project remains Query-conditioned Personalized Product Ranking and
LLM4Rec. The universal direction-gap hypothesis is closed: Amazon-C4 has
strong target-nonrepeat direction, and JDsearch nonrepeat direction is above
chance. The pre-audit candidate-overlap surface is also no longer called target
recurrence.

The active motivation is the target-aware incremental-personalization problem:

> Ordinary full-token history rankers can respond broadly to history while
> target recurrence, target-nonrepeat recovery, and the end-model observed-label
> gap remain sharply different. In the frozen KuaiSearch Full confirmation, an
> aggregate-successful task-adequate model learns strong target recurrence, but
> does not establish practically meaningful same-checkpoint history increment on
> the majority target-nonrepeat/no-candidate-overlap surface. Response,
> true/null, and overall therefore do not individually identify reliable
> incremental personalization.

The 2026-07-16 V1 extension applies the identical target-aware decomposition to
three frozen query-conditioned Transformer rankers on the same KuaiSearch
confirmation population. Qwen3, TEM, and InstructRec all have a significant
target-repeat positive control and a positive repeat-minus-no-overlap contrast,
while none establishes recovery on target-nonrepeat/no-candidate-overlap. TEM
and InstructRec have near-zero aggregate true-minus-null because surface gains
and losses cancel, not because their scores ignore history. The current concise
claim and code audit are in
[`40_transformer_recurrence_transfer_motivation_v1_zh.md`](40_transformer_recurrence_transfer_motivation_v1_zh.md).

The exact descriptive accounting is:

```text
end-model gap = null-path gap + same-checkpoint recovery
```

Matched-null has now shown that missing-history OOD explains part of old
null-path gaps. `base erosion`, `repayment`, negative transfer, and architecture
failure remain unsupported causal descriptions of the final bounded result.

## Dataset roles

- **Main:** KuaiSearch Full. Latest-window runs supply exploration, and the
  older disjoint `full_confirm_preceding10k_v1` population supplies the passed
  frozen confirmation.
- **Replication:** KuaiSAR Full. It supplies functional behavioral replication
  and cannot support a plaintext-semantic claim.
- **Pre-registered fallback:** JDsearch. Existing v3 results remain valid
  exploratory functional evidence. It replaces KuaiSAR only if the latter
  fails its source boundary, not because of model outcome.
- **Non-binding boundary:** Amazon-C4 plus history companion. It is a constructed
  semantic stress test and cannot estimate natural-search prevalence.

KuaiSearch Lite, HSTU, SASRec, ZAM, and LLM-SRec remain exploratory or
diagnostic boundary evidence. The fresh Qwen confirmation remains the binding
pre-registered ordinary-decoder result. TEM and InstructRec add a matched
post-confirmation cross-model surface replication on the same frozen population;
they do not convert the result into a universal LLM-family theorem.

## Historical authorization (inactive)

The motivation-repair round is complete. Its superseded plans, duplicate
audits, and exploratory generated reports were removed during V1 consolidation.
The active evidence chain is now limited to the disjoint frozen Qwen
confirmation and the same-population Qwen3/TEM/InstructRec audit linked from
the V1 entry.

This authorization paragraph described the stopped V1 workflow. It is not the
current task boundary; use the V1.2 plan and execution prompt instead.
