# Transformer mechanism current-batch freeze (v1)

Status: frozen for closeout before any next-batch decision (2026-07-20).

This file fixes the meaning of the current batch: complete the workers already
running or already registered as downstream queues, then produce one combined
mechanism view.  No new layer, head, seed, dataset, control family, or
outcome-dependent branch may be appended while this batch is running.

## Included in the closeout batch

1. The currently live D2/D5/D6 workers and their existing shared evaluators,
   including the active D5 Q2 RoPE shards, Q3 selected-branch shards, and the
   already-started Q0 branch lane.  These are not restarted or duplicated.
2. D4 MLP-formation recovery, component-necessity lanes, and the registered
   component-design synthesis/evaluator gates.
3. The already registered operator chain N8--N16:
   joint attention/MLP composition, history-path formation/transport,
   candidate-gap and rank-path geometry, pre-mask QK logits, SwiGLU stage,
   Q/K/V projection, history embedding, residual composition, and RMSNorm.
4. The already registered boundary chain N17--N20:
   head RMSNorm, GQA grouping, complete Q3 LoRA branch, and Q1 cache phase.
5. N25 SwiGLU formation, because its gate/up/SiLU/product intervention is
   already queued and directly closes the remaining MLP-formation question.
   N26 is an integration/readout cross-check of existing D6 evidence only; it
   does not create another GPU family.

The order and gates remain those in the existing manifests and queue scripts;
the freeze only prevents adding work to them.

The D4 recovery queue additionally requires all physical GPUs to be idle after
the upstream wave closes.  This prevents older registered handoff watchers from
sharing a card at the boundary; it changes scheduling safety only, not the
registered experiment set.

## Explicitly deferred to a later batch

- N27 mask/softmax topology, N28 complete scaled-QK formation, and N29
  attention-by-MLP factorial interaction: CPU primitives may be tested, but no
  GPU scorer or queue is authorized in this closeout batch.  Their value will
  be judged only after the N8--N25 evidence view is assembled.
- N21--N24 training-boundary probes: not needed for the first inference-side
  mechanism diagnosis and remain inactive.
- N30--N34 embedding, RMSNorm, residual-addition, GQA, and adapter extensions:
  preparation code may remain local, but no formal runs are authorized.

## Stopping rule and deliverable

When every included evaluator has a terminal status (completed or an explicit
mechanical failure), assemble a single report with the H0--H5 matrix,
attention/MLP/residual evidence, identity and coverage gates, cross-model
consistency, and a ranked list of design-relevant conclusions.  Pause there;
do not launch a next batch until that report has been reviewed.
