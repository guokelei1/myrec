# C63 proposal — finite-evidence memory Transformer

Status: pre-outcome design.  No C63 model outcome or repository data has been
observed.

## Observation → architecture consequence → falsification

**Observation.** C62's two-phase write/read graph was mechanically correct,
fully trainable, and exactly query-independent, but four standard cross-
attention slots behaved like one pooled state.  The cause is structural:
ordinary cross-attention normalizes over events independently for every slot,
so the same event can be written at full strength into all slots.  Learned slot
seeds being numerically different does not create exclusive binding.

**Architecture consequence.** C63 preserves C62's history-only write and
immutable query-candidate read, but replaces the write normalization.  For
history event `t` and ordered slot `s`, a Transformer compatibility logit
produces a break fraction `beta_ts`.  The event allocates

```text
p_t1 = beta_t1
p_ts = beta_ts * product_{r<s}(1 - beta_tr)
p_tNULL = product_s(1 - beta_ts).
```

Thus `sum_s p_ts + p_tNULL = 1` for every event.  Evidence already written to
an earlier slot cannot be independently duplicated into a later slot, while
irrelevant events may retain mass in NULL.  Each preference value is the
mass-normalized Transformer aggregation of events assigned to that slot.  The
memory is still written without query/candidate access, frozen for the forward
pass, and read by joint query-candidate states before listwise Transformer
scoring.

This is not a score gate: NULL and finite mass exist only inside representation
formation.  The final logits come from the end-to-end Transformer candidate
states.  Empty history and repeat requests retain exact structural fallbacks.

**Falsification.** C63 is rejected before repository data unless all three
seeds:

1. conserve every event's write mass including NULL;
2. assign more NULL mass to planted nuisance events than to useful preference
   events;
3. solve a four-interest, nuisance-contaminated binding task with correct and
   wrong histories behaving differently;
4. beat same-parameter Slot Attention-style competition, balanced Sinkhorn
   transport, C62 per-slot softmax, and a single pooled memory by frozen
   margins;
5. pass query-independence, no-history, repeat, candidate-permutation,
   determinism, finite-gradient, and parameter-parity contracts.

Failure forbids break-temperature/bias, slot-order, slot-count, width, depth,
step, seed, or repository-data rescue.

## Matched modes

- `finite_evidence_memory` (primary): event-wise stick-breaking allocation with
  explicit NULL remainder.
- `slot_competition_memory`: event-wise softmax across slots, the minimal Slot
  Attention/inverted-attention normalization.
- `balanced_transport_memory`: fixed-iteration doubly normalized event-slot
  transport, the nearest balanced OT reduction.
- `standard_slot_memory`: C62-style independent softmax over events for every
  slot.
- `single_pooled_memory`: primary slots averaged and repeated before read.

All modes instantiate the same projections, slot keys/values, history
Transformer, reader, candidate-set Transformer, and output head.  They use the
same initialization family, batches, labels, loss, optimizer, and steps.

## Innovation boundary

Slot Attention already establishes event competition; MESH establishes the OT
view and stronger tie-breaking; stick-breaking attention and Ordered Memory
establish sequential finite allocation in other attention/memory settings.
C63 therefore does not claim any of those ingredients alone.  Its provisional
claim is narrower: **event-wise finite evidence conservation with a NULL
remainder is the binding law for query-independent personalized memory**.  It
must beat all named reductions before this is treated as a candidate
architecture rather than a recombination of known mechanisms.

## Dataset-independence boundary

G0 is entirely synthetic and represents only the prerequisite information
object: four independently addressable preferences plus nuisance events.  If it
passes, a separately locked dual-domain gate must use the identical operator
and hyperparameters on KuaiSearch and Amazon-C4.  C26 internal-A, C39 reserve,
dev, test, and qrels remain closed under this proposal.
