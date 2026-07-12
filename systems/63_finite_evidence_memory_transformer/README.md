# C63 — Finite-Evidence Memory Transformer

C63 is a data-free prerequisite test for a new Transformer write operator.  A
history event owns one finite unit of write mass and allocates it sequentially
across preference slots; any unallocated remainder goes to an explicit NULL
sink.  This replaces C62's independent per-slot softmax, which let every slot
pool the same events.

The first and only current authorization is a locked three-seed synthetic G0.
No repository data or label may be read unless that gate passes against Slot
Attention-style competition, balanced Sinkhorn transport, C62 standard slots,
and a pooled reduction.
