# C62 — Write-Once Preference-Memory Transformer

C62 tests one architectural primitive: a history-only Transformer writes a
small latent preference memory once; query-candidate tokens may subsequently
read that immutable memory but cannot write back into it.  Candidate logits are
produced by the end-to-end Transformer ranking path.

The first authorized outcome is a data-free, three-seed synthetic G0.  If G0
passes, an exposed-fit dual-domain gate may train the identical architecture on
KuaiSearch and Amazon-C4.  C26 internal-A and the C39 Amazon reserve remain
label-closed until all prior structural and label-free gates pass.

This candidate is not a C61 edge/scale rescue and has no dataset, category,
query-type, or candidate-count branch.
