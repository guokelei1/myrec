# Motivation V1.1 Axis B freeze — 2026-07-16

Axis A is complete on the byte-identical V1 KuaiSearch confirmation set.
Both declared InstructRec seeds selected epoch 2 by the train-only internal
dev metric. TEM's Axis A was the pre-declared 20-to-40 epoch extension, and
the Axis A run is therefore fixed at epoch 40. Axis B uses exactly TEM 40 and
InstructRec 2 for both seeds.

Axis B changes only the preceding KuaiSearch training population from
`full_confirm_preceding10k_v1` to `full_confirm_preceding40k_v11`. The
confirmation records, qrels, candidate manifest, assignments, and shared
evaluator remain the corresponding frozen V1/V1.1 objects. No epoch selection
or result selection is allowed from the Axis B confirmation surface.

JDsearch and any other dataset remain closed until both KuaiSearch axes are
reported.
