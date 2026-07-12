# C08 G1 outcome — failed, terminal stop

The fully specified, hash-locked G1 learned synthetic falsifier ran exactly once
on the three frozen CPU seeds. Result: **0/3 seeds passed; C08 stops before any
repository data, GPU, dev evaluator, or test access.**

Execution was bound to aggregate
`51d7c0a0f8c940f85271313293b9d16ba77e158040369211d11e19da004bd924`.
The independent verifier passed both before and after execution. All four methods
had 5,994 parameters, byte-identical per-seed initialization, the same 4,096
training requests, batch order, optimizer, and 400 steps. Raw artifacts are
ignored under `runs/g1_51d7c0a0f8c940f8/`; its `RUN_COMPLETE.json` SHA-256 is
`42f2e4c6f70b3e73e01ed3de8cac1af34d28e61034c8c1b813cd9449a484c309`.

| Seed | RWPU repeat / item control | RWPU supported | Ordinary | Attention | Pooled FFN | Full gate |
|---:|---:|---:|---:|---:|---:|---|
| 20260711 | 0.6445 / 1.0000 | 0.5859 | 0.5000 | 0.5000 | 0.4941 | fail |
| 20260712 | 0.5430 / 1.0000 | 0.5273 | 0.5469 | 0.5586 | 0.5508 | fail |
| 20260713 | 0.5410 / 1.0000 | 0.5430 | 0.5547 | 0.7129 | 0.5449 | fail |

The load-bearing failures are decisive:

- repeat preservation failed all three seeds; the frozen floor was 0.99 against
  the deterministic 1.0 item-recurrence control;
- supported non-repeat surplus over the best matched control occurred only in
  seed 20260711, not 3/3; seeds 20260712 and 20260713 lost to attention;
- surplus over ordinary memory likewise failed seeds 20260712 and 20260713;
- corruption retention failed seeds 20260712 and 20260713; in seed 20260713,
  wrong, shuffled, and disjoint margins retained approximately the entire clean
  margin;
- the strict `1e-6` candidate-permutation numerical threshold failed all seeds
  (maximum errors `2.15e-6`, `5.48e-6`, and `1.91e-6`). The threshold remains
  unchanged.

Exact empty-history fallback, finite optimization, and positive clean margin
passed all seeds. Those structural properties do not rescue the failed learned
mechanism claim. The endpoint-collision witness remains a mathematical property,
not evidence that RWPU learns reliable personalized ranking.

Terminal decision: **do not run G2 or any real-data probe; do not tune, rerun,
relax the permutation tolerance, or replace the failed controls under C08.**
