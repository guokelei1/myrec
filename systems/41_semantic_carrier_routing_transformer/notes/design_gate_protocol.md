# C41 inherited data-free design gate

Status: frozen before `probe/run_design_gate.py --stage run`.

C41 is a direct successor to the winning C40 reduction. It does not construct a
new favorable generator. The gate instead binds the immutable C40 report and
requires both inherited conditional evidence and exact implementation identity.

## Checks

1. `reports/pps_c40_design_gate.json` has the frozen SHA-256 and confirms no
   repository data/dev/test read.
2. C40 D0 passed every check.
3. C40 `selection_only` exceeds coupled primary by at least `0.03` in all three
   seeds and has clean-minus-wrong NDCG@10 at least `0.45` in all seeds.
4. C41 primary is numerically identical to C40 `selection_only` under the same
   parameters and inputs within `1e-7`.
5. All four C41 modes have equal parameter counts and paired initial state.
6. Both factors receive gradients; outputs are finite and deterministic.
7. Candidate permutation error is at most `1e-6`; no-history, absent-query, and
   repeat corrections are exactly zero.
8. The primary profile exactly equals attention-weighted raw normalized history,
   attention is nonnegative, and each head sums to one.

Any failure closes C41 before repository-data training. A pass authorizes only
freezing the minimal untouched train-internal boundary gate. It does not prove
novelty or open any label/dev/test input.
