# C24 train-gate outcome

Status: **terminal failure at label-free A0**.

The locked GPU run completed three seeds, three 170,049-parameter modes and two
epochs per mode (2,376--2,388 optimizer steps per seed across the modes).  Fifteen
of sixteen A0 checks passed.  The primary changed 228/600 request orders and
93/600 top-10 memberships relative to registered item-only, while all
determinism, permutation, centering and structural fallback contracts held.

The mechanism-specific check failed decisively.  Removing only
candidate-to-candidate attention changed the numerical correction on 600/600
requests in every seed, but changed **0/600 rankings in every seed**.  Training
loss traces for set attention, diagonal attention and query-vector ablation
were also nearly identical.  The rank-active component therefore came from
independent recurrence calibration; cross-candidate competition was a
non-load-bearing perturbation.

Per the frozen stop rule, internal-A labels were never opened.  Escrow, dev and
test also remain closed, so C24 has no utility/NDCG verdict.  It cannot be tuned
or promoted.  This closes generic set-attention competition as the next
information object on the frozen recurrence representation; it does not claim
that every possible recurrence-specific interaction is useless.

Raw authority:
`artifacts/c24_multi_recurrence_competition_probe/train_gate_v1/train_gate_report.json`
with SHA-256
`efbcfd15c30a297a7339966b95481f203eadc466458e776a577dadc27a9795a5`.
