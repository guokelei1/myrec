# C23-A train gate outcome

Status: **failed at label-free A0; terminal stop before internal-A labels**.

The single hash-locked run completed 3 seeds × 4 equal-parameter modes × 2
epochs on physical A40 GPU 3 in 727.96 seconds.  Every fit was finite; every
mode had 341,569 parameters and matched initialization within seed.  Candidate
hashes, deterministic rescore, candidate centering, candidate permutation,
query-absent item-only fallback, and no-history/non-repeat D2p fallback passed.

## Binding result

RRST changed the seed-averaged full ordering on 238/1,200 requests (19.83%) and
top-10 membership on 97/1,200 (8.08%), so the write was not too small or
candidate-common.  The last-recurrence mask was also exact: replacing all
masked pre-anchor tokens changed scores by `0.0`.

The architecture nevertheless failed its unique load-bearing test.  A frozen
post-anchor suffix shuffle changed the learned correction on only:

| seed | requests changed |
|---:|---:|
| 20260733 | 0 / 1,200 (0.0000%) |
| 20260734 | 1 / 1,200 (0.0833%) |
| 20260735 | 5 / 1,200 (0.4167%) |

The locked minimum was 5% in every seed.  This is not an inactive corruption:
a post-outcome, label-free opportunity audit found a non-identity suffix
permutation on 592/1,200 requests (49.33%; 1,748 candidate suffixes).  Training
loss trajectories for reset, unreset, orderless and query-independent modes
were also virtually identical within each seed.  The model learned a static
recurrence calibration shortcut from anchor/count/action features and did not
use the claimed post-recurrence trajectory.

## Decision and boundary

C23-A is closed.  Internal-A, delayed-B, escrow, dev and test labels were not
opened, so there is intentionally no NDCG utility verdict.  The result rejects
the **last-exact reset plus suffix-survival Transformer** as a load-bearing
primitive under the frozen recipe; it does not reject query-conditioned
recurrence calibration generally.  Do not rerun with more layers, larger
suffixes, auxiliary order losses or a relaxed intervention threshold.

Raw report SHA-256:
`a5c66d72e3f5e87a12e8cbafc4fc903d111501a5b79c9a546c31e2ff3a15296e`.
Proposal-lock SHA-256:
`e23e1cf3a6968784ddc66d4a25f730fe6bead3c35ec00617d1b1e1dfa617ec44`.
Execution-lock SHA-256:
`d028d47e15c88626482f96220c33163c7500580dbbdc557aceb311fb11ddc44b`.
