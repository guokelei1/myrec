# 2026-07-11 — C23 recurrence-reset terminal decision

C23 first rejected a query-conditioned delta-rule sketch because DeltaNet,
Gated DeltaNet and recommendation-specific SinkRec already cover that operator.
The replacement RRST hypothesis made the last exact recurrence a hard
candidate-specific Transformer origin and allowed only later events to evolve
the recurrence state.

The label-free structure justified one gate: 76.29% of repeated candidates had
a nonempty post-anchor suffix.  Selection was frozen before labels; correct
seed-20260708 D2p states and 765,874 candidate scores were key-aligned bitwise;
G0 then copied fit labels only.  Three seeds and four equal-parameter modes
completed on GPU 3.

RRST passed every safety/activity check except the mechanism witness.  It
changed 19.83% of request rankings, but suffix shuffle affected at most 0.42%
per seed even though 49.33% of requests received a non-identity shuffle.  The
nearly identical loss traces across reset/unreset/orderless/query-independent
modes reinforce the same interpretation: rank supervision selected a static
recurrence shortcut and ignored query-conditioned suffix evolution.

The runner stopped before internal-A labels.  C23 may not be tuned or promoted.
The next gate should not force sequence sensitivity with an auxiliary loss;
it should test a different information object.  A defensible next object is
**competition among multiple exact-recurrence candidates**: 14,179/25,122
repeat requests contain more than one repeated candidate.  A new candidate may
ask whether a Transformer-internal, listwise recurrence allocation pays rent
over independent recurrence calibration and ordinary candidate self-attention.
That hypothesis requires a new fingerprint and untouched delayed label role.
