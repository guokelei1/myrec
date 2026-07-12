# 2026-07-12 — C47 posterior-supported ridge terminal result

## Hygiene and execution

C47 bound the proposal, three selection-lock generations, final selection,
operator code, thresholds, shared metric source, BGE snapshot, and every
label-free structural input before feature or score materialization.  Its
Amazon feature pass encoded 46,897 distinct items and 300 queries on four A40
GPUs.  A0 passed every check on both domains before train-internal labels were
opened.  The prelock Kuai label-scope incident remained contained: all 2,370
affected indices were excluded from C47 outcome and reserve roles.

## What survived

- History-subspace query transport has a real cross-domain signal.  C47's
  posterior write beat query base by +0.023799 on Amazon with a positive
  interval and by +0.006421 on Kuai with all folds positive, although the Kuai
  interval crossed zero.
- Correct history matters strongly on Amazon: true minus wrong was +0.031113
  with CI [0.013282, 0.048860].  Kuai's +0.005171 point estimate was weaker and
  not significant.
- The fixed operator was numerically clean.  Deterministic error was exactly
  zero; candidate/history permutation errors were at most 3.58e-7; supports
  stayed in [0.00617, 0.76655] on Kuai and [0.06146, 0.74283] on Amazon.

## What failed

The proposed primitive was not the history subspace itself; that is the Cubit/
plain-KRR nearest control.  The primitive was multiplying the mean ridge write
by candidate self-support from the same posterior geometry.  It failed its
incremental-rent tests:

- on Kuai it lost to plain ridge by -0.002871, with all three folds negative;
- on Amazon its +0.008096 advantage over ridge had a zero-crossing interval;
- it tied fixed-temperature softmax on both domains (+0.000324 Kuai,
  +0.000113 Amazon), with zero-crossing intervals and unstable folds.

The exposed-cohort C46/C42 diagnostic therefore did not replicate.  The
candidate self-support scalar is correlated with useful semantic evidence on
Amazon but is not a general fidelity law.  Its contraction suppresses useful
Kuai directions and behaves like an ordinary attention gate on Amazon.

## Next-design constraint

Do not tune `rho`, its exponent, ridge, temperature, or a dataset-specific
mixture between plain and posterior ridge.  A successor must change the
information object, not calibrate the failed contraction.  The remaining
cross-domain fact is narrower: a history-derived query write can help, but its
relevance cannot be certified from candidate membership in the history span
alone.  Any next primitive must determine whether a history direction is
query-conditioned and candidate-discriminative before aggregation, and must
pay rent over both plain KRR and ordinary softmax on a separately frozen fresh
role.
