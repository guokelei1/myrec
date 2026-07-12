# C06 review1: selective cycle-energy numeric repair

Status: **implementation review only; review1 execution lock not yet created**.
No GPU retry or internal-A operation is authorized by this note alone.

## Pre-repair evidence boundary

The v1 G0 report is frozen at
`7430dae9c56b257cb64a9c75e3e0cbf932856d9419e296b64fa1e9cc81a0af1e`.
The independently trained centered cross-attention control completed both fit
epochs; its report and checkpoint are frozen at:

- report `b464a99a9d679c64d8388e43a3fe801dcbbd9f798a936c852415f5e29c5252e2`;
- checkpoint `6226df4a49d82e051b65aa28beaa6e29bff1738139486adb870879ae42e36602`.

Local Hodge, untrusted, and direct-gate v1 fits stopped on the same contracted
`candidate_cycle_energy` negative-roundoff guard. Their original ledgers remain
unchanged and are frozen as:

- local `c90717a7079beb97c4873ab0d08674660da4772be927f42aded8d0659a060a92`;
- untrusted `8fbb37f0f87f3a6d33197f483ffa1fa22ac0f82dad1193637aa344e70de88cb3`;
- direct `31c3020a0231280b71def42a7d07f7857dbb3dcc61cce673e13cd046e3a75740`.

Each contains one `started` fit-only attempt and explicitly records no A
feature scoring, A label opening, delayed-B, or escrow access. The failed
processes did expose fit-training telemetry in memory and may have completed
some fit optimizer steps; no comparative/A ranking outcome was observed.
Centered fit loss telemetry and its final fit checkpoint/report were observed.
The repair decision uses only the common numeric exception, never centered
quality, A, a matched comparison, dev, or test.

The original exception path did not persist factor tensors or minimal numeric
values (`n/r`, contracted value, tolerance, magnitude, explicit energy).
Review1 declares that metadata unavailable rather than reconstructing it after
the fact. No input tensor is added to repair logs.

## Byte-identical scientific parent

The parent lock is
`58b361ec3bebbb306b0d3069f1411de82eb92495f3b2ec3bfa72623f26ab0a42`.
The exact parent config bytes are preserved at
`configs/c06_real_mechanism_gate_parent_v1.yaml`, whose SHA256 is
`c487018f0a2cb831bef97cfb6e7b71c1c8a8c028ba90cdc08f1cb5ce5afad650`.
Thus cohort IDs, model dimensions, controls, seed, optimizer, learning rate,
batching, two epochs, thresholds, metric, and GPU mapping are mechanically
unchanged. The review1 config adds only provenance, retry run IDs, and the
numeric implementation repair.

## Mathematical repair

The fast path remains the original FP64 centered-factor Gram contraction. For
candidate/event rows whose contracted energy crosses the existing roundoff
guard, and only those rows, review1 computes

```text
C_ikj = alpha * (x_ij^T y_kj - y_ij^T x_kj)
EC_ij = sum_k C_ikj^2
```

in FP64. This is exactly the same Hodge cycle-row energy and costs `O(C*r)` per
fallback row. It remains in the autograd graph.

The fallback cannot hide a contraction/index bug. Independently computed
primitive absolute dot sums define a standard FP64 `gamma_k` forward-error
bound. Both the original contracted energy and its registered component
magnitude must agree with the explicit identities inside that bound. A
material discrepancy still raises `FloatingPointError`; only a negative value
consistent with honest forward error is replaced. One- and two-candidate exact
zero behavior is unchanged.

Tests cover production BF16-to-FP64 factors with a legal negative-error
intervention, explicit oracle equality, oracle-gradient equality, gross-error
rejection even when magnitude is correct, ragged masks, inactive negative
rows, candidate permutation, and ordinary zero-fallback behavior.

## Retry and logging semantics

Only `local_hodge`, `untrusted`, and `direct_learned` receive one repair retry.
Each uses its original physical GPU but a distinct repair run ID and a new
`formal_attempt_repair1_<variant>.json`; the original v1 ledger is never
modified. A retry exception is durably marked `failed` with exception type and
message before it is re-raised. Centered execution is rejected in code and its
v1 files are hash-revalidated.

Each retry report records per-epoch and total
`candidate_cycle_energy_fallback_rows`. The repair ledger repeats the total.
Before A0, the audit loader requires all three retry ledgers to be `completed`,
their reports/checkpoints to use review1, and the exact parent G0/centered
artifacts to remain unchanged.

## Review1 lock requirements

The executable review1 lock must declare:

- observed numeric implementation failure and observed fit telemetry;
- no internal-A-or-later ranking outcome, A score, A label, B/escrow, dev/test;
- parent v1 lock/config, G0, centered report/checkpoint hashes above;
- the three unchanged v1 failure-ledger hashes;
- all candidate-local source/config/protocol/test hashes and shared metric hash;
- no threshold, model, data, seed, optimizer, epoch, batching, or metric change.

Until that lock passes `assert_real_gate_lock`, all repair commands fail closed.
