# C41 pre-outcome implementation review

## Verdict

C41 is suitable for proposal lock and label-free feature materialization as an
architecture-boundary probe. It is not suitable for a novelty claim before its
real gate, and a positive gate would still require a new design review.

## Evidence and isolation

- The inherited design gate passed 12/12 checks and is exactly equivalent to
  C40 `selection_only` at matched state (`0` max error).
- C41-A is exactly C38 delayed-B: 1,200 requests with zero overlap against
  C38-A, C39-A, and all prior C38 feature indices. C41 delayed-B is C38 escrow
  and remains unmaterialized.
- Fit is exactly the 6,000 requests used to train the frozen C38 strong control.
  C41 creates its own fit-label artifact only after proposal lock.
- Wrong donors have 100% coverage, 100% same-length-bin matching, and zero
  same-user assignments.

## Architecture and controls

- The primary has 49,152 parameters and learns routing only. History values,
  query transport, and candidate readout remain raw normalized BGE states.
- `single_wide_routing`, `asymmetric_routing`, and `coupled_content` have the
  exact same parameter tensors and paired initialization.
- Fixed semantic attention, uniform history, and three immutable C38
  unprojected checkpoints are scored before A labels open.
- Every method uses the shared metric implementation; no candidate-local
  evaluator or qrels read exists.

## Known limitations

1. Identity V/O and routing/content separation have direct prior art. C41 is a
   foundation test, not the final innovation.
2. The reused C38 feature pipeline writes `candidate_id=c38` in its mechanical
   manifest. G0 explicitly binds the C41 selection hash and records this reuse;
   it does not reinterpret the manifest as a C38 outcome.
3. The C38 control has fewer parameters than C41 but is a strong predecessor,
   not the only capacity control; three C41 controls are exactly matched.
4. All positive gates are conjunctive. A large base gain cannot compensate for
   tying C38/fixed/matched controls or for true/wrong nonspecificity.

## Stop conditions

Any lock, feature, G0, A0, or A1 failure is terminal. No head/rank/temperature/
scale/loss/epoch/seed/cohort adjustment is permitted, and delayed-B/dev/test
stay closed.
