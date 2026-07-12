# C13 Set-Relative RMS/Whitened Write Transformer

Status: **rejected at the paper pre-implementation gate**.  No model, runner,
config, manifest, GPU run, or data access exists.

Candidate centring is structurally useful, but the scalar RMS form is exactly a
request-adaptive scalar rescale.  The only form that is not scalar—set-covariance
whitening—systematically gives larger relative gain to weak singular modes and
has no evidence-fidelity mechanism to distinguish weak signal from wrong-user
noise.  With strong regularization it tends back to a fixed scalar; with weak
regularization it equalizes noise.  Generic set normalization and activation
whitening also predate this placement.

- `proposal.md`: hypothetical Transformer information flow and contracts
- `algebraic_and_constructive_audit.md`: reductions and requested witnesses
- `novelty_and_risk_audit.md`: epsilon/noise dilemma and prior-art boundary
- `matched_controls_and_complexity.md`: exact controls, parameters, and FLOPs
- `minimal_falsifier_design.md`: synthetic and real A0 gate if revisited
- `preimplementation_decision.md`: binding rejection
