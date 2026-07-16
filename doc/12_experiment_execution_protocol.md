# Experiment execution protocol

Status: supporting operational contract for Motivation V1.2. Method order,
seed staging, compute limits, and stopping rules come from the V1.2 plan.

Every run records its command, config, code revision, dataset/version and
manifest hash, seed, environment, checkpoint reference, and output paths.
Use `YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>` run IDs.

Before evaluation:

- assert the candidate-set hash and complete score coverage;
- assert that dev/test records are label-free;
- assert that scoring/training has not opened qrels;
- use the shared evaluator only;
- append every dev call to `reports/dev_eval_log.jsonl` and reconcile it with
  the registered budget.

For counterfactual response measurements, keep the checkpoint, query,
candidate slate, token budget, and scoring parameterization fixed. Change only
the registered history condition (`true`, `null`, or matched `wrong`).
Determinism checks establish a numerical noise floor before any direction
threshold is interpreted.

Confirmation is an independent frozen execution. It may not change data
eligibility, checkpoint selection, thresholds, endpoint definitions, or
analysis after outcome access. Test remains closed until the complete claim is
confirmed and explicitly released.
