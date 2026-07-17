# Current Motivation evidence reports

The frozen V1.2 summary, assignment release, and append-only development ledger
are kept here:

- [`motivation_current_summary.json`](motivation_current_summary.json):
  frozen first-round Q0--Q3 and W0 evidence baseline;
- [`motivation_kuaisearch_assignments.json`](motivation_kuaisearch_assignments.json):
  release-bound history assignments;
- [`dev_eval_log.jsonl`](dev_eval_log.jsonl): shared-evaluator development log.

The summary is preliminary at one frozen pilot seed. Current mechanism work is
authorized only by
`experiments/motivation/mechanism_analysis_plan.md`; it does not authorize
source-test access, method replacement, or a new architecture. Mechanism
results must use a new report rather than overwrite this freeze. Historical
authority strings inside the frozen JSON may name the deleted first-round plan
and execution prompt; they are provenance, not active links.
