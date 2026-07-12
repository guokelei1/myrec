# Full-train SASRec diagnostic rejected for target contamination

A read-only exploratory command scored the already-open C43-A cohort with the
existing official RecBole SASRec checkpoint. The apparent true-history signal
was extremely large. It is invalid and must not guide architecture selection:
the checkpoint was trained from the complete KuaiSearch `records_train`
interaction artifact, while C43-A is itself drawn from `records_train`.
Consequently, C43-A current positive outcomes could enter the SASRec user
sequence and item vocabulary before that same request was scored.

No source, report, checkpoint, selection, or tracked result was written by the
diagnostic. It did not access dev/test qrels and is not a protocol incident for
an existing candidate, but its numbers are target-contaminated and must never
be cited as behavioral-representation evidence.

The replacement C46 protocol must enforce a chronological source cutoff:
behavioral-model training labels come only from requests strictly earlier than
every outcome request, and outcome labels remain closed until all label-free
scores and corruption controls are frozen.
