# C56 v1 pre-outcome execution abort

The first label-blind selection invocation wrote a terminal `failed` status
even though every substantive isolation check passed.  The implementation had
placed the expected sentinel `fit_labels_read: false` inside a dictionary later
reduced by `all(checks.values())`; the sentinel therefore made the aggregate
false by construction.

This was detected immediately from the selection JSON.  No contextual token
was materialized, no model was initialized or trained, no holdout label was
read, and no ranking/metric outcome existed.  The immutable v1 selection and
locks remain preserved under `signal_gate_v1`.

The clean v2 supersession changes only the status predicate representation:
the check becomes `fit_labels_closed: true`, while `fit_labels_read: false`
remains a separate audit field.  It also uses disjoint v2 selection/artifact/
checkpoint paths and new proposal/execution locks.  Architecture, split seed,
role counts, modes, loss, thresholds, seeds, GPU mapping, and unopened-label
boundaries are unchanged.
