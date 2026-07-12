# Amazon token-HSO report-flag mechanical recovery

Status: pre-outcome recovery protocol.  Frozen before any reserve label or
metric is opened.

All three token-HSO seeds completed fixed training and label-free reserve
scoring.  Every substantive check is true: decreasing/finite loss, backbone and
head gradients, candidate hash, deterministic scoring, exact candidate-order
equivariance, and closed dev/test/qrels.  The generated aggregate
`passed_mechanics` field is false only because the report dictionary stores the
factual field `reserve_labels_opened: false` and then applies
`all(checks.values())`; correct label isolation is therefore interpreted as a
failure.

This recovery binds the three existing report and score hashes.  It defines a
seed as mechanically valid iff every check other than `reserve_labels_opened`
is true and `reserve_labels_opened` itself is false.  It performs no training,
checkpoint selection, rescoring, threshold change, seed change, or candidate
change.  Only after the recovery lock may the unchanged summarization logic
open reserve labels and apply the original doc/28 outcome gate.
