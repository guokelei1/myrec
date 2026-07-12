# C47 preselection v2 mechanical abort

The physical-line reader repair worked and all structural facts were correct,
but v2 still stopped before writing the selection. Its checks dictionary stored
the facts `labels_opened: false` and `dev_test_qrels_opened: false`, then applied
`all(checks.values())`. The two correct false values were therefore treated as
failures.

No selection, feature, score, label, dev/test record, or qrels was written or
opened. The only authorized v3 change is to express those same facts as
`labels_closed: true` and `dev_test_qrels_closed: true`. Scientific settings and
cohort construction remain byte-identical.
