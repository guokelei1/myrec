# 2026-07-11 qrels schema-inspection incident

Status: recorded and quarantined; no model/evaluator contamination observed.

While checking whether standardized KuaiSearch records expose `user_id`, an
interactive read-only schema-inspection command globbed every JSONL file under
the standardized directory.  Its exclusion covered `item_catalog` but did not
exclude `qrels_dev.jsonl` or `qrels_test.jsonl`; the command printed the first
record of each qrels file before returning to the record files.

Scope and impact:

- this happened after the C28 formal run, terminal report, and all C28-A
  post-terminal probes described before the user-ID check;
- no candidate source, training/scoring process, checkpoint, selection,
  evaluator, metric computation, or configuration opened either qrels file;
- the printed relevance IDs are quarantined and must not be copied into any
  selection, prompt, feature, test, example, or design decision;
- prior method-level declarations that their scoring/training code did not
  read dev/test labels remain true;
- the stronger repository-wide statement that no operator ever viewed any
  dev/test qrel is no longer strictly true and must not be made after this
  timestamp.

Containment: future schema inspection must use an explicit allow-list
(`records_train.jsonl` only for labeled training inspection, and label-free
record files only when authorized), never a directory glob.  Dev/test qrels
remain forbidden to every model, scorer, trainer, selector, and evaluator
outside the common locked evaluation workflow.
