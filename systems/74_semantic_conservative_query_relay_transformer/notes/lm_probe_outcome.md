# C74 pretrained token-level LM probe outcome

Decision: `close_c74_before_validation_labels`.

The data-free design gate passed in all three seeds, and the separately locked
pretrained-LM G0 also passed every check.  Formal training then completed all
twelve fits (three seeds × four equal-parameter modes) under execution lock
`87739fd784e4d47d9b6f6f8d1d4881ad7d027d9347d708307f4d4aaa228c39d2`.

The intended information path was strongly load-bearing on the 1,200
label-free validation records:

- primary versus base changed `83.67%--84.42%` of complete orders and
  `39.67%--42.33%` of Top-10 sets;
- true versus wrong history changed `82.83%--85.50%` of complete orders and
  `34.75%--38.58%` of Top-10 sets;
- primary correction RMS was `0.09599--0.10660` and true/wrong correction
  difference RMS was `0.08527--0.09830`;
- all three primary/control comparisons passed the frozen activity threshold;
- candidate permutation, determinism, no-history, query-mask, repeat, finite
  gradients, equal capacity, and candidate hashes passed.

A0 nevertheless failed before labels.  Seed 20265202 had decreasing loss in
all four modes, but seeds 20265201 and 20265203 had increasing fixed-window
loss in all four modes.  The all-seed loss-trend condition is binding.

There is also a non-decision-changing report defect.  The seed report set
`validation_labels_closed_during_scoring` from `store._labels is None`; the
same store had necessarily loaded the exposed *training* label container, so
that value became false.  Validation scoring itself always called
`collate(..., label_access=False)`, the score reports recorded labels closed,
and A0 opened no labels.  Correcting the bookkeeping would not pass A0 because
the independent loss-trend failure remains.

A post-terminal, label-free representation audit compared each final primary
checkpoint with its pretrained initialization on a fixed validation batch.
Mean cosine similarity was only `0.909--0.925` for query WordPieces,
`0.786--0.852` for history carriers, and `0.792--0.846` for candidate
carriers.  Thus adapting the final BGE layers materially moved the semantic
coordinate that C74 intended to conserve.  This does not rescue C74, but it
supports a new architectural hypothesis: freeze the LM semantic carrier and
train only the two-hop routing/chronology operator.

The 1,200 validation labels, fresh roles, dev, test, and qrels remain unopened.
No C74 loss, seed, layer, learning-rate, epoch, threshold, report-flag, or A1
rescue is authorized.

Authoritative A0 report:
`reports/pps_c74_pretrained_lm_probe_a0.json`, SHA-256
`0a7a526471138d0bddab5dca2f092a57e131efd152f5e11677d7b9d49f8f455d`.
