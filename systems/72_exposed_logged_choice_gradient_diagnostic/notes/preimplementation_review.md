# C72 preimplementation review

Decision: `authorize_exposed_fit_formulation_diagnostic_only`.

- C71 produced no utility outcome, so C72 does not select a formula from C71
  labels.
- C47 fit labels are already exposed and explicitly bound through C53's
  materialization/report evidence.
- A preliminary structural scan found 4,544 eligible requests; 600 can be
  selected without looking at their labels.
- Every mathematical and statistical setting is copied from C71. C72 may not
  tune after seeing exposed-fit utility.
- Results must be described as descriptive formulation evidence, not a fresh
  architecture result.

Forbidden: C47-A/reserve, C71 target reuse, alternate label, temperature,
normalization, coefficient, donor rule, second subset, dev/test/qrels, or C70
implementation without a second real domain.
