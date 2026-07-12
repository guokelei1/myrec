# C70 preimplementation review

Decision: `reject_current_execution_missing_cross_domain_choice_context`.

Positive properties:

- the hypothesis directly targets the ranking-direction failure exposed by
  C28--C33 and C69;
- negatives come from the user's real opportunity set rather than a tuned
  semantic or cross-user sampler;
- the proposed internal write is dataset-agnostic and admits exact no-history
  and recurrence contracts;
- matched reductions can distinguish information-object value from generic
  fast-weight capacity.

Blocking property:

- recoverable real logged-choice coverage is 96.56% on KuaiSearch and 0% on
  Amazon-C4; JDsearch does not expose historical slates in its published row
  format and the full dataset is not local.

Implementing only the KuaiSearch path would violate the current cross-domain
specificity requirement and would answer the user's concern by moving toward
the dataset. No source tree beyond design notes, no config lock, and no GPU run
is authorized. Acquisition/interface work is a separate scope decision.
