# C65 — Counterfactual Residual-State Transformer

C65 prevents the adaptive LM from acting as a generic query-candidate reranker.
The same LM/joint Transformer produces factual-history and NULL-history
candidate states; only their internal residual may change the strong base.
Matched wrong history is trained to make that residual rank-neutral.

The first gate reuses C64's exposed-fit split whose validation labels remain
closed.  Hidden residual without wrong loss, ordinary factual state, and output
logit difference are equal-parameter controls.  Fresh roles, dev, test, and
qrels remain inaccessible.
