# C54 — History-Carrier Competition Transformer

C54 tests one internal Transformer law derived from the terminal C53 audit:
candidate-to-candidate attention may use candidate states as queries and keys,
but its values may carry only a candidate-specific factual-minus-null history
state.  A history-free candidate-list reranker therefore has no value path into
the residual score.

This first gate is mechanics-only.  It reuses C53 fit inputs and the already
feature-exposed, label-unopened C53 A surface.  It cannot open A labels or make
a utility/novelty claim.  Passing only authorizes a separately frozen fresh-
cohort comparison with matched controls.
