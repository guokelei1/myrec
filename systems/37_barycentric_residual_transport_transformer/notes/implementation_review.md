# C37 pre-outcome implementation review

The implementation removes C36's failed soft trust operator rather than
changing its coefficient or gate. The candidate-axis barycenter acts on the
internal history residual field; it is not an output router or baseline mix.
The frozen BGE LM remains load-bearing, and all four modes share the same 16,384
trainable adapter parameters, initialization, fit, optimizer, and loss.

A0 directly checks hidden-state mean conservation, exact inactive state,
natural residual/global norm, and direction alignment. It also checks complete
ranking and top-10 activity versus every exact reduction. Candidate hashes are
asserted at every stage; repeat/no-history/no-auth/query-absent fallbacks are
exact; method code has no dev/test qrels or metric input.

The key risk is deliberately unresolved before labels: a mean-preserving
candidate residual can still hurt relevance even when it is selective and
direction-preserving. That is the A1 falsifier. No learned router or post-A
scaling adjustment is permitted.
