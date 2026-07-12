# C26 pre-proposal token opportunity audit

This descriptive audit used a deterministic 1,000-request sample from C25's
already-open fit role only.  It did not read C25/C26 internal-A, delayed-B,
escrow, dev or test labels.  It decoded registered BGE WordPieces from the
label-free corpus and query-token arrays.

- 65.0% of requests had at least one query token also present in history;
- 57.0% had candidate-varying exact query/history/candidate bridge counts;
- clicked-minus-unclicked exact bridge mean was `+0.00785295`;
- clicked-minus-unclicked query-candidate token match was `+0.02077431`;
- only 44.3% had a positive bridge maximum above the negative mean.

This establishes a nonempty but weak token-level surface.  It does not validate
the architecture, select a subset, tune a coefficient or support a paper
claim.  In particular, the stronger query-candidate point motivates a mandatory
candidate-only late-interaction control.
