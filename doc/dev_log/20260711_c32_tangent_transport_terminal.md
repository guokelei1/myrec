# C32 spherical tangent transport terminal

C32 isolated the geometric hypothesis suggested by C31: authenticated history
may change the query only in the tangent space orthogonal to its current LM
direction.  It inserted `t=(I-qq^T)p` and changed nothing else—same frozen BGE,
rank-16 shared adapter, raw semantic attention, fit data, loss, optimizer,
scales, one epoch, and exact fallbacks.

The result is the first clearly positive formal architecture outcome in this
search.  On a cohort never used by C31, every seed gained about 0.004 NDCG@10,
the ensemble gain was +0.004268, the 95% interval was strictly positive, and
true history significantly beat wrong history.  A0 passed completely.  This is
substantive evidence that removing the query-parallel history write improves
candidate-relative personalization.

It is not yet a finished proposed system.  One of three preregistered hash folds
was -0.000288, so the all-fold condition failed and delayed-B controls remained
closed.  The negative fold persisted under post-terminal adapted-attention,
agreement-scaled, and unprojected reductions; it cannot honestly be attributed
to the one attention mismatch previously suspected.

The post-terminal diagnostic also exposed a fold-audit bug: using a different
bootstrap seed per variant changed the fold hash itself.  The scripts and all
external notes were corrected to use the formal fixed seed.  Formal C31/C32
reports used the correct implementation and are unchanged.  The locked C32
proposal retains its pre-outcome wording; this outcome note supersedes the
incorrect diagnostic fold statement without rewriting the lock.

Next work should not tune the failed fold or quietly replicate until passing.
Either run a separately justified architecture-level change—preferably moving
the validated tangent-write law into the Transformer residual stream—or stop at
C32 as a strong positive candidate pending an independently preregistered
confirmation and matched controls.
