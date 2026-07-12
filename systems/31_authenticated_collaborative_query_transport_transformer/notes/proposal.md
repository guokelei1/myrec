# C31 proposal: Authenticated Collaborative Query Transport Transformer

Status: pre-outcome draft; immutable after proposal lock.

## Hypothesis and primitive

C29 established reliable provenance but failed candidate direction because each
candidate received an independently learned scalar mediation score.  C31
instead learns one collaborative-semantic coordinate system and moves the query
once per request.

For frozen BGE embedding `x=LM(text)`, the only trainable representation change
is a shared rank-16 residual adapter:

```text
e(x) = normalize(x + U V x).
a_h  = softmax(cos(x_q, x_h) / 0.1), h in H_auth.
p    = sum_h a_h e(h).
q_H  = normalize(e(q) + p).
d_c  = 2 [cos(e(c), q_H) - cos(e(c), e(q))].
s_c  = z(D2p_c) + center_candidates(d_c).
```

D2p retains its registered fine-tuned query tower only for the baseline score.
The C31 transport path uses the untouched BGE snapshot for queries and the
matching frozen BGE item-title embeddings for history/candidates.  Candidate
computation is canonicalized by stable item ID before caller order is restored.

Thus all candidates share the same transported query.  There is no candidate
scalar head, candidate-specific adapter, router, user-ID score, or fixed
collaborative channel.  Full-request listwise loss and an equal-weight positive
versus negative correction-margin loss train `U,V` on all candidates.

No authenticated history or absent query gives exact D2p.  Exact candidate
recurrence gives the registered item-only score.  The BGE encoder is part of
the ranking model; exact cached item embeddings are only a frozen-LM execution
optimization, not offline features passed to an MLP.

## Evidence and novelty boundary

Post-terminal diagnostics on the already-open C30 A tested one rank, one
temperature, one profile scale, and one correction scale.  Three seeds produced
positive-CI NDCG gains around +0.004.  This authorizes only a new untouched
train-internal gate, not a result claim.

LLM2Rec learns collaborative-aware semantic embeddings; GenSAR uses dual-purpose
identifiers; ItemRAG combines semantic and co-purchase retrieval.  C31 does not
claim those ideas.  Its falsifiable difference is strict causal event admission
plus a single request-level query transport inside a shared LM space, with
exact identity and no per-candidate history direction.  C02's Cayley
hyperadapter was candidate-conditioned and failed responsiveness; C31 is one
low-rank shared coordinate update and one globally coherent query displacement.

## Stages

G0 must reproduce strong true/wrong authentication on untouched C31-A.  Phase
1 trains only the primary for three new seeds.  A0 checks activity, wrong
history, fallbacks, determinism, and candidate permutation without labels.  A1
requires mean gain >=0.001, positive CI, every seed, and every hash fold, plus
positive-CI true-over-wrong NDCG.  Average clicked direction is reported but is
not the primitive's gate: the query transport is judged at the top-ranked
request decision surface.

Only an A1 pass permits matched controls and delayed-B materialization.  Escrow,
dev, and test stay closed.
