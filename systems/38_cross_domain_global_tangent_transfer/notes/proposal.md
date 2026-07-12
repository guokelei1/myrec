# C38 cross-domain global tangent transfer proposal

## Bounded observation

C37 passed all 34 label-free structural checks.  Its primary was nominally
`+0.001673` NDCG@10 over D2p, but the confidence interval crossed zero and it
was effectively tied with both candidate-shared global tangent transport and
uncentered additive transport.  Candidate-axis conservation therefore paid no
utility rent.  The only weakly surviving component is a shared authenticated
history write, and even that is not validated.

Continuing to modify candidate geometry on KuaiSearch would condition the
research program on serial observations from one dataset.  C38 instead moves
the unchanged shared operator to an independent language, candidate
construction, history source, and Transformer representation.

## Single hypothesis

On Amazon-C4, a true user's released strict-past history supplies a
query-relevant direction that is useful after removing the component parallel
to the current query.  Formally, for normalized adapted states

```text
q = A(query)
h_j = A(history_j)
c_i = A(candidate_i)
a_j = softmax(<q,h_j>/tau)
p = sum_j a_j h_j
g = p - <p,q>q
q' = normalize(q + g)
Delta_i = 2(<c_i,q'> - <c_i,q>)
```

where `A(x)=normalize(x+W_up W_down x)`, rank is 16, `tau=0.1`, and the
coefficient `2` is carried unchanged from C32--C37.  There is no learned gate,
candidate-specific parameter, category rule, or Amazon-specific branch.

The mechanism is falsified if it cannot beat the frozen query-candidate base,
if matched wrong-user history reproduces its gain, or if its tangent/query-
attended components do not pay rent against equal-parameter reductions.

## Architecture boundary

- Backbone: frozen `BAAI/bge-small-en-v1.5`; its Transformer hidden states are
  the ranking state space and the low-rank adapter is trained end to end for
  ranking.
- Candidate pool: official Amazon-C4 sampled-1M catalog, SQLite FTS5 BM25
  top-100 plus the positive.  To bound long-query retrieval cost, the query
  uses at most eight unique terms with the lowest catalog document frequency;
  the DF table and candidate lists are fixed before model work.
- History: upstream temporal-cutoff release, globally timestamp-sorted and
  truncated to the most recent 50 events during standardization.
- Base score: frozen BGE query-candidate cosine plus the fixed exact-recurrence
  component carried from the validated item-only contract; no popularity or
  KuaiSearch feature is available.  If any candidate exactly recurs in true
  history, every transport mode rejects the cross-item write for that request.
- Primary: query-attended tangent transport above.
- Equal-parameter reductions: query-attended unprojected transport and
  mean-history tangent transport.
- Causal controls: true history, history-length-bin-matched wrong user, and
  exact no-history fallback.  Target category is unavailable to matching.

## Decision

A pass authorizes a new architecture formulation around relevance-aligned
Transformer value learning, using the transferred shared write as its public
anchor.  It does not promote C38 itself as the proposed system.  A failure
closes the C31--C38 transport lineage and redirects the next architecture to
the Transformer's representation/value-learning interface rather than another
geometric gate.
