# C26 — Query-Pivot Token-Bridge Transformer

C26 changes representation granularity after C23--C25 closed further operator
search on pooled D2 states.  A compact shared token Transformer preserves
WordPiece-level query, candidate-title and history-title states.  Candidate and
history tokens are independently aligned to each query token; only a
same-query-token agreement bridge may enter the personalized history
Transformer.

This is a train-only signal/architecture gate.  Generic token late interaction,
fine-grained history interaction and triple attention are established prior
art, so C26 has no novelty claim unless the restricted bridge beats all
registered controls.  Dev/test remain unauthorized.
