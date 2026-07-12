# C17 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

The proposed evidence ledger does not survive the mechanism-innovation gate:

1. a free persistent `candidate × event` tensor is an Edge Transformer or
   generic edge-state message-passing network;
2. an exact chain-rule ledger is forward-mode attribution and leaves the
   ranking function unchanged;
3. feeding that attribution back into values or scores is an ordinary gate or
   joint pair message;
4. antisymmetric margin readout by divergence returns to C06; and
5. keeping cyclic pairwise state returns to higher-order edge/tournament
   ranking rather than supplying a ledger-specific primitive.

The safety properties motivating C17 remain requirements for future systems,
but they do not distinguish this operator from its nearest neighbours.  No
source model, config, lock, synthetic probe, repository data, GPU, checkpoint,
dev evaluator, test evaluator, qrels, or label access is authorized or present.

The next candidate must change the ranking operator itself while preserving a
strong query-only anchor and exact recurrence, rather than adding another
learned history state that is later pooled into a score.
