# C73 proposal — Counterfactual Query-Relay Transformer

Status: pre-outcome design formulation.  No C73 trained-model outcome,
repository record, label, qrel, dev, or test input has been observed.

## Observation → architecture consequence → falsification

**Observation.** C31--C43 repeatedly found a weak, cross-domain
query-transport direction, while C53--C66 showed that ordinary joint context
and final factual-minus-NULL candidate states learn a mostly generic
query--candidate reranker: changing the user's history rarely changes Top-10
decisions.  C69 additionally ruled out an item-only adjacent relation.  The
unresolved object is therefore not another history value or output residual;
it is a constrained *path* by which current-query semantics mediate history
before candidate scoring.

**Architecture consequence.** Let a shared LM produce query WordPiece states
`Q`, history-event states `H`, and candidate-token states `C_i`.  A query block
forms factual and structurally NULL trajectories with shared parameters:

```text
Q_H = QueryBlock(Q, H)
Q_0 = QueryBlock(Q, NULL).
```

There is no `H -> C` attention edge.  A shared candidate attention is evaluated
against both query trajectories, and only its internal difference is allowed
to enter the personalized residual stream:

```text
U_i^H = Attn_theta(query=C_i, key/value=Q_H)
U_i^0 = Attn_theta(query=C_i, key/value=Q_0)
R_i   = FFN_theta(U_i^H - U_i^0)
delta_i = center_candidates(head(R_i)).
```

The strong non-personalized score is an anchor coordinate, not a separately
routed scorer.  Empty history or absent query makes `Q_H == Q_0` and skips the
write exactly.  Repeat-present requests retain the registered item-only
fallback exactly.  For non-repeat requests the LM, query block, relay
attention, and ranking head train jointly.

The primitive is the **counterfactual query relay**: a path-specific,
candidate-conditioned attention *operator difference*.  It is not merely a
final hidden-state subtraction.  History must first modify token-resolved
query states, and candidates can read only the resulting change in their own
attention computation.

**Falsification.** Before repository data, three fixed GPU seeds must solve a
two-hop query/history/candidate associative-ranking task under a held-out
nuisance shift, preserve repeat and no-history behavior, reject wrong,
shuffled, coarse-value, and query-masked histories, and beat three
parameter-identical reductions:

1. late factual-minus-NULL candidate-state attention (C65/66 nearest);
2. pooled query relay (C31/32 nearest);
3. factual query relay without internal NULL subtraction.

Failure closes C73 without a pretrained-LM or real-data run.  Passing permits
only a separately locked pretrained token-level probe on exposed train labels
with validation labels still closed through its mechanical gate.

## Why this is not a dataset recipe

The operator consumes only query tokens, ordered prior-history tokens,
candidate tokens, and evidence-presence masks.  It has no dataset ID,
category rule, historical-slate field, user table, query type, score threshold,
or corpus-specific branch.  The identical graph is defined for KuaiSearch,
Amazon-C4, and JDsearch; missing text or history is represented by masks.

## Predicted failure modes

- Query tokens may become a generic bottleneck and lose useful event detail.
- The factual relay may match the counterfactual relay, showing that internal
  subtraction pays no rent.
- A late direct attention block may learn the same two-hop relation with no
  practical disadvantage.
- Wrong histories may still produce coherent query changes and retain utility.
- The synthetic inductive bias may pass while real non-repeat evidence remains
  too weak to move strong-base Top-10 margins.
- Two LM trajectories may double encoder cost without enough quality gain.

No layer-count, width, temperature, loss, nuisance strength, threshold, seed,
or generator rescue is authorized after the lock.
