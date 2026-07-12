# C74 proposal — Semantic-Conservative Query-Relay Transformer

Status: pre-outcome design formulation.  No C74 trained-model outcome,
repository record, label, qrel, dev, or test input has been observed.

## Observation → architecture consequence → falsification

**Observation.** C73 established that routing history through current-query
tokens beats late direct and factual-only attention on a held-out nuisance
shift, but its independently learned values, output map, and head produced one
near-base seed and paid too little rent over pooled transport.  Earlier C39--C41
localized a related failure: learned `V/O/FFN` maps can erase true-history
specificity, whereas raw LM-semantic carriers remain stable.  Full metric
coupling is not the answer—C40's raw-value `selection_only` control beat it.

**Architecture consequence.** A shared LM yields normalized query tokens `q_t`,
history-event tokens `h_j`, and candidate tokens `c_i`.  Two low-rank residual
maps are allowed to influence attention logits only:

```text
a_tj = softmax_j(<R_1(q_t), R_1(h_j)>/tau + chronology_j)
p_t  = sum_j a_tj h_j
q^H_t = normalize(q_t + p_t),        q^0_t = q_t

b^H_it = softmax_t(<R_2(c_i), R_2(q^H_t)>/tau)
b^0_it = softmax_t(<R_2(c_i), R_2(q^0_t)>/tau)

e^H_i = sum_t b^H_it <c_i,q^H_t>
e^0_i = sum_t b^0_it <c_i,q^0_t>
delta_i = center_candidates(3*tanh(e^H_i-e^0_i)).
```

`R_1/R_2` and chronology are trainable routing coordinates.  The carried
history values, transported query values, candidate values, and final energy
remain in one immutable LM-semantic coordinate; no learned `W_V`, `W_O`, FFN,
or scalar head can rewrite admitted evidence.  This is one nested attention
operator, not two scorers or an output router.

Empty history or absent query skips the write exactly.  Repeat-present uses the
registered item-only score exactly.  The full pretrained version trains LM
token layers jointly, so "immutable" means no *separate* value/readout
coordinate: the same current LM state is used at all carrier sites.

**Falsification.** On a fresh generator seed, all three GPU seeds must preserve
fallbacks, reject wrong/shuffled/coarse/query-masked evidence, and beat three
equal-parameter reductions:

1. `coupled_value_relay`: routing maps also rewrite values/readout (C40/C42);
2. `pooled_semantic_relay`: history values are pooled before candidate relay
   (C31/C41 family);
3. `factual_semantic_relay`: raw factual energy without internal NULL
   subtraction.

Failure closes C74 without repository data.  Passing authorizes only a new,
separately locked pretrained token-level probe with closed validation labels.

## Pre-outcome formulation evidence

After C73 terminated, one zero-training, one-temperature raw-semantic formula
was evaluated on C73's already-open synthetic validation surface.  It improved
NDCG@10 from `0.719691` to `0.908285`, while wrong and coarse histories returned
near base.  Event shuffle retained the full gain, so the formula is explicitly
insufficient.  This diagnostic fixes C74's hypothesis—learn routing and
chronology while conserving semantic values—but is not a C74 outcome and is
excluded from its fresh generator seed.

## Why this is not dataset tuning

The graph reads only query/history/candidate tokens, order, and presence masks.
It has no dataset ID, category, user table, historical slate, query type, or
score threshold.  Chronology is the common strictly-prior sequence coordinate.

## Predicted failure modes

- Learned routing alone may still overfit the nuisance event.
- Raw semantic values may be too rigid once the pretrained LM sees real item
  text rather than the constructed associative task.
- The pooled semantic reduction may retain all useful information.
- Chronology may become a fixed recency shortcut rather than event-specific
  evidence and fail shuffled-history specificity.
- Coupled learned values may match or beat conserved values, repeating C42.

No rank, temperature, chronology, scale, steps, seed, generator, or threshold
rescue is authorized after lock.
