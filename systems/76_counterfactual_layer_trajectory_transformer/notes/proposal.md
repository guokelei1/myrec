# C76 proposal — Counterfactual Layer-Trajectory Transformer

Status: pre-outcome architecture formulation.  No C76 outcome, validation,
dev, test, or qrel has been observed.

## Observation → architecture consequence → falsification

**Observation.** Pooled full/text/ID histories did not expose a practical
strict-nonrepeat source on KuaiSearch or Amazon.  In contrast, an ordinary
full-token BGE cross-encoder produced a three-seed, user-disjoint Amazon
reserve gain of `true-null +0.02530` and `true-wrong +0.03594`.  Weight-frozen
edge interventions then showed that removing candidate-history edges destroys
the gain, removing query-history edges leaves only 37.6%, and preventing
history tokens from reading query/candidate context also destroys utility.
The information object is therefore the *multi-layer joint token trajectory*,
not an input item vector or a one-way relay.

**Architecture consequence.** For candidate `c_i`, form the raw WordPiece
sequence `[CLS] q [SEP] c_i [SEP] H`.  One adaptive LM with shared parameters
is evaluated twice on exactly the same token IDs and positions.  Stochastic
dropout is disabled in this paired path (or must share an identical mask), so
the trajectory cannot encode random branch noise:

```text
F_i^l = LM_theta^l(tokens; all Q-H-C attention edges)
N_i^l = LM_theta^l(tokens; H isolated from Q and C)
D_i^l = (F_i^l - N_i^l) /
        sqrt(mean((F_i^l)^2 + (N_i^l)^2)/2 + eps).
```

The mean in the denominator is the per-token hidden-coordinate mean.  It is a
factual/cut carrier scale, not the norm of the small difference, so numerical
dust is not amplified.  For every layer, masked means
of `D_i^l` over query/readout, candidate, and history WordPieces form three
trajectory tokens.  A compact causal-free Transformer reads the ordered
layer-by-segment ledger.  Its residual is anchored on the final candidate
delta token, so disabling earlier tokens gives the registered final-state
degeneration rather than an unrelated head.

A separately protected query-candidate LM coordinate supplies `base_i`; it is
fit before personalization and receives no personalized gradient.  The final
candidate-set score is

```text
delta_i = rho * tanh(head(TrajTransformer(D_i^1,...,D_i^L)))
score_i = base_i + center_candidates(delta_i).
```

When history is absent, the trajectory path is structurally skipped and the
base is returned exactly.  Exact candidate recurrence is a universal identity
relation, not a dataset rule: repeat candidates retain the registered
item-only coordinate and semantic trajectory writes cannot lower it.  Missing
query/text/history is expressed only by masks.

The primitive is the **counterfactual layer trajectory**: the ranker preserves
the complete factual token graph, while the personalized readout is restricted
to the depth-resolved intervention response of that graph.  It is neither a
second factual scorer nor a final factual-minus-null scalar.

**Falsification.** C76 first fails data-free unless full/cut identity,
no-history exactness, candidate permutation, active gradients, nonzero
multi-layer trajectory, and recurrence safety all pass.  On the frozen
triadic-token shift, it must learn supported nonrepeat relations, reject
wrong-user and query-masked histories, remain insensitive to an order-only
permutation, and be distinct from all matched controls.  A pass authorizes one
fresh, separately locked real fit probe.  Real utility must beat the protected
base and the following equal-capacity reductions with positive interval and
all-seed evidence:

1. ordinary factual full-token cross-encoder;
2. final factual/cut logit difference (C04 family);
3. final factual/cut hidden-state difference (C65-C66 family);
4. layerwise factual-state side network without subtraction.

Failure closes the counterfactual depth-trajectory primitive.  No layer set,
normalizer, trajectory width, correction scale, loss, seed, epoch, or cohort
rescue is allowed.

## Scope and efficiency

The graph has no dataset, category, query type, popularity threshold, or
handcrafted semantic branch.  It uses one local pretrained LM and one compact
trajectory Transformer; online external LLM calls are zero.  The paired LM
forward approximately doubles adaptive-branch compute, so latency and a
single-final-layer degeneration are binding measurements, not optional
appendix details.
