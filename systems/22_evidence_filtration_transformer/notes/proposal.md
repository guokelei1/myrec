# C22 proposal — Evidence-Filtration Transformer

## Observation → consequence → falsification

**Observation.** Exact candidate recurrence is the only stable history
component in the current benchmark.  C18 shows that protecting it only after
the score proposal cannot create transfer, while C21 shows that an active
multi-step write over frozen states can alter many rankings without a useful
direction.  Standard residual mixing and normalization give reliable and
speculative evidence equal authority at every layer.

**Architecture consequence.** Represent evidence reliability as a filtration
of each candidate residual state, and require every Transformer block to
preserve it.  Reliable coordinates are not a scalar gate or an auxiliary
feature; they form an invariant prefix subrepresentation that later computation
can read but cannot overwrite.

**Falsification.** C22 fails unless a learned, matched-capacity, three-seed
synthetic gate simultaneously preserves exact recurrence, learns supported
non-repeat transfer, rejects hard corruptions, returns the base exactly without
history, and beats dense mixing, uncoupled parallel streams, and a C18-like
final safety projection on worst-stratum utility.

## Single primitive

For every token and layer, split the residual state as

```text
x = [x_anchor | x_recur | x_transfer]
F0 = span(x_anchor)
F1 = span(x_anchor, x_recur)
F2 = full state.
```

Each residual map `M` obeys `M(F_k) subset F_k`; in row-vector convention its
weight is block lower triangular.  Attention heads align with block boundaries,
FFN projections use the same mask, and ordinary RMSNorm is replaced by prefix
RMS normalization so `x_transfer` cannot change the scale of `F0/F1`.

The query/candidate anchor initializes `F0`.  A symbolic, dataset-independent
candidate/history identity equality relation plus event metadata writes a
nonnegative recurrence atom into `F1/F0`.  Content-based history enters only
`F2/F1`.  Because maps preserve the filtration, transfer may condition on the
recurrence computation but the recurrence state never depends on transfer.

The candidate logit is produced from the final candidate token.  The recurrence
coordinate has a nonnegative readout coefficient; the transfer readout is
candidate-centred and bounded.  No-history bypasses both personalized quotients
and returns the supplied anchor score bitwise in the minimal falsifier.  A later
full model, if authorized, must internalize that anchor in the LM ranking core.

## What is and is not being claimed

The architectural claim is the **causal one-way evidence order**, not multiple
heads, exact matching, monotone weights, block masks, or prefix normalization in
isolation.  The model has one loss and one final ranking logit.  It is not a
fixed-score ensemble, request-type classifier, post-hoc router, ordinary
candidate attention, hyperadapter, or generic pair MLP.

Generic block-triangular Transformers can instantiate the same algebra.  Thus
global operator novelty remains uncertain before empirical rent is paid.  A
synthetic pass would authorize only a separate real train-internal probe; a
failure closes C22.
