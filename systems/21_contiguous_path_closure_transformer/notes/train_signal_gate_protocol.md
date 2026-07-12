# C21 train-only path-signal observability protocol

Status: **pre-outcome draft; binding after the label-free selection, source,
configuration and tests are hash-locked before any C21 optimizer step or probe
label metric is computed**.

This is a signal gate, not full proposed-system training.  It decides whether a
real-data premise is strong enough to justify a Transformer architecture.

## Frozen question

On non-repeat requests, does a short contiguous and directed history path carry
candidate-ranking signal that cannot be explained by the same frozen query,
candidate and history states with simpler operators?

For projected states `z`, C21 forms the candidate relation
`r_j = z(candidate_j) - z(query)`.  For every valid history segment with
`1 <= b-a <= 3`, it forms `p_ab = z(history_b) - z(history_a)`.  The primary
evidence is a soft maximum of a directional closure term between `r_j` and
`p_ab`, plus query-to-start and candidate-to-end anchors.  It writes only a
bounded candidate-centred residual onto the frozen D2p score.  No category,
action, recency, popularity, dataset ID or request-type feature is available.

## Label barrier and split

The only source cohort is the 12,000-request `fit` role frozen by C06 G0.  C21
first verifies the registered C06 selection/G0 hashes and exact fit-index
alignment.  Without opening `fit_labels.npy` or any label source, it sorts the
fit requests by:

```text
sha256("c21-path-signal-v1\0" + request_id), then packed request index
```

The first 9,000 become `train_fit`; the remaining 3,000 become
`internal_probe`.  The durable selection records request IDs, packed indices,
candidate-key hashes, source hashes, disjointness and explicit zero overlap
with every C06 non-fit role.  Only after the complete implementation lock may
the runner open C06's compact fit-label artifact for those two subsets.

Forbidden throughout: the original train label array, C06 `internal_A`,
`internal_B`, `escrow`, qrels, dev/test records, evaluator outputs and paper
metrics.  The C06 512-request no-history feature role may be scored without
labels solely for the exact base-equivalence contract.

## Operators and matched controls

Every mode has identical parameter names, shapes, initialization, optimizer
steps and full candidate sets.  Only the evidence algebra changes:

- `contiguous_path`: directed segments of lengths 1--3, directional closure,
  and correctly oriented start/end anchors;
- `one_step`: the same equation and parameters, restricted to adjacent events;
- `unordered_pair`: the same segments but direction is removed by symmetric
  closure and the better of the two endpoint orientations;
- `endpoint_only`: keeps oriented start/end anchors but removes relation
  closure;
- `pooled_history`: an order-free candidate-conditioned soft pool over single
  history events using the same projected states.

These controls distinguish multi-step closure, direction, the closure term and
ordinary history pooling.  A control win cannot be rescued by calling the
primary more novel.

## G0 structural and information-barrier gate

Before lock, unit tests use synthetic tensors only and must establish:

1. a hand-computed directed three-point path ranks the closing candidate above
   its reverse, while `unordered_pair` cannot distinguish them;
2. a length-two closure witness changes `contiguous_path` evidence relative to
   the otherwise identical `one_step` operator;
3. event shuffling changes primary evidence, while pooled-history evidence is
   permutation invariant;
4. candidate permutation equivariance and candidate-centred residual sum;
5. no-history and query-absent rows return the supplied base bitwise;
6. all modes expose identical parameters and identical initial state;
7. finite gradients reach state, relation and anchor projections plus the
   residual write;
8. the selection is exactly 9,000/3,000, disjoint, a subset of C06 fit, and has
   zero overlap with C06 non-fit roles;
9. every configured/input path passes the forbidden-path firewall; and
10. no code path accepts the original label array, qrels, dev/test records or a
    repository evaluator result.

## G1 one-shot GPU gate

- physical GPU 1, visible as `cuda:0`, deterministic CUDA;
- seeds `20260727`, `20260728`, `20260729`;
- exactly two epochs per mode, fixed final state, AdamW `1e-3`, weight decay
  `1e-4`, no candidate sampling or corruption training;
- projection width 32, latest 20 events, maximum path horizon 3;
- full-candidate masked listwise click loss;
- the same request order and dynamic batch boundaries for all modes of a seed;
- one learned attempt, no retry, sweep, early stopping or result-conditioned
  implementation repair.

The internal probe is scored cleanly and under two deterministic interventions:

- `wrong_history`: a hash-fixed donor permutation with no self donor, matched
  exactly on clipped history length so evidence availability cannot change;
- `shuffled_event`: an independently hash-fixed non-identity permutation of
  each history with at least two events.

`query_absent` is a structural intervention and must return D2p bitwise.  No
coarse-category proxy is fabricated from D2 states; therefore this gate does
not satisfy or replace the eventual architecture's required coarse-only test.

## Frozen decision rule

Metrics use request-equal NDCG@10 and the shared repository tie break.  For
comparisons across three seeds, per-request NDCG is averaged over seeds before
10,000-draw paired bootstrap.  C21 passes only if all checks hold:

1. `contiguous_path - D2p` mean is at least `+0.001`, its 95% bootstrap lower
   bound is above zero, and every seed has positive mean difference;
2. it exceeds every independently trained control by at least `+0.0005`, every
   paired lower bound is above zero, and every seed difference is positive;
3. each of three hash folds is positive versus D2p and every control;
4. wrong-history and shuffled-event each retain at most `0.25` of the clean
   NDCG gain over D2p; their bootstrap upper bounds may not exceed `0.50`;
5. at least 5% of probe requests change any candidate order and at least 1%
   change top-10 membership relative to D2p;
6. clicked-minus-unclicked score-delta bootstrap lower bound is above zero;
7. no-history and query-absent scores are bitwise D2p, deterministic rescore is
   bitwise equal, all values/gradients are finite, and matched parameter/init
   audits pass.

Any failure closes the real temporal-path signal hypothesis.  Passage permits
only a separately reviewed C21 Transformer formulation with repeat preservation,
coarse-only corruption, synthetic/real common contracts and nearest-neighbour
ablations; it does not authorize dev, test or full training.
