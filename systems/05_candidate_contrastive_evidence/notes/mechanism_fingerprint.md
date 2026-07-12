# C05 mechanism fingerprint

## Identity

- Candidate: `c05`
- Name: Candidate-Contrastive Evidence Budget Transformer (`CCEBT`)
- Primitive: Candidate-Contrastive Evidence Budget (`CCEB`)
- Novelty verdict before outcomes: **uncertain**
- Execution state: **parked until the simpler G2a/G2b signal gates pass**

## Exact operator

For each history event, a triadic query/candidate/event alignment `a_ij` is
computed inside a Transformer personalization block.  The operator then:

1. centers `a_ij` across valid candidates for the same event;
2. applies a symmetric dead zone, producing signed rather than simplex-only
   evidence;
3. divides by `1 + L1 mass`, normalizing attention mass (this does **not** yet
   bound the downstream hidden-state or score residual);
4. injects the resulting history value into the candidate residual stream with
   a zero-initialized bounded global scale.

The current exact item relation is only a positive alignment atom.  Pre-run
review showed that downstream signed/value/head maps can reverse its final
score direction, so it is not called protected.  Exact recurrence is excluded
from G2a and must later be reintroduced with action/recency semantics and a
monotone final-logit contract.

## Architecture locus

| Field | Frozen definition |
|---|---|
| Transformer intervention | replace one candidate-to-history attention/residual sublayer |
| Query state | contextual query hidden state from the same local LM/Transformer |
| Candidate state | contextual candidate hidden state before the last FFN/ranking layer |
| History state | strictly-prior event hidden states with padding/missing masks |
| Ranking path | updated candidate state -> standard Transformer FFN/layer norm -> shared ranking head |
| Inference inputs | true `(query, candidates, strictly-prior history, evidence masks)` only |
| Online LLM calls | zero |
| Dataset branches | none |

## Structural contracts

- `history_present=false` implies `Delta c=0` algebraically.
- Identical support for every valid candidate implies `d_ij=0` and therefore
  no update, even when raw query/history similarity is high.
- Candidate permutation permutes outputs identically.
- `sum_j |w_ij| < 1` for every candidate.
- Padded history and candidates contribute exactly zero.
- With one valid candidate, cross-item contrast is undefined in substance and
  the operator safely produces zero update.

## Training signal

Current G2a supervision is request-level ranking only:

```text
L_G2a = L_listwise(scores_true, labels)
```

Wrong-user, shuffle, query-mask, event-replacement, and coarse-only histories
are held-out G2b audits and are not used to train G2a.  Any later CCEB
counterfactual objective must be separately re-locked, and corruption families
used for training cannot also serve as the decisive evidence-authenticity gate.
Update norm, attention mass, or certificate separation remains diagnostic only.
Empty evidence subsets return a differentiable zero loss, never `mean(empty)`.

## Probe versus final system boundary

G2a does not use CCEB.  It wraps ordinary target attention around frozen
calibration-checkpoint states and adds a bounded score delta to a byte-aligned,
scope-correct D2p coordinate.  It answers only whether non-repeat transfer is
learnable.  It is an adapter-only falsifier and is not eligible as the final
proposed system.

If the gate passes, the final implementation must first build a query/item
Transformer base that observes the same text, identity, and legal train-only
popularity factors as D2p.  That base is frozen after parity; CCEB then remains
inside the candidate hidden-state path, and the final logit is emitted by the
same Transformer ranking head.

## Complexity

- Evidence logits: `O(B * C * H * d)`.
- No Sinkhorn/Newton solve, per-corruption inference pass, or second factual/null
  LM pass.
- Current probe targets one CCEB layer, one seed, and short history (`H <= 20`).

## Matched degenerations

| ID | Degeneration | What it tests |
|---|---|---|
| `target_softmax` | ordinary positive candidate-to-history softmax attention | reduction to DIN/target attention |
| `no_candidate_contrast` | dead-zone/L1 budget on raw alignment without candidate centering | reduction to Denoising Attention-like filtering |
| `unbounded_contrast` | candidate-centered signed update without the `1 + L1` budget | whether safety comes from the bounded operator |
| `history_only` | remove query/candidate terms from evidence alignment with matched parameters | whether request conditioning is load-bearing |

The first three use identical projections and ranking supervision.  Parameter
and optimizer-step parity must be asserted before comparison.
