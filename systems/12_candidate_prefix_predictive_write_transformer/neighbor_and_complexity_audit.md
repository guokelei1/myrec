# Nearest-neighbour and complexity audit

## Required controls

Any future implementation would need, from the same initialization and with the
same shared LM/backbone:

| control | load-bearing difference |
|---|---|
| prefix hidden similarity | drops only `DeltaA`; directly tests whether normalized prediction pays rent |
| candidate-independent event LLR | removes candidate prefix; reproduces the rejected C11 boundary |
| pooled prefix LLR | pools history before likelihood comparison; tests event preservation |
| centred cross-attention | candidate query over event-only keys/values, with matched late capacity |
| paired scalar logit | sums token/event likelihood ratio and adds it after the base head |
| NULL/target-leak canary | exposes target token or changes sequence length; must look artificially strong and is never a valid model |

Parameter matching is insufficient: the prefix-hidden control must receive the
same candidate/event prefix states, and compute matching must include the full
decoder normalization as dummy work or report the true efficiency gap.

## Load-bearing complexity

Let `C` be candidates, `H` events, `T` predicted candidate tokens, `V` decoder
vocabulary, and `d` hidden width.  The NULL prefix is cached across events, but
exact likelihood still requires at least

```text
C (H+1) T
```

candidate-prefix decoder states per request and

```text
Omega(C (H+1) T V d)
```

multiply-accumulates for exact full-vocabulary normalization.  KV caching saves
re-encoding query/event prefixes; it does not remove this decoder lower bound.
The late LM block also pays approximately `O(CHT d^2)` FFN work and attention
over cached query/event tokens.

Illustrative decoder-only lower bounds:

| C,H,T,V,d | MACs/request |
|---|---:|
| 7,8,3,333,32 (tiny synthetic) | 2,013,984 |
| 7,8,3,32,768,256 | 1,585,446,912 |
| 50,6,3,32,768,256 | 8,808,038,400 |
| 100,20,8,32,768,768 | 422,785,843,200 |

These numbers exclude the Transformer and late integrator.  Ordinary centred
attention's candidate/event interaction is `O(CHd)` after encoding; C12's exact
normalizer is not a constant-factor replacement.

## Why the expensive term cannot simply be removed

The symbolic witness shows that `P_C DeltaA` is the only term separating C12's
normalized likelihood ratio from prefix hidden similarity under a tied linear
decoder.  Sampled-softmax, a shared normalizer, or unnormalized target logits
may be cheaper, but unless separately proven candidate-specific and accurate,
they remove the claimed primitive.  Temperature scaling also cannot solve the
problem.

Hierarchical/adaptive softmax, a small learned product-token codebook, or binary
codes could reduce normalization cost.  Each changes tokenization/decoder
semantics and introduces another load-bearing design choice.  None is justified
by the current evidence, and treating it as a free optimization would violate
the one-primitive budget.

## Complexity decision

The tiny falsifier is computationally feasible, but the proposed architecture
is not currently a credible lightweight full-candidate LLM4Rec ranker.  Its
non-reducible term and its prohibitive term are the same exact normalizer.  This
fails the pre-implementation complexity rent, so no implementation is
authorized merely because a toy GPU probe would fit.
