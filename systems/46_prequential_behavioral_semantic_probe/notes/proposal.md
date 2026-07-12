# C46 proposal — prequential behavioral-semantic signal gate

## Observation

C43--C45 progressively ruled out metric coupling, candidate-axis flow, NULL
mass, output space, and prefix-conditioned factual-minus-NULL state updates as
the missing cross-item primitive. Every route transformed the same frozen LM
semantic item states. The unresolved alternative is that those states omit the
user-transition relation itself.

## Probe consequence

C46 trains a compact Transformer on sequences of frozen LM item-title states.
For a user prefix `(h_1,...,h_t)`, a learned `[READ]` token attends to projected
item states and predicts the next distinct clicked item against sampled source
items. The target remains represented by the same content projection, so every
outcome candidate has a representation even if its item ID never appeared in
source training.

The source contains only requests with index below 40,000. C46-A contains 600
strict-nonrepeat requests with index at least 50,000 inherited exclusively from
unopened delayed/escrow roles after removing every materialized role. Thus no
current A outcome can enter the behavioral Transformer or its vocabulary.

## Falsification

The learned representation must simultaneously:

1. rank C46-A clicked candidates better under true than matched wrong-user
   history with a positive paired interval;
2. beat an equal-parameter Transformer trained after deterministically
   shuffling source prefix/target pairing;
3. beat frozen semantic mean-history similarity;
4. give a positive clicked-minus-unclicked direction;
5. remain deterministic, candidate-permutation equivariant, and exactly zero
   with no history.

Failure closes behavioral-semantic representation as the next architecture
premise. Passage authorizes formulation of a separate query-conditioned dual-
representation Transformer; it does not make C46 itself a proposed system.

## Claim boundary

Content-initialized sequential Transformers are established by UniSRec,
Recformer, S3-Rec and related work. C46 claims no novelty and cannot enter the
paper as the proposed architecture. It is a leakage-safe signal instrument
that prevents another architectural search over a missing information source.
