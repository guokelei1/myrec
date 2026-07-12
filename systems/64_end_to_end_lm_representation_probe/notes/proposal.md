# C64 proposal — end-to-end LM representation learnability

Status: pre-outcome prerequisite probe.  C64 claims no architectural novelty
and consumes no fresh outcome role.

## Why this probe is now necessary

C53--C61 built increasingly constrained ranking Transformers over frozen LM
states.  C62--C63 then showed, before repository data, that rank loss cannot
make a query-independent latent bottleneck discover addressable preferences,
even with Slot Attention, balanced transport, finite evidence, and NULL mass.
Continuing to invent read/write laws over the same frozen representations would
overfit the experiment history.

One material position remains insufficiently tested: **the pretrained token
representation itself**.  C05 closed one shallow frozen-state target attention
recipe, not end-to-end LM adaptation.  C53 used D2/BGE states as fixed input to
a new Transformer; C61 likewise froze contextual BGE tokens.  C64 therefore
asks whether adapting LM token layers changes the evidence boundary.

## Fixed probe architecture

The shared BGE encoder processes query, item, and history-item token sequences.
Its final two Transformer layers are trainable; all earlier pretrained layers
remain fixed.  Mean-pooled contextual token states enter the exact C53-style
directed joint-context ranker:

- query/history tokens cannot read candidates;
- every candidate reads query, history, and the full candidate set;
- candidates have no position/rank embedding;
- a centered candidate head writes a residual over the registered strong base;
- empty history in the primary is an exact base identity.

The LM encoder, joint Transformer, and ranking head are trained end to end from
the common listwise training labels.  Candidate sampling is training-only and
fixed before outcome; the held-out exposed-fit evaluation scores every
candidate and asserts the candidate-set hash.

## Binding modes

- `adaptive_history_lm` (primary): last two BGE layers trainable; true history
  enters the joint Transformer.
- `adaptive_query_candidate_lm`: identical trainable capacity and optimizer,
  but no history tokens enter the ranking path.
- `frozen_history_lm`: identical graph with all BGE layers frozen, isolating the
  rent paid by token-representation adaptation.

The primary checkpoint is also evaluated with matched wrong history.  The
registered strong base is never retrained or selected by C64.

## Falsification

On a deterministic 4,800/1,200 split of the already exposed C26 fit role, all
three seeds must:

1. improve over the strong base;
2. improve over both adaptive query-candidate and frozen-history controls;
3. score true history above matched wrong history;
4. materially change order and Top-10 under wrong-history replacement;
5. preserve exact empty-history/repeat fallbacks, candidate permutation,
   determinism, finite gradients, and candidate hashes.

Failure means frozen representations are not the only bottleneck and closes
this fixed last-two-layer joint-context probe.  There is no layer-count, LoRA,
history-length, candidate-sample, epoch, learning-rate, seed, or fresh-cohort
rescue.

Passing is only a representation-learnability signal.  It authorizes the exact
same exposed-fit probe on Amazon-C4; only a two-domain pass may motivate a new
internal Transformer primitive.  C26 internal-A, C39 reserve, dev, test, and
qrels remain closed throughout C64.

## Why this is not dataset tuning

The model reads only common query/item/history tokens and evidence masks.  It
has no dataset ID, category, query type, candidate-count, score threshold, or
top-k branch.  The choice to train the last two pretrained layers is frozen as
a low-cost representation test and may not be swept after outcome.
