# C74 pretrained token-level LM probe protocol

Status: pre-outcome execution protocol.  It is outside the immutable C74
design proposal and receives its own execution lock.

## Authorized question

Does the data-free-passing semantic-conservative relay remain mechanically
load-bearing and useful when its states come from a shared pretrained LM whose
last two token layers train end to end?

This is an exposed-fit formulation probe, not an independent result.  It uses
the exact C64 4,800/1,200 split of C26's already exposed fit role so that C64,
C66, and C74 have a matched representation boundary.  Validation labels are
not read during training or A0 scoring.  No fresh role, dev, test, or qrels may
be accessed.

## Token-level graph

- one shared BGE encoder processes query WordPieces, each history-item text,
  and each candidate text;
- the final two BGE Transformer layers are trainable for every mode;
- history items and candidates use content-token means as semantic carrier
  atoms; query WordPieces remain separate through both relay stages;
- learned low-rank maps affect only history-to-query and candidate-to-query
  attention logits plus chronology;
- primary history values and candidate energy use the same current BGE state,
  with no separate V/O/FFN/head coordinate;
- no history/query gives exact D2p and exact-repeat gives item-only.

## Modes and budget

Primary, coupled-value, pooled-semantic, and factual-semantic modes have the
same BGE, trainable layers, route parameters, optimizer, examples, sampled
candidates, batch order, one epoch, and final-checkpoint rule.  Three fixed
seeds run on physical GPUs 0/1/2.

## Staging

1. G0 is label-free and binds split/candidate hashes, design-gate hash,
   backbone layer boundary, equal parameters, gradients, raw-carrier identity,
   query masking, fallbacks, determinism, and candidate permutation.
2. After the execution lock, training may read only the 4,800 exposed fit
   labels.  All 1,200 validation scoring is label-free.
3. A0 requires finite/decreasing loss, every gradient group, primary/base,
   true/wrong, and primary/control order/Top-10 activity in every seed, score
   hashes, and exact numerical contracts.
4. Only an A0 pass may open the 1,200 exposed-fit validation labels.  A1 uses
   the shared metric implementation, registered candidate hash, paired
   bootstrap, fixed folds, all-seed signs, and all three controls.
5. Only a complete Kuai A1 pass can authorize the identical graph/settings on
   Amazon-C4.  Dev/test/qrels remain closed regardless.

No route rank, LM depth, temperature, scale, learning rate, epoch, loss, seed,
cohort, threshold, or mode rescue is permitted after the execution lock.
