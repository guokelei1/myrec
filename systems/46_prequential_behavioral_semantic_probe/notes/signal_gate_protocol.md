# C46 frozen signal-gate protocol

## Staging

1. The label-free selection is already frozen at SHA-256 `905b7de...`.
2. Proposal lock hashes source, config, tests, selection, and all provenance
   inputs before any source label row is read.
3. G0 may open click labels only for request indices `[0,40000)` and materialize
   strict prequential prefix/next-item examples. C46-A labels remain closed.
4. Three seeds train both `true_pairs` and `shuffled_pairs`, then score true,
   wrong, reversed, and semantic controls on C46-A without labels.
5. A0 must pass before the aggregate process opens exactly the registered 600
   train-internal A label rows once.

No dev/test record, qrel, shared evaluator, current A label during training, or
source row at index 40,000 or later is authorized.

## G0

- source max timestamp is strictly below outcome min timestamp;
- at least 5,000 non-recurrent prefix/next examples and 3,000 unique targets;
- all prefix items and targets index the registered LM item-state table;
- selection, candidate set, source-label array, item states and manifests match
  their frozen hashes;
- only source labels are opened.

## A0

- equal parameters and paired initialization across both training modes;
- all losses/gradients/scores/states finite and parameters updated;
- deterministic max difference `0`, candidate-permutation max `1e-6`;
- no-history behavioral score bit-exact zero;
- true/wrong histories and true-pair/shuffled-pair models each change at least
  5% of complete candidate orders in every seed;
- all score artifacts/checkpoints exist with matching hashes;
- A labels, dev/test and qrels remained closed during training/scoring.

## A1

Seed-averaged primary NDCG@10 must exceed:

- wrong history by at least `+0.005`, with CI low above zero and every seed and
  fixed fold positive;
- shuffled-pair training by at least `+0.005`, with the same sign rules;
- frozen semantic mean by at least `+0.002`, with the same sign rules.

Primary clicked-minus-unclicked score direction and its advantage over wrong
history must each have a positive CI. Any failure is terminal; no source
cutoff, example filter, width, loss, negative count, step, seed, or threshold
rescue is permitted.
