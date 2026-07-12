# C46 signal-gate outcome

Status: terminal at A1; representation premise not established.

G0 opened only request indices `[0,40000)` and produced 42,371 strict
prefix-to-distinct-next-item examples with 40,685 unique targets. Three seeds
trained true-pair and deterministically shuffled-pair Transformers for 500
steps each. A0 passed every structural, gradient, determinism, permutation,
activity, hash, and label-isolation check before the 600 registered A labels
opened. Dev/test and qrels remained closed.

The true-pair model learned real source structure: seed-averaged NDCG@10 was
`0.301775`, versus `0.261857` for the equal-parameter shuffled-pair model.
The `+0.039918` difference had 95% CI `[+0.016970,+0.063089]` and every
seed/fold was positive.

That is not sufficient behavioral-representation rent:

- frozen semantic mean was `0.301470`; primary-minus-semantic was only
  `+0.000306`, CI `[-0.017184,+0.018029]`, with two negative seeds and one
  negative fold;
- true-minus-wrong history was `+0.013744`, above the point threshold but with
  CI `[-0.011262,+0.037652]`;
- primary clicked direction was positive, but its true-minus-wrong direction
  CI crossed zero;
- reversing history changed many complete orders but reduced NDCG by only
  `0.002693`.

Decision: close C46. A content-initialized sequential Transformer learns
source co-occurrence, but on this untouched cohort that learning does not add
stable information beyond raw semantic mean history. It cannot serve as the
premise for a dual-representation proposed architecture. A successor must
remove the semantic shortcut using a problem-defined information object, not
increase sequence depth, pretraining steps, or adapter width.

Promoted report SHA-256:
`5e8b4a29d63367559a9112e7d36a9239fe1f22a52b3d2ef021d8f227e0ef9727`.
