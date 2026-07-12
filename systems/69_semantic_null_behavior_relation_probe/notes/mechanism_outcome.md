# C69 mechanism outcome

Decision: `failed_signal_terminal`.

All six fixed GPU runs passed the label-free mechanical gate under execution
lock `ed5f622d4bddc20efa2c7f33ad1a6d8ef48180a96a709ad1cf0f875353cf7bc1`.
The semantically matched negatives were materially harder than the random
control: mean source--target cosine gaps were about `0.121` versus `0.325` on
KuaiSearch and `0.019` versus `0.094` on Amazon. Losses decreased, all
parameter groups had gradients, candidate hashes matched, permutation error
was at most `2.3842e-7`, and no-history/source-zero outputs were exactly zero.

The ranking signal nevertheless failed in both domains:

| Domain | Primary NDCG@10 | Semantic | Random-negative | True - wrong |
|---|---:|---:|---:|---:|
| KuaiSearch | 0.304455 | 0.307626 | 0.302754 | +0.015081, CI crosses zero |
| Amazon-C4 | 0.020831 | 0.192450 | 0.129216 | +0.001277, CI crosses zero |

KuaiSearch primary-minus-semantic was `-0.003171`; primary-minus-random was
only `+0.001701` with a zero-crossing interval and mixed folds/seeds. Amazon
primary-minus-semantic was `-0.171619` and primary-minus-random was
`-0.108386`, both significantly negative. Amazon clicked direction was also
significantly negative (`-0.128886`, 95% interval
`[-0.164957,-0.092397]`).

Thus adjacent-event compatibility learned against semantic-matched negatives
is not the missing relevance-aligned personalization relation. Hard-negative
training successfully removed the easy semantic shortcut, but what remained
was unstable on KuaiSearch and directionally wrong on Amazon. No negative
cost, temperature, width, step, aggregation, seed, or cohort rescue is
authorized.

The first formal Kuai invocation had stopped before store construction because
of a package-name import collision. It produced no checkpoint, score, label,
or scientific outcome. The original lock and the strictly mechanical import
amendment are preserved in the notes directory.

Authoritative report:
`reports/pps_c69_semantic_null_behavior_relation_gate.json`, SHA-256
`4d084107f64c012c7172c97c6eb14d01316ce5b1092062b7c957a03a063c0157`.
Fresh reserve, dev, test, and qrels were not opened.
