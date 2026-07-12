# C59 outcome — exact mechanics pass, utility terminal

C59's sorted-float64 numeric-equivalence implementation passed every A0 check.
All four GPU shards had exact-zero determinism and candidate-permutation error;
the real no-history and repeat fallbacks were exact. Activity was identical to
C58: primary changed 1,193/1,200 complete orders and 857 Top-10 sets versus the
strong base, while wrong history and the history-axis reduction were both
load-bearing.

Only then were the 1,200 compact fit-holdout labels read. Mean NDCG@10 was:

| score | NDCG@10 |
|---|---:|
| strong base | 0.577405 |
| primary candidate+NULL budget | 0.507303 |
| history-axis | 0.508105 |
| pooled history | 0.507473 |
| no-NULL candidate budget | 0.507188 |
| raw query | 0.499178 |
| wrong history | 0.479485 |

Correct history is real but insufficient: primary beat wrong history by
`+0.027817`, 95% CI `[+0.012463,+0.043129]`, yet lost to the strong base by
`-0.070103`, CI `[-0.084088,-0.056205]`. Candidate+NULL paid no unique rent:
its differences from no-NULL, history-axis, and pooled were approximately zero
and did not satisfy the registered control gate.

Decision: close fixed semantic candidate-budget attention and do not tune its
scale, NULL, temperature, token interaction, or mixture weight. The new
architectural lesson is that a history branch cannot emit an independently
standardized reranking score. Any future direction must change the residual
write contract itself and prove that evidence is ranking-aligned relative to
the strong base, rather than merely history-specific.
