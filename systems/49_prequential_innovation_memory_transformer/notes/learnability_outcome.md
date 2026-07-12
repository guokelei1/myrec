# C49 exposed learnability outcome

All six GPU predictors passed the label-free A0 contract.  Kuai transition
loss fell from about 3.0 to 1.58 over 40,923 examples; Amazon loss fell from
about 4.60 to 4.23 over 176,413 examples.  Every parameter group trained,
scores were deterministic and candidate-equivariant, no-history was exact
zero, and the innovation path changed 69%--70% of Kuai and 99.7% of Amazon
rankings relative to raw KRR.  C49 labels were read only after A0.  Fresh
reserve, dev/test, and qrels remained closed.

| domain | primary | vs base | vs raw KRR | vs DeltaNet | true vs wrong |
|---|---:|---:|---:|---:|---:|
| KuaiSearch | 0.309062 | +0.008191, CI crosses 0 | +0.000754, CI crosses | +0.000584, CI crosses | +0.005023, CI crosses |
| Amazon-C4 | 0.221624 | -0.031579, CI entirely negative | -0.053089, CI entirely negative | +0.049506, CI positive because DeltaNet is worse | -0.022824, wrong history nominally better |

Amazon clicked correction was positive in isolation, but clicked
true-minus-wrong was significantly negative.  Prequential errors therefore
carry large activity without a reliable user-specific relevance direction.

Decision: `failed_exposed_learnability_terminal`.  Close C49 before fresh
reserve.  Do not change steps, scale, ridge, predictor width, residual sign, or
select the favorable Kuai surface as a rescue.
