# C28 nearest neighbours and reduction audit

| Neighbour | Overlap | Required distinction/control |
|---|---|---|
| PairRank and active pair sampling | prioritizing uncertain comparisons | C28 uses a fixed differentiable inference graph, not online exploration; uncertainty itself is not claimed new |
| weighted BTL / weighted pairwise LTR | nonuniform pair importance | fixed margin weighting is prior art territory; only the history-evidence interaction is eligible for later review |
| hard-negative and margin ranking | focus on close competitors | no sampled negatives or training-only heuristic; `uniform_evidence_contest` tests inference placement directly |
| PRM / SetRank / CMC / RAISE | listwise and personalized candidate interaction | generic/candidate controls must be beaten; candidate interaction and dynamic personalization are not new |
| C27 evidence contest | identical token evidence and odd comparator | exact uniform control isolates only the margin-local graph |
| rank-weighted/top-k objectives | emphasize evaluation boundary | C28 deliberately forbids rank or `k`; locality is continuous in standardized base gap |

Primary sources checked before implementation:

- https://arxiv.org/abs/2103.00368
- https://proceedings.mlr.press/v119/hendrickx20a.html
- https://proceedings.mlr.press/v202/lyu23b.html
- https://arxiv.org/abs/2005.10084
- https://arxiv.org/abs/1904.06813
- https://arxiv.org/abs/2201.05333

Verdict: margin-local pair emphasis is not independently novel.  C28 is a
mechanism gate with an outcome-isolated role and an exact C27 control.  Any
positive result still requires a deeper literature and contribution audit.
