# C24 nearest neighbors and reduction decision

| Neighbor | Relation | Decision |
|---|---|---|
| Context-Aware Learning to Rank with Self-Attention (Pobrotyn et al., 2020) | candidate/document interactions are encoded with self-attention | C24 primary is a domain-specific signal probe, not novel over this family |
| DLCM (Ai et al., SIGIR 2018) | refines an initial ranking from local list context | registered as a strong conceptual predecessor |
| SetRank / Set Transformer rankers | permutation-equivariant setwise candidate encoding | candidate-set processing itself cannot carry novelty |
| DirectRanker / RankNet | antisymmetric pairwise preference functions | rejected an antisymmetric tournament as final primitive; it is generic pairwise LTR |
| C06 conservative Hodge flow | zero-sum candidate-relative write | C24 does not reuse its event-flow/cycle-trust mechanism |
| C23 RRST | independent candidate reset/suffix evolution | C24 changes the information object from temporal suffix to co-present repeated-candidate competition |

Primary sources checked before implementation:

- https://arxiv.org/abs/2005.10084
- https://arxiv.org/abs/1804.05936
- https://arxiv.org/abs/1909.02768
- https://arxiv.org/abs/2002.09841

Reduction verdict: a generic candidate-set/tournament architecture is known.
C24 is authorized only as a matched-control **signal existence gate**.  It may
not be promoted as a proposed-system contribution.  If it passes, the next
candidate must introduce and ablate a recurrence-specific information law that
cannot be reduced to ordinary candidate self-attention.
