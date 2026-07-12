# Nearest-neighbour audit

Only primary papers and exact project-internal mechanism records are used.

| neighbour | primary mechanism | C16 boundary |
|---|---|---|
| [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need) | softmax score allocation followed by a value read | linear C16 is the constrained case whose values are tied to score gradients |
| [Hopfield Networks Is All You Need](https://openreview.net/forum?id=tL89RnzIiCd) | continuous modern-Hopfield retrieval is equivalent to Transformer attention | the candidate gradient of log-sum-exp similarity is exactly this retrieval update |
| [Energy Transformer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html) | Transformer token updates designed to minimize an engineered interaction energy | nonlinear conservative candidate/history writes occupy the same energy-gradient family; restricting interactions to cross-partition edges gives the bipartite case |
| [Hopfield--Fenchel--Young Networks](https://www.jmlr.org/papers/v26/24-1961.html) | generalized associative-memory energies, entropies, sparse transformations, and post-transformations | changing the retrieval regularizer or energy normalization is already an explicit design axis |
| [Object-Centric Learning with Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) | inputs allocate competitively over exchangeable slots, which then aggregate inputs | treating candidates as slots and history events as inputs covers candidate-axis competition |
| [Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) | signed attention formed from the difference of two softmax maps | `softmax(s)-softmax(0)` is its fixed-uniform special case |
| [ZeroS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1363163299a172662dcf0c0f9932acf6-Abstract-Conference.html) | removes the uniform zero-order softmax term and reweights zero-sum residuals | softmax-uniform removal is the stated core operation, not a new C16 energy insight |
| C06 conservative wedge flow | a constrained conservative candidate write with local trust | already showed that conservativity and safety can coexist but do not guarantee a load-bearing write |
| C15 candidate-conditioned value write | pair-specific value directions and the generic edge-conditioned escape | a non-conservative mixed-Hessian survivor returns to the generic pairwise-vector family closed there |

The closest exact neighbour depends on the branch, but all branches are covered:
linear gradients by modern Hopfield, nonlinear conservative fields by ET/HFY,
candidate competition by Slot Attention or a bipartite ET restriction, and
uniform removal by Differential Transformer/ZeroS.  Combining those known axes
does not supply one new falsifiable primitive.

“Bipartite ET” is used descriptively for an Energy Transformer energy containing
only candidate--history cross-partition interactions.  The audit does not cite
or imply a separate publication with that title.
