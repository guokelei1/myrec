# C20 nearest-neighbor audit

The audit was completed before any C20 learned outcome.

| neighbour | established mechanism | C20 boundary |
|---|---|---|
| [OptNet](https://proceedings.mlr.press/v70/amos17a.html) | a quadratic program as an end-to-end differentiable layer | owns optimization-as-layer; C20's only possible delta is the request-local transition-cone evidence law |
| [Differentiable convex optimization layers](https://papers.nips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html) | general disciplined convex programs differentiated through a solver | prevents a broad “differentiable NNLS” novelty claim |
| [LISTA](https://icml.cc/Conferences/2010/papers/449.pdf) | fixed-depth neural approximation to sparse coding | owns unrolling; `relu1` and fixed-iteration controls test whether iterations pay rent |
| [Edge Transformer](https://papers.nips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html) | learned edge states with triangular relational attention | C20 has no persistent edge-token update; it constructs adjacent displacement columns and solves one candidate-local reconstruction |
| [relative-position attention](https://arxiv.org/abs/1803.02155) | relation representations inside attention | owns relation-aware attention broadly; transition differences alone are not novel |
| [induction heads](https://arxiv.org/abs/2209.11895) | retrieve a successor after matching a prior context | C19/forward retrieval is the nearest endpoint control; C20 tests positive multi-transition composition rather than copying one successor |
| C03 triadic transport | partial mass coupling with a null intersection | C20 has no transport marginals/dustbin and reconstructs a displacement under NNLS |
| C06 wedge flow | candidate-edge skew flow plus Hodge projection | C20 has no candidate graph or flow divergence |
| C18 order cone | isotonic projection of final score deltas | C20's cone is in hidden relation space and precedes the shared rank head |
| C19 oriented lag | diagonal plus forward-minus-reverse endpoint affinity | C20 represents full query-to-candidate displacement and forbids negative transition coefficients |

Verdict: **locally distinct, globally high risk, synthetic-only authorization**.
The solver, cone, relation vectors and Transformer are all known ingredients;
only a reproducible advantage over the frozen nearest controls could justify a
later real-data design.
