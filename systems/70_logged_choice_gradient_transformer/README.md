# C70 Logged-Choice Gradient Transformer

C70 is a pre-outcome architecture formulation motivated by C69's failure to
learn ranking direction from positive-only adjacent item sequences. Its
history tokens are signed gradients of choices within historical candidate
slates, rather than embeddings of clicked/purchased items alone.

The current dual-domain data gate fails: KuaiSearch has excellent recoverable
logged-choice coverage, while Amazon-C4 has none under the frozen standardized
interface. JDsearch is not locally downloaded and its published row format
does not include historical candidate slates. No C70 model, GPU training, or
outcome evaluation is authorized yet.

See:

- [notes/proposal.md](notes/proposal.md)
- [notes/coverage_audit.md](notes/coverage_audit.md)
- [notes/nearest_neighbors.md](notes/nearest_neighbors.md)
- [notes/preimplementation_review.md](notes/preimplementation_review.md)
