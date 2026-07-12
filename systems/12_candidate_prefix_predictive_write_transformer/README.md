# C12 Candidate-Prefix Predictive-Write Transformer

Status: **rejected at the paper pre-implementation gate**.  The proposed
candidate-prefix likelihood ratio is algebraically distinct from C11, but its
load-bearing candidate-specific vocabulary normalization has unacceptable
full-system complexity under a standard LM decoder.  No model, runner, config,
manifest, GPU run, or real/dev/test/qrels access exists.

The paper audit is preserved because it identifies both the first predictive
quantity in this sequence that survives candidate centring and the exact
computation that makes it impractical.

- `proposal.md`: hypothetical information flow and structural contracts
- `symbolic_reduction_audit.md`: proof, witnesses, and degeneration conditions
- `neighbor_and_complexity_audit.md`: nearest neighbours and FLOP lower bound
- `minimal_falsifier_design.md`: probe that would be required if an exact,
  generic normalization strategy later passes a separate complexity gate
- `preimplementation_decision.md`: binding rejection
