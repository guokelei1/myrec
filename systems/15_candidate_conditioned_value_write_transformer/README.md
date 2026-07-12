# C15 Candidate-Conditioned Value-Write Transformer

Status: **rejected at the paper pre-implementation gate**.  No model, runner,
config, manifest, GPU run, or data access exists.

The constrained low-rank/bilinear value path factors through an ordinary
attention aggregate and becomes candidate FiLM/hyperadapter.  Adding the joint
nonlinearity required for truly event-specific directions turns it into a
standard dynamic-filter/edge-conditioned message function.  An unrestricted
pair MLP only adds capacity and compute; it does not create a bounded new
primitive.

- `algebraic_reduction_audit.md`: exact bilinear/post-pooling reduction
- `jacobian_and_witness_audit.md`: required witness and non-factorization test
- `nearest_neighbors.md`: primary-source mechanism boundary
- `complexity_and_controls.md`: active capacity/FLOP matching
- `preimplementation_decision.md`: binding rejection
