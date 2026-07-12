# C16 Mixed-Gradient Energy-Write Transformer

Status: **REJECTED at the paper pre-implementation gate.**  Do not implement
or run this candidate family.

C16 asks whether candidate-conditioned history evidence can be written through
the gradient or mixed Hessian of a scalar candidate/history energy.  The family
does not survive the novelty gate:

1. a linear candidate gradient is exactly cross-attention whose score and value
   maps are weight-tied, equivalently the modern-Hopfield retrieval update;
2. a nonlinear conservative field is still the gradient of a scalar energy and
   falls inside Energy Transformer / Hopfield--Fenchel--Young territory;
3. normalizing over candidates makes candidates compete for events, the
   allocation primitive already used by Slot Attention; restricting an Energy
   Transformer energy to candidate--history edges is the corresponding
   bipartite energy construction;
4. a mixed-Hessian contraction is either conservative and therefore the
   gradient of another scalar potential, or non-conservative and therefore
   cannot retain the proposed energy-descent interpretation; and
5. subtracting the softmax-uniform component remains an HFY-expressible scalar
   energy/regularizer choice, is a fixed-uniform Differential Transformer
   special case, and is addressed directly by ZeroS.

This directory is paper-only evidence.  It contains no model, runner, config,
frozen manifest, test outcome, checkpoint, GPU run, or data access.

- `mechanism_fingerprint.md`: the proposed family and required novelty witness
- `algebraic_reduction_audit.md`: exact gradient, Hessian, and centring reductions
- `nearest_neighbors.md`: primary-source mechanism boundary
- `complexity_and_controls.md`: compute consequences and mandatory controls
- `preimplementation_decision.md`: binding rejection
