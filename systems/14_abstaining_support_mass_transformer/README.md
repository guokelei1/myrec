# C14 Abstaining Support-Mass Transformer

Status: **rejected at the paper pre-implementation gate**.  No model, runner,
config, manifest, GPU run, or data access exists.

The proposed subprobability attention weights `w=rho p` are exactly ordinary
attention over real events plus one zero-valued NULL token of mass `1-rho`.
The support/allocation Jacobian factorization is the same null-softmax Jacobian
written in radial/tangent coordinates.  An independent support head instead
reduces to per-head sigmoid output gating, already a direct Transformer
neighbour.  Exact sparsity moves the proposal into sparsemax/entmax/screening;
one-way zero/dustbin target attention is also already present in ZAM/C03.

- `exact_reparameterization_audit.md`: exact forward/function-class proof
- `jacobian_and_gradient_audit.md`: backward equivalence and starvation
- `nearest_neighbors_and_complexity.md`: primary literature and matched controls
- `minimal_falsifier_design.md`: synthetic and real A0 stop rules
- `preimplementation_decision.md`: binding rejection
