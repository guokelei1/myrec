# Complexity and matched-control audit

## Complexity by branch

Let `C` be the candidate count, `H` the history length, `d` the state width, and
`r` a pair-energy hidden width.

| branch | forward cost beyond projections | important consequence |
|---|---:|---|
| bilinear candidate gradient | `O(CHd)` | same score/read complexity as cross-attention, with values tied to score derivatives |
| nonlinear scalar pair energy | `O(CHr)` plus gradient evaluation | training through `grad_z epsilon_theta` requires mixed parameter/state derivatives and retains all pair activations |
| explicit mixed Hessian | up to `O(CHd^2)` storage/work | a Hessian-vector product avoids materializing the Hessian but does not avoid the scalar-gradient reduction |
| candidate-axis competition | `O(CHd)` per iteration | Slot-style repeated refinement multiplies this by its iteration count |
| two-map or centred attention | one or two `O(CHd)` maps | the fixed uniform map is cheaper but exactly the Differential/ZeroS neighbour |

Automatic differentiation does not create a free new architecture.  A
Hessian-vector product can compute a contraction at roughly derivative-program
cost, but backpropagating into the parameters of that derivative generally
requires higher-order derivatives and additional saved state.  Detaching the
gradient/Hessian to avoid those derivatives changes the training rule and
breaks the claimed end-to-end energy mechanism.

## Controls that would be mandatory if a branch survived

Any implementation request would first need pointwise and gradient comparisons
against:

1. ordinary cross-attention with matched `Q/K/V/O` width;
2. the exactly tied-value cross-attention derived in the algebraic audit;
3. a modern-Hopfield retrieval layer;
4. an equal-capacity Energy Transformer or HFY energy update;
5. Slot Attention-style candidate-axis allocation;
6. `softmax(s)-softmax(0)` Differential attention and ZeroS-style uniform
   removal; and
7. an equal-capacity non-conservative pairwise value network for any branch
   that abandons integrability.

Required accounting would include active-gradient parameters, `C x H` pair
activations, higher-order derivative memory, wall time, and maximum supported
candidate/history sizes.  Dummy parameters could not be used for capacity
matching.

These controls are not scheduled: exact algebra already predicts equality to a
nearest neighbour or loss of the defining energy claim.  GPU testing cannot
repair a failed mechanism-identity gate.

No benchmark, synthetic generator, real record, label, model, or GPU was used
for this audit.
