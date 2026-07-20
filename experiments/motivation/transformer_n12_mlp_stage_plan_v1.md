# N12 SwiGLU stage operator plan

N12 is preregistered after N8--N11 and remains diagnostic only. It covers a
different branch of the Transformer than N11: the native MLP computation
`gate_proj -> SiLU`, `up_proj`, and the joint SwiGLU product before `down_proj`.

The fixed grid is Q2/Q3, all 8,000 internal-dev requests, blocks 13/20/27,
full/null histories, and a shared qrels-gated evaluator. The intervention hooks
the selected gate/up projection output rows and leaves native SiLU, down
projection, residual composition, token positions, and candidate slate intact.

The primary contrasts are null-history gate-from-full, up-from-full, and joint
from-full; symmetric full-history gate/up/joint-from-null conditions test
necessity. Sign-flip controls and full/null identity controls are retained. No
neuron, group, layer, or coordinate is selected from effects. A significant
stage contrast is not by itself evidence for a new MLP architecture.

`scripts/run_deep_dive_next_wave_n12_mlp_stage_queue.sh` waits for the N11
evaluator, runs four fixed block jobs in parallel, then the two remaining block
jobs, and evaluates all six bundles only after finite-coverage and identity
audits.

