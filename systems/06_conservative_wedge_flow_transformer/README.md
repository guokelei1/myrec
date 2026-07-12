# C06: Candidate-Local Hodge-Trusted Flow Transformer

Status: **CPU contracts, the locked synthetic probe, and real-gate G0 passed;
the centered fit completed. Three Hodge-path fits stopped on a pre-A numeric
guard. A strictly mathematical one-retry repair is staged, but cannot execute
before review1 lock. No A score/label exists; dev, full training, delayed B,
escrow, and test remain unauthorized**.

C06 addresses the concrete failure observed in C05: an unconstrained history
residual can move many parameters and saturate while adding the same constant to
every candidate.  C06 removes that direction from the architecture rather than
penalizing it with a dataset-tuned loss.

Its single primitive is a **history-conditioned, candidate-local Hodge-trusted
flow**. The LM produces two bounded factors for every candidate/history pair.
Their wedge product defines an implicit skew-symmetric candidate graph. C06
decomposes every event field into a globally rankable gradient and a cyclic
residual, then measures gradient-versus-cycle energy separately at every
candidate. Cycles may lower trust but may never supply a preference direction:

```text
F_ikj = -F_kij
u_ij = (1/n) * sum_k F_ikj
G_ikj = u_ij - u_kj;  C_ikj = F_ikj - G_ikj
t_ij = ||G_i.j||^2 / (||G_i.j||^2 + ||C_i.j||^2 + eps)
T_ikj = (t_ij * t_kj / 2) * (u_ij - u_kj)
delta_i = rho * sum_j omega_j * (1/n) * sum_k T_ikj
s_i = b_i + delta_i
```

Consequently, `T_ikj=-T_kij`, `sum_i delta_i=0`, candidate-common factors give
exactly zero update, and `|delta_i|` is bounded in final-score space. C05's
common `+1` translation is not representable. Pure cyclic evidence receives
zero trust and abstains; reversing a cycle cannot reverse the ranking because
cycle direction is excluded from `T`. Low-rank centered-factor identities
compute the local cycle energy in `O(B*H*C*r^2)` without materializing `C*C`
edges.

This is not claimed globally novel. Pairwise ranking, antisymmetry, Borda-style
aggregation, Hodge consistency diagnostics and candidate-set Transformers all
predate C06. The narrow hypothesis is whether making a bounded,
candidate-local Hodge-trusted conservative flow the **only** personalization
path inside an LM ranker is a useful inductive bias.
Novelty risk is high until it beats the identical flow with `t=1`, the rejected
global-event Hodge gate, centered attention, MIR/SetRank-style context, and
pairwise-additive controls.

## Why this is architecture, not dataset fitting

- no dataset ID, category bucket, query type or hand-built semantic channel;
- no KuaiSearch-specific click/purchase coefficient in the primitive;
- query, candidate, action, time and history are represented as ordinary LM
  tokens/states under the unified schema;
- a block-sparse information barrier keeps query, each candidate segment, and
  history isolated before flow: history and other candidates cannot bypass the
  wedge layer into the base score;
- the common-mode null, skew symmetry, score bound and permutation equivariance
  hold for every input tensor before seeing labels;
- D2p may be used only as a frozen falsifier coordinate; the final ranker must
  jointly train the Transformer base and the wedge-flow ranking head while
  preserving that information barrier.

## Layout

```text
configs/c06_architecture_probe.yaml  pre-outcome architecture/protocol values
configs/c06_synthetic_mechanism_probe.yaml  locked data-free falsifier
configs/c06_real_mechanism_gate.yaml  unauthorized real-gate review config
model/wedge_flow.py                  candidate-local Hodge-trust primitive
model/controls.py                    t=1/global/direct-gate/centered controls
model/information_barrier.py         final-LM block-sparse attention contract
model/transformer_core.py            jointly trained barrier-Transformer core
train/real_data.py                   label-isolated selection/features
train/materialize_real_g0.py         frozen D2p G0 materializer
train/run_real_gate.py               per-GPU smoke/train and A0/A1 audit
train/real_gate_metrics.py           paired train-internal summaries
train/losses.py                      shared probe listwise contract
tests/                               algebra, masks, gradients and bounds
experiments/                         locked CPU-only synthetic runner
notes/proposal.md                    observation -> architecture -> falsifier
notes/mechanism_fingerprint.md       exact operator and reductions
notes/nearest_neighbors.md           primary-source novelty audit
notes/gate_protocol.md               untouched-cohort G0--G3 protocol
notes/real_gate_protocol.md          executable gate and label barrier
notes/numeric_repair_review1.md      pre-A numeric repair and provenance
notes/synthetic_mechanism_probe_*    locked protocol and conditional result
```

## Current command

```bash
python -m pytest -q systems/06_conservative_wedge_flow_transformer/tests
```

The current 62 CPU tests also cover the implemented `t=1`, global-event,
direct learned candidate-gate and parameter-matched centered-attention
reductions, randomized mask/cancellation guards, real-gate cohort isolation,
post-selection label-source registration, the durable A0 barrier, and paired
statistics. The synthetic probe showed the intended conditional behavior: local trust
helped when cycle energy was planted as an error cue, but hurt after that
coupling was removed or reversed. This proves neither that the coupling exists
in recommendation data nor useful real-history ranking, throughput, or
empirical novelty. It makes a direct learned candidate gate and corruption
controls mandatory before any positive claim.
