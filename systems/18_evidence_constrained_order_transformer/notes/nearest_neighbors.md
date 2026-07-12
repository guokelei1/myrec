# C18 nearest-neighbour audit

The audit was performed before any learned C18 outcome.

| Neighbour | Primary source / local evidence | Relation and boundary |
|---|---|---|
| OptNet | [Amos and Kolter, ICML 2017](https://proceedings.mlr.press/v70/amos17a.html) | establishes differentiable quadratic programs as neural layers; C18 does **not** claim generic optimization-layer novelty, only its evidence-conditioned PPS order set and matched mechanism test |
| Fast differentiable sorting/ranking | [Blondel et al., ICML 2020](https://proceedings.mlr.press/v119/blondel20a.html) | projects onto the permutahedron to relax ranks; C18 projects score proposals onto request-specific recurrence inequalities and returns anchored logits, not a soft permutation or ranking loss |
| Differentiable sorting networks | [Petersen et al., ICML 2021](https://proceedings.mlr.press/v139/petersen21a.html) | relaxes comparator networks for ranking supervision; ECOT's active constraints come from observed evidence and enforce final-margin non-degradation |
| Differentiable ranking via OT | [Cuturi, Teboul, and Vert, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html) | produces smoothed sorting/ranking assignments; ECOT solves a Euclidean feasible-score projection and uses no transport plan |
| Target attention / DIN | registered in `doc/15_proposed_system_design_principles.md` | may produce the semantic proposal `u`, but cannot itself guarantee recurrence-anchored final margins; it is a backbone/control, not the claimed primitive |
| C06 flow | `../../06_conservative_wedge_flow_transformer/notes/mechanism_fingerprint.md` | directly adds a bounded zero-sum flow; ECOT instead computes the minimum feasible correction to a proposal and is idempotent for a fixed evidence set |
| C15 pair value / C16 energy write | `../../15_candidate_conditioned_value_write_transformer/preimplementation_decision.md`; `../../16_mixed_gradient_energy_write_transformer/preimplementation_decision.md` | neither pooled value modulation nor one energy-gradient step provides exact feasibility/minimality/idempotence; an exact generic QP remains the acknowledged nearest class |

Novelty verdict before outcome: **mechanistically distinct from C01--C17;
moderate global-neighbour risk**.  A positive paper claim would require both a
nearest optimization-layer control and evidence that the recurrence-anchored
constraint set, not merely extra listwise supervision, causes the gain.
