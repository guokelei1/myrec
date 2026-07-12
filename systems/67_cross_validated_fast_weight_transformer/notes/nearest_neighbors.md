# C67 nearest-neighbor and reduction audit

Status: provisionally distinct only through exact within-request held-out
validation. No global novelty claim is made.

| Neighbor | What it already covers | Binding difference / control |
|---|---|---|
| [TTT layers](https://arxiv.org/abs/2407.04620) | a model-valued hidden state updated by a learned self-supervised gradient step | TTT writes every event; C67 requires its update to improve other events; `standard_ttt_write` is binding |
| [TTT4Rec](https://arxiv.org/abs/2409.19142) | test-time gradient adaptation inside sequential recommendation | plain TTT-for-recommendation is not a C67 claim; C67 separates history-only write from query/candidate read and validates each write out of event |
| [GradMem](https://arxiv.org/abs/2603.13875) | optimizing compact memory at inference with context reconstruction | GradMem optimizes context fit; C67 tests cross-event generalization before aggregating a proposed update |
| [Gradient Agreement](https://arxiv.org/abs/1810.08178) | weighting meta-learning tasks by gradient inner products | its first-order statistic is an explicit C67 control; primary uses exact post-update held-out loss and must beat it |
| [Profile-to-PEFT](https://arxiv.org/abs/2510.16282), [iLoRA](https://arxiv.org/abs/2408.10159) | forward-generated user-specific adapters / expert mixtures | C67 does not generate weights from a pooled profile; a differentiable inner learner writes them without query/candidate access |
| C02 CHHT | candidate-conditioned hypernetwork changes LM internals | C02's write sees query/candidate and is forward-generated; C67 write is history-only inner optimization |
| C48/C49 | signed KRR influence consensus and prequential DeltaNet memory | those operate on fixed score/embedding memories; C67 applies exact cross-event validation to end-to-end internal fast weights; first-order and ordinary-update controls bind this distinction |

Reduction warning: for infinitesimal `eta`, exact held-out improvement has
first-order term `eta <g_e, mean_{j!=e} g_j>`. C67 therefore has no mechanism
claim unless finite-step exact validation consistently beats the registered
gradient-agreement control. This is an outcome gate, not a rhetorical
difference.
