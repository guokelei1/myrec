# Functional-identifiability audit after C67

Date: 2026-07-12
Status: architecture-entry audit; no new repository label was read.

## Corrected conclusion

Functional dependence on history is necessary, but it is not the remaining
missing condition by itself. Several earlier candidates already made a
history-scrubbed correction algebraically zero and prevented an ordinary
query-candidate head from expressing the same primitive. They still failed
utility or nearest-control gates. C67 adds one more example: excluding query
and candidates from fast-weight writing does not identify the written
function if the read side can learn an equivalent generic comparator.

The next candidate must pay two rents simultaneously:

1. **functional identity rent** — the particular history-derived function,
   not merely a nonzero carrier, must change candidate margins;
2. **cross-domain specificity rent** — the same operator must retain
   true-over-wrong history evidence on KuaiSearch and Amazon-C4.

## Existing identifiability mechanisms and outcomes

| Mechanism class | Candidates | Identity contract | Outcome |
|---|---|---|---|
| higher-order / common-mode annihilation | C06, C08, C25 | wedge, reversible commutator, or anchored Möbius interaction removes lower-order/query-candidate-only terms | common-mode collapse, synthetic-control tie, or Top-10 inactivity |
| conjunctive or authenticated paths | C09, C22, C29--C30 | disagreement/no-authentication gives exact base; query and candidate effects must traverse history | nearest simple attention wins or A1 ties/loses base and wrong history |
| fixed-read query transport | C31--C43 | history modifies a query state and candidate score uses a fixed semantic inner product | C32/C33 weakly positive; C42 strong on Amazon; C43 loses stable true/wrong specificity on KuaiSearch |
| factual/NULL functional differences | C04, C45, C61, C65--C66 | only factual-minus-reference function/state/logit reaches the residual | inactive, raw factual control wins, or wrong history is not Top-10-load-bearing |
| write-once/function-valued memory | C62--C63, C67 | candidate cannot write memory; no-history is exact identity | slots pool/collapse; held-out fast weights become a generic carrier |

This table rules out the claim that one more zero-mask, NULL subtraction,
three-way product, commutator, or history-only writer is an uncovered
architecture position.

## Strongest empirical boundary

C42 is the strongest positive architecture-family evidence currently in the
repository. Its metric-coupled model reached `0.333347` NDCG@10 on fresh
Amazon escrow, beating its base by `+0.110718`, C38 by `+0.010250`, and wrong
history by `+0.035234`, all with positive intervals. It nevertheless tied two
close routing controls under the full uniqueness gate.

C43 transferred the exact operator to KuaiSearch without tuning. It beat base
by `+0.004124` with a positive interval, but true-minus-wrong was only
`+0.000487` with an interval crossing zero, and shifted/single-wide controls
matched it. Thus the transferable component is a generic query-candidate
metric improvement, not established personalized evidence.

This is more informative than another synthetic pass: the architecture family
can improve ranking, but its history-specific part is domain-fragile.

## Pre-implementation rejection of obvious C68 drafts

The following drafts are not authorized as proposed-system candidates:

- **plain request-local TTT / fast LoRA with a fixed readout**: test-time
  training layers, TTT4Rec, GradMem, Profile-to-PEFT, and instance-wise LoRA
  already cover the central mechanism; C02/C67 cover the local repository
  reductions. This can only be a non-novel signal probe.
- **gradient-surgery memory**: projecting per-event gradients into a common
  descent cone reduces to PCGrad, MGDA, or CAGrad-style multi-objective
  optimization, while C20/C34/C48 already cover cone/consensus reductions.
- **per-user neural process**: task-adaptive neural processes already model a
  user as a stochastic function; adding one without a new information graph is
  not a mechanism innovation.
- **latent reasoning tokens alone**: recurrent latent recommendation and
  generic memory tokens are established; C62 already tests a closely related
  write/read lifecycle. More recurrent steps are not a falsifiable new
  primitive.
- **another C42 head gate, temperature, rank, or candidate-specific router**:
  this is explicitly forbidden by C42/C43 stop rules and would fit the opened
  outcomes.

Primary references for these boundaries include TTT
<https://arxiv.org/abs/2407.04620>, TTT4Rec
<https://arxiv.org/abs/2409.19142>, GradMem
<https://arxiv.org/abs/2603.13875>, PCGrad
<https://arxiv.org/abs/2001.06782>, CAGrad
<https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html>,
TaNP <https://arxiv.org/abs/2103.06137>, and latent-reasoning recommendation
<https://openreview.net/forum?id=eUtIZT2ONS>.

## Remaining admissible search target

No C68 implementation is currently justified. A valid proposal must introduce
an information object that is absent from the table above, and its
pre-implementation proof must show all of the following before GPU use:

1. the correction family cannot be represented by query/candidate-only
   parameters or by replacing history with a request-constant carrier;
2. it does not reduce to attention, query transport, dynamic LoRA/TTT,
   higher-order products, candidate transport, or a fixed-score router;
3. a closest known mechanism and a function-equivalence control are executable;
4. one identical synthetic or exposed-fit falsifier applies in both domains;
5. passage requires both utility over the strong base and true-over-wrong
   specificity, not absolute activity.

The search remains open, but implementing an already closed algebra under a
new name would move toward experiment-history overfitting rather than toward a
good architecture.

## Post-C68 correction

C68 subsequently satisfied the lower-order cancellation proof but failed every
seed's utility, wrong-history, and fixed-carrier gates. A separate read-only
C42/C43 audit also showed that KuaiSearch C43 true/wrong correction functions
are much less correlated than Amazon C42 (`0.426` versus `0.968`) despite C43's
near-zero relevance gain. Functional identity is therefore necessary but not
the dominant missing information by itself: a history-specific function may
still point in a behaviorally irrelevant direction. The next prerequisite is
an open-catalog behavioral relation that pays rent over ordinary semantics;
see `20260712_c42_c43_function_alignment_and_catalog_coverage_audit.md`.
