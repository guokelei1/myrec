# Autonomous pipeline controller

Status: minimal current controller for the active exploratory stage. The user
authorized open exploration on 2026-07-14. Exploratory ordinary-model work and
development diagnostics may proceed under the tracked manifest; confirmation,
test, proposed architecture, and paper claims remain separately locked.

Persist state in `experiments/history_response_gap/pipeline_state.yaml` after
each audit, run, evaluator call, budget change, and decision. During exploration,
the state records the current question, observation ledger, active dataset/model,
last artifact, resource ledger, corrections, and next reversible probe. Once a
confirmation protocol is frozen, it additionally records admission verdicts,
endpoints, thresholds, and the fixed selection rule.

The controller may automatically resume an incomplete authorized run, finish a
read-only summary, or perform an integrity repair. It may not:

- open dev/test labels;
- present an outcome-selected dataset or slice as independent confirmation;
- create a model tree before a Failure Card;
- increase a frozen budget or create a rescue round;
- modify a confirmation analysis after outcome access;
- open test automatically.

At every transition it must write the evidence delta, alternative explanations,
decision changed, and next stop condition. A failed exploratory premise may
redirect the next question, including from Lite to Full, but it cannot be erased
or promoted into a broader terminal conclusion. Confirmation failures receive a
scoped terminal conclusion rather than an outcome-driven rescue slice.
