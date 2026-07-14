# Autonomous pipeline controller

Status: minimal current controller. It is paused until the doc 34 E0 protocol
is reviewed and frozen; it does not authorize training by itself.

Persist state in `experiments/history_response_gap/pipeline_state.yaml` after
each audit, run, evaluator call, budget change, and decision. The state must
record the current E-phase, data-admission verdict, active model family,
registered endpoint, last artifact, budget ledger, and next authorized action.

The controller may automatically resume an incomplete authorized run, finish a
read-only summary, or perform an integrity repair. It may not:

- open dev/test labels;
- select a dataset or slice from model outcomes;
- create a model tree before a Failure Card;
- increase a frozen budget or create a rescue round;
- modify a confirmation analysis after outcome access;
- open test automatically.

At every transition it must write the evidence delta, decision changed, and
next stop condition. If a premise fails, the controller records a scoped
terminal conclusion instead of searching for a favorable replacement slice.
