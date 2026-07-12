# C76 data-free design-gate outcome

Decision: `close_c76_before_repository_data`.

All three seeds passed the full G0 contract.  Factual/history-cut token IDs and
positions were shared; history-cut Q/C states equaled null-history states with
zero error; no-history returned the protected base exactly; repeat returned
item-only exactly; deterministic and candidate-permutation errors were zero;
the trajectory and early-layer coordinates were nonzero; gradients reached
the adaptive Transformer and trajectory Transformer but not the protected
base; every mode had 324,480 parameters.

All 15 fits (three seeds x five modes) were finite, completed 500 steps, and
lowered loss sharply.  Primary final-window loss was approximately
`0.0255--0.0263`.  The model therefore fit the training surface and the
operator was not dormant.

The held-out nuisance reversal failed maximally.  In every seed:

- primary supported accuracy was `0.0000` versus the tied-base `0.1091`;
- clean supported margin was approximately `-5.9994`;
- wrong-history and query-mask gain retention were approximately `1.0`;
- `final_logit_delta`, `final_hidden_delta`, `factual_trajectory`, and
  `ordinary_full` also had supported accuracy zero, so every registered primary
  control advantage was exactly `0.0`;
- repeat and no-history accuracy remained `1.0`, and event shuffle retained
  the outcome, as expected for the exchangeable surface.

The key failure is not that a final delta discarded early useful signal.  A
candidate-only nuisance changes the candidate hidden state that produces
candidate-history attention, so it enters the factual-minus-cut trajectory as
an *interaction modulation*.  Structural subtraction removes a direct generic
factual path but does not authenticate which candidate coordinates are allowed
to condition history evidence.  Reading more layers therefore preserves the
shortcut rather than identifying history evidence.

C76 receives no normalization, trajectory-width, layer, nuisance, scale,
step, seed, or real-data rescue.  No repository data, label, dev, test, qrel,
or evaluator was accessed.  Under the C76--C80 terminal budget, the next
primitive must authenticate raw candidate-history token interaction with the
current query; it may not be another factual/cut output subtraction.

Authoritative report: `reports/pps_c76_design_gate.json`.
