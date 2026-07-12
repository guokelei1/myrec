# C45 synthetic design-gate outcome

Status: terminal failure before repository data.

C45 froze and trained three seeds of four equal-parameter modes for 360 steps
each. All tensors, losses, primary gradients, checkpoints, deterministic
rescoring, candidate permutation, no-history, query-absent, repeat, and NULL
contracts were finite and structurally correct. No repository record, label,
qrel, dev/test record, or shared evaluator was read.

The primary learned the constructed task, improving its mode-specific base by
`+0.1847` to `+0.2122` NDCG@10. That does not establish architectural rent:

- primary minus `ordinary_delta` was positive in all seeds but only
  `+0.0051/+0.0074/+0.0315`;
- primary minus `factual_state` was `+0.0037/-0.0064/+0.0237`;
- primary minus `raw_event` was negative in all seeds,
  `-0.0020/-0.0193/-0.0218`;
- shuffled-event gain retention was `0.996/0.609/0.862`, above the frozen
  `0.45` maximum in every seed.

Wrong-user history did destroy the gain and clicked-direction/activity checks
passed. The failed controls and event-order specificity are independently
binding, so the paired factual-minus-NULL update does not pay representation
rent even on its own constructed latent-transition task.

## Conformance defect

The frozen D0 protocol required every mode's trainable groups to receive a
nonzero gradient. `raw_event` deliberately had no ranking dependency on the
recurrent transition, and its transition gradient group was inactive in all
seeds. The aggregate implementation mistakenly enforced this check only for
the primary and reported D0 passed. No code or report is rewritten after
outcome. Under the literal protocol C45 also fails D0. This does not change the
decision: D1 independently fails, and the simpler raw-event control won despite
using less active capacity.

Decision: close C45 without real-data access or rescue. Do not tune NULL,
history length, order loss, transition depth, synthetic process, or thresholds.
The next step must test a genuinely new information source—behavioral/
collaborative item representation—rather than another transformation of the
same frozen semantic event vectors.

Promoted report SHA-256:
`fe80c6a70ac45ecd5cdf9a856258d1d383f1392ab9c8347fd80035f230be4d1b`.
