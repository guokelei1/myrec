# Paper design constraints

Status: active, but deliberately narrow. This file contains repository-wide
evidence rules; it does not select an architecture or reopen any historical
search.

## Binding from the first audit

1. Use one unified record contract: optional query, strictly prior history,
   fixed candidate slate, evidence masks, and labels outside dev/test records.
2. Keep a fixed split, identical candidate sets, one shared evaluator, and
   candidate-set hashes across methods.
3. Make every control claim-specific. Wrong-user history is for provenance;
   shuffle is for order; no-history is for base preservation; recurrence and
   strict-nonrepeat are separate surfaces.
4. Decide on development data only. Freeze the confirmation cohort, endpoint,
   model-selection rule, thresholds, and analysis before confirmation.
5. Report response, candidate-relative direction, ranking utility,
   specificity, attribution, and data sufficiency separately. A nonzero
   response is not evidence of correct direction.
6. Do not use a per-dataset branch to turn incompatible information objects
   into one claim. A secondary track can narrow a claim, not silently enlarge
   it.

## Current scientific order

Follow doc 34: source admission and power audit; strong query-candidate base;
ordinary full-token family adequacy; label-free response instrumentation;
direction and user-specificity evaluation; simple controls and signal witness;
then replication. A new architecture is out of scope until doc 31's Failure
Card gate passes.

## Explicitly non-binding historical rules

The former C0–C5 phase gates, C01–C80 candidate rules, R0 round budgets,
event-permutation contracts, exact-recurrence premises, and old architecture
eligibility language are archived. They must not be applied to the new
direction unless a future pre-outcome protocol explicitly reinstates a rule.
