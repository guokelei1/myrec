# Motivation V1.1 axis ordering

## Decision

The follow-up is ordered as two independent KuaiSearch axes:

1. Axis A fixes the V1 `full_confirm_preceding10k_v1` population and tests
   training sufficiency by increasing the epoch budget.
2. Axis B freezes the epoch chosen by the train-only Axis-A rule and enlarges
   only the strictly preceding training population to
   `full_confirm_preceding40k_v11`.
3. Only after both axes are closed does a separate population extension test
   the same observation on JDsearch or another admitted dataset.

The V1 entry and its evidence remain immutable. A combined change of data
population and epoch budget is exploratory and cannot support attribution.

## Execution correction

An InstructRec run using the 40k population and two epochs was started before
this ordering was made explicit. It was interrupted before producing a
checkpoint or confirmation result and is not evidence. The protocol and
configs were amended before the new Axis-A runs were accepted.

TEM's V1 run already used 20 epochs, so its Axis-A extension is strictly
above 20. InstructRec's V1 run used one epoch, so its Axis-A extension is two
epochs under the fixed V1 population.
