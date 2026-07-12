# C68 Population-Relative Interaction Free-Energy Transformer

C68 tests one architecture hypothesis: user history should alter a candidate
only through a query--candidate--event interaction that is unusually supported
by that user's events relative to an exchangeable population-event reference.
The primitive is a four-way difference of log partition functions inside a
shared Transformer ranker.

The first gate is data-free.  It may use GPUs after proposal and execution
locks, but it cannot read repository records, labels, qrels, dev, or test.

```bash
python -m pytest -q \
  systems/68_population_relative_interaction_free_energy_transformer/tests

python systems/68_population_relative_interaction_free_energy_transformer/execution/freeze_proposal_lock.py
python systems/68_population_relative_interaction_free_energy_transformer/execution/freeze_g0_lock.py

CUDA_VISIBLE_DEVICES=0 python \
  systems/68_population_relative_interaction_free_energy_transformer/execution/run_g0.py \
  --seed 20264801 --output-root artifacts/c68_population_relative_interaction_free_energy_transformer/g0_v1
```

No real-data continuation is automatic.  A complete three-seed G0 pass only
authorizes a separate implementation review.
