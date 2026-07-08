# Systems

Proposed-system source tree for **query-conditioned evidence routing**.

This is where the method that passes the C3/C5 motivation gates (see
[doc/11_experiment_and_dataset_plan.md](../doc/11_experiment_and_dataset_plan.md))
is developed. Two candidate insights drive the design:

- **Insight-1 (Slot-Complementarity)**: personalization should act only on
  attribute slots the query leaves unspecified.
- **Insight-2 (Consensus Law)**: the degree a query is worth personalizing
  equals the cross-user click entropy for that query.

If both survive falsification, they merge into one primitive:
*query-conditioned evidence routing*.

## Layout (evolves with development)

```text
systems/
  config/          system-specific configs
  model/           gate / facet-routing model code
  train/           training entry points
  notes/           design notes and ablation pre-registration
```

## Git policy

Track source code, configs, and design notes. Do not track checkpoints,
runs, logs, or caches - the `.gitignore` catches:

```text
systems/**/checkpoints/
systems/**/runs/
systems/**/logs/
systems/**/outputs/
systems/**/.cache/
```

Early prototyping that has not passed C3 can stay under `src/myrec/models/`.
Once the method is committed to, move it here.
