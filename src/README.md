# Shared source

`src/myrec/` contains reusable baseline adapters, evaluation, metrics, hashing,
and JSONL utilities. The old Lite/C0 data materializers and analysis modules
are archived; a new E0 data adapter will be added only after its source
contract is reviewed.

The new direction must use the standardized record interface and shared
evaluator, and must not read dev/test qrels from training or scoring code.

`src/myrec/data/contracts.py` and `src/myrec/eval/history_response*.py` are the
active foundations for E0 and counterfactual direction measurement. Historical
analysis modules remain in `archive/` and are not import dependencies.

Future shared data conversion belongs under `src/myrec/data/`. Hypothesis-
specific model code belongs under `systems/<hypothesis>/` only after doc 31
authorizes it.
