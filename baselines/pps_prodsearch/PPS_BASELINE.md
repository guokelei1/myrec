# PPS B9 ProdSearch Baseline

This directory vendors the ProdSearch code used for B9 ZAM/TEM nearest-neighbor
baselines in `doc/16_next_round_c3_router_neighbor_plan.md`.

## Upstream

- Repository: `https://github.com/kepingbi/ProdSearch`
- Commit: `449335ba652fe7c877a008e154157d7b2a4b0e76`
- License: Apache-2.0 (`LICENSE.md`)

## Boundary

B9 uses this repository as `official-code, adapter to KuaiSearch interface, not
externally aligned`. It is distinct from the downgraded B6o HEM official path.
B9 does not attempt Amazon external alignment because the B6o reconnaissance
found the original benchmark split/checkpoint unrecoverable.

## Local Scope

Adapters, score export, and protocol notes live in this repository. Generated
training data, checkpoints, logs, score dumps, and model artifacts must remain
under `artifacts/`, `runs/`, or `models/` and must not be tracked here.

## Environment And Commands

- Freeze: `configs/env/pps_prodsearch.txt`
- Materialize: `python scripts/materialize_b9_prodsearch.py --output-root artifacts/b9_prodsearch/full`
- Train/score: `CUDA_VISIBLE_DEVICES=0 python scripts/run_b9_prodsearch.py --model zam --run-id <run_id> --seed <seed>`
- Audited checkpoint recovery finalization, when needed: `python scripts/finalize_b9_checkpoint_resume.py --help`
- Shared evaluation is separate and must use `scripts/evaluate_scores.py`.

## Input And Output

The project adapter writes the upstream native `product/users/vocab/review_*`
files plus `train/valid/test_id`, query-index, and bias ranklist files under
`artifacts/b9_prodsearch/`. It reads only standardized train/dev records. Every
clicked train target receives an independent synthetic user whose sequence is
the exact frozen history followed by that target. Dev candidates are padded
deterministically to 1,500 for the upstream fixed-width scorer; the converter
removes fillers and asserts exact request/candidate equality against the frozen
manifest before writing project `scores.jsonl`.

## Local Patch Summary

The complete diff is tracked at `reports/b9_prodsearch_patch.diff`.

- Carry each interaction's request-level query through train and test instead
  of sampling a product-level query. This is the only semantic data patch.
- Seed NumPy in addition to Python and PyTorch for reproducibility.
- Cast legacy byte masks to bool and load legacy checkpoints explicitly under
  modern PyTorch.
- Give index tensors an explicit `torch.long` dtype so a batch whose histories
  are all empty remains valid under modern PyTorch.
- Preserve the batch dimension when a shuffled training batch contains only
  one example; upstream's unconstrained `squeeze()` otherwise collapses it.
- Pass the parsed rank cutoff to score export; upstream otherwise silently
  exports only its hard-coded top 100.
- Retain only current and copied-best checkpoints to bound disk use; training
  and checkpoint selection are unchanged.
- Exempt `others/logging.py` from the upstream repository's broad `*log*`
  ignore rule so the vendored runtime dependency remains tracked.

Known limitation: only 11.71% of unique dev candidates (22.00% of candidate
rows) occur as clicked train targets. The official item-ID/PV path therefore
leaves most candidate embeddings without target-text training. Option A keeps
and reports this limitation rather than adding catalog-wide pretraining.
