# B9 Option A Execution Log

Date: 2026-07-10

Superseding note: numerical, determinism, and convergence checks remain valid.
The top-5 sheet has no recorded human reviewer/authorization, so strict internal
validity is provisional until author confirmation. Paper reporting uses
three-seed means rather than the highest seed.

Protocol: `doc/16_next_round_c3_router_neighbor_plan.md` Step 3 and the
approved Option A decision in
`doc/baseline_notes/20260710_b9_prodsearch_adapter_protocol_decision.md`.

## Adapter Evidence

- Upstream: `kepingbi/ProdSearch` commit
  `449335ba652fe7c877a008e154157d7b2a4b0e76`, Apache-2.0.
- Identity: `official-code, adapter to KuaiSearch interface, not externally
  aligned`.
- Full materialization: 232,566 clicked train examples, each mapped to an
  independent synthetic user; 5,000 train-only validation examples; 12,229
  label-free dev requests.
- Runtime assertions: synthetic history equals the frozen input history;
  sibling positives added by the adapter = 0; deterministic fillers are
  disjoint; original candidate order is preserved.
- Cold-product evidence: 11.71% unique and 22.00% row-level dev candidates
  occur as clicked train targets. No catalog-wide pretraining was added.
- Both ZAM and TEM passed native loader/model/export smoke. The smoke exposed an
  upstream export bug: parsed `rank_cutoff` was not passed to `Trainer.test`, so
  output silently stopped at top 100. The patch now passes the frozen 1,500
  cutoff; exact candidate restoration then passed.

## Environment Correction

The first full processes were started from base Python. A protocol re-read
found that doc/16 requires a separate `pps-prodsearch` environment. All four
processes were terminated before any shared dev evaluation and marked
`aborted_before_evaluation`; no qrels or test data were read. The isolated
environment was then created by cloning the exact frozen base package set.

Environment-transition smoke rescoring produced byte-identical score hashes
for both models. Formal runs therefore restart under new `r2` run IDs using
`/home/gkl/miniconda3/envs/pps-prodsearch/bin/python`. This is the second and
final configuration attempt; the three required frozen seeds are repetitions
of that one attempt, not additional tuning configurations.

## Formal Runs

Complete. Frozen models/configs: ZAM (`embedding_size=128`, `lr=0.002`,
official batch 32) and TEM (`embedding_size=128`, `lr=0.0005`, official README
batch 384, one transformer layer), each at seeds 20260708/09/10 and 20 epochs.
Each final run was evaluated exactly once after all six score files were complete.

ZAM exposed two modern-PyTorch compatibility faults after long training: an
all-empty-history index tensor inferred as float, and an unconstrained
`squeeze()` that removed the batch dimension for a singleton shuffled batch.
Both fixes are shape/dtype corrections covered by unit tests and recorded in
`reports/b9_prodsearch_patch.diff`. With user authorization, the three runs
continued from complete epoch 13/13/12 checkpoints rather than retraining prior
epochs. Model and optimizer state were restored; the upstream checkpoint does
not store RNG state, so the continuation is not bit-identical to a hypothetical
uninterrupted run. Commands, checkpoint hashes, preserved pre-resume
checkpoints, and this caveat are recorded in each run metadata file.

## Results And Decision

- ZAM: best seed 20260710 NDCG@10 0.2994; three-seed mean 0.2986 +/-
  0.0006. Best-vs-B7-bge delta -0.0311, paired-bootstrap CI
  [-0.0365, -0.0256].
- TEM: best seed 20260710 NDCG@10 0.2948; three-seed mean 0.2940 +/-
  0.0009. Best-vs-B7-bge delta -0.0358, paired-bootstrap CI
  [-0.0412, -0.0303].
- All six seeds are significantly above Random. Both models pass three seeds,
  exact determinism, decreasing loss, and the label-free top-5 integrity
  review. The review retains two tail-rank anomaly classes rather than hiding
  them.
- Candidate identity is exact for all runs: 12,229 requests and 575,609 score
  rows, candidate manifest SHA256
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- Frozen wording branch: `b9_not_claimably_above_b7_bge`. B9 strengthens the
  representative-method motivation but retains the explicit not-externally-
  aligned and cold-product caveats.
- Conservative aggregate GPU use, including the short environment-aborted
  processes, smoke/rescore work, formal TEM runs, initial ZAM segments, and
  checkpoint continuations, is below 1.2 GPU-days; the frozen 4 GPU-day cap was
  not approached. No post-result configuration was added.

Primary report: `reports/pps_b9_neighbor_summary.md`. No test record or test
qrels was read.
