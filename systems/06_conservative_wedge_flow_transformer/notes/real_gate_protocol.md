# C06 train-internal real mechanism gate

Status: **v1 G0 and centered fit completed; the three Hodge-path fits stopped
on a pre-A numeric implementation failure. A selective mathematical repair is
coordinator-authorized but execution requires the review1 lock**.  This document defines a cheap
D2p-state architecture falsifier.  It is not dev evaluation, paper evidence,
or authorization for a full LM run.

## Frozen question

Does candidate-local Hodge trust add ranking value beyond the same projected
flow with `t=1`, a learned direct candidate/event gate, centered cross-attention,
and the same local checkpoint reduced to global event trust?  The test changes
the history-to-score operator, not dataset features.  Every variant receives
the same frozen D2p query/item states, uniform valid-history prior, full
candidate set, listwise objective, two epochs, batch order, and seed.

## Cohort and label barrier

`train/real_data.py` loads only request IDs and structural arrays.  Before a
label-shaped array can be opened it verifies and excludes every request in the
registered C05 selection, defines non-repeat from complete history, and freezes
the following pairwise-disjoint roles by
`sha256(c06-relative-v1\0role\0request_id)`:

- 12,000 pre-cut non-repeat fit requests;
- 1,200 post-cut non-repeat `internal_A` requests;
- 600 delayed `internal_B` requests;
- 515 escrow requests;
- 512 post-cut no-history requests.

G0 hashes the train click-label source only after `selection.json` is durable.
It materializes D2p query states, the sorted unique item states referenced by
fit/A/no-history candidates and truncated histories, and full-candidate base
scores, but copies labels only for fit.  B and escrow receive neither features
nor labels.  The G0
report records per-role ordered `(request_id, candidate_item_id)` hashes and
every output hash.  Smoke/training revalidate all G0 outputs and the fit/A/
no-history candidate hashes before optimizer use.

The audit scores A twice without labels and durably writes
`a0_label_free_audit.json`.  `assert_internal_a_opening_barrier` requires that
exact report to pass, remain label-free, and match the frozen A candidate hash.
Only then is the registered label-source hash recomputed and A slices opened.
No runner path accepts qrels, dev/test records, paper metrics, B, or escrow.

## G0 and GPU layout

The later review lock must hash the config, protocol, model/control/core files,
all real-gate source, the shared metric source, and registered input manifests.
It does not exist yet.  After review and explicit authorization, the fixed
physical mapping is:

| Work | Physical GPU |
|---|---:|
| G0, local Hodge, A0/A1 audit | 0 |
| untrusted `t=1` | 1 |
| direct learned gate | 2 |
| centered cross-attention | 3 |

Each command sees exactly one device and addresses it as `cuda:0`.  Variant
smoke/training writes a disjoint report, checkpoint, attempt ledger, and run ID;
shared artifacts are read-only.  Thus the four two-epoch fits may run in
parallel without concurrent writes.

## G1 fit-only GPU smoke

Each variant runs separately on the maximum registered dominant-work batch
from the two formal fit orders.  Two optimizer steps must be finite, open the
residual scale on step one, and open query/candidate/history plus mechanism
paths on step two.  Direct-gate smoke specifically requires a nonzero
`direct_gate_projection` gradient.  Checkpoint reload must rescore bitwise.
Parameter difference is at most 2%; the frozen dominant-FLOP accounting differs
by at most 10% (the registered centered control is exactly matched in that
accounting).

## G2 variant fits

Each variant has one immutable attempt:

- seed `20260708`, exactly two epochs, fixed final checkpoint;
- identical full-candidate dynamic batches and batch orders;
- AdamW, LR `1e-3`, weight decay `1e-4`, BF16 outer autocast;
- shared masked request-listwise loss;
- no early stopping, grid, corruption, candidate sampling, or A access.

The local checkpoint is reloaded under `untrusted` and `global_hodge` modes for
same-factor counterfactuals; neither is separately trained.

### Pre-A numeric-repair allowance

V1 local Hodge, untrusted, and direct-gate fits stopped at the contracted FP64
cycle-energy nonnegativity guard before any A score or label. Under the existing
pre-A implementation-repair allowance, `notes/numeric_repair_review1.md`
defines one retry for exactly those variants. The retry changes only invalid
row evaluation to an identity-checked explicit FP64 squared-edge sum. It uses
new run/ledger IDs, preserves the original failed ledgers, G0, and completed
centered v1 files, and records fallback rows per epoch and in total. All
scientific settings are byte-checked against the parent config snapshot.

## G2-A0 label-free gate

Before A labels, all trained variants and the two local-checkpoint
counterfactuals must rescore bitwise.  The local model must satisfy:

- maximum common-mode ratio `<=1e-5` and absolute candidate sum `<=1e-5`;
- maximum absolute conservative delta `<=1+1e-7`;
- at least 10% of requests have delta range above `0.001` of the bound;
- at least 5% change an order and 1% change top-10 membership;
- local-to-`t=1` changes deltas on 5% and orders on 1%;
- local-to-global changes deltas on 5%;
- all local trust is finite and in `[0,1]`;
- candidate-common factors produce exact zero;
- every no-history score is bitwise D2p.

The first 128 packed-order A IDs also receive nested-prefix, duplicate-first,
and cross-request-distractor interventions for every variant.  Set sensitivity
has no pass threshold; the report binds only finiteness, conservation, and the
score bound.  The distractor base sentinel is the request minimum D2p score
minus one and cannot enter any evidence generator.

A0 failure writes a terminal report and stops before labels.

## G2-A1 shared ranking gate

Only an A0 pass opens A click labels.  All methods call
`src/myrec/eval/metrics.py::request_metrics`, including its registered tie
break.  Request-equal NDCG@10 and the exact same A requests are used for every
comparison.  With 10,000 paired bootstrap draws and three frozen request-hash
folds, local Hodge must simultaneously:

- exceed D2p by at least `+0.001`, with CI low above zero and all folds positive;
- exceed independently trained `t=1`, direct gate, centered attention, and the
  local-checkpoint global-Hodge counterfactual by at least `+0.0005` each, with
  every paired CI low above zero;
- have clicked-minus-unclicked score-delta CI low above zero.

Any failure closes C06.  A pass authorizes nothing automatically; delayed B
requires a new reviewed lock.  Escrow cannot rescue a failure.

## Commands after a future lock and authorization

All commands require the later assigned environment and deterministic CUBLAS
setting.  They are documented for review, not authorized for execution now.

```bash
CUDA_VISIBLE_DEVICES=0 python systems/06_conservative_wedge_flow_transformer/train/materialize_real_g0.py \
  --config systems/06_conservative_wedge_flow_transformer/configs/c06_real_mechanism_gate.yaml

CUDA_VISIBLE_DEVICES=<registered> python systems/06_conservative_wedge_flow_transformer/train/run_real_gate.py \
  --config systems/06_conservative_wedge_flow_transformer/configs/c06_real_mechanism_gate.yaml \
  --mode smoke --variant <variant>

CUDA_VISIBLE_DEVICES=<registered> python systems/06_conservative_wedge_flow_transformer/train/run_real_gate.py \
  --config systems/06_conservative_wedge_flow_transformer/configs/c06_real_mechanism_gate.yaml \
  --mode train-variant --variant <variant>

CUDA_VISIBLE_DEVICES=0 python systems/06_conservative_wedge_flow_transformer/train/run_real_gate.py \
  --config systems/06_conservative_wedge_flow_transformer/configs/c06_real_mechanism_gate.yaml \
  --mode audit
```

The commands are run from the repository root; the scripts resolve their import
roots from `__file__`.  Raw reports/checkpoints remain under
ignored `artifacts/`, `models/`, and `runs/` roots.
