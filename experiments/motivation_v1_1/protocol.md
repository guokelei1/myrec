# Motivation V1.1 staged protocol

Amended: 2026-07-16, before accepting any V1.1 confirmation outcome. This is a
robustness and population-extension protocol, not an architecture-search plan.
The amendment makes training-sufficiency and population-size extensions
separate axes after review of the initial execution attempt.

## 1. Immutable V1 boundary

The V1 entry point, three-model audit, Qwen decision, candidate surface
definitions, label mode, bootstrap implementation, and V1 claim boundary are
immutable:

- `doc/40_transformer_recurrence_transfer_motivation_v1_zh.md`;
- `reports/pps_three_transformer_history_surface_audit.json`;
- `reports/pps_motivation_confirmation_decision.json`;
- `data/standardized/kuaisearch/full_confirm_preceding10k_v1/`;
- the V1 candidate manifest SHA-256
  `f535c43774c88387440df0f1dec3273c3a6401523973c2a37ad1d75f1e3a7d15`.

No V1 report or run is overwritten. No test split is opened. V1.1 is reported
separately even when it agrees with V1.

## 2. Ordered KuaiSearch axes

The KuaiSearch work is executed in order. A run that changes both the
training population and the epoch budget is exploratory only and cannot be
used to attribute an outcome. The current interrupted InstructRec
`full_confirm_preceding40k_v11 x 2 epochs` attempt is therefore not evidence.

### Axis A: training sufficiency with the V1 population fixed

Use the original `full_confirm_preceding10k_v1` source population and the
unchanged V1 confirmation cohort. Rebuild one fixed train-only internal-dev
partition from the V1 training records for checkpoint selection; the same
partition and model-visible subset are used for every epoch/seed cell. No
confirmation qrels are opened during this axis.

The epoch budget and seeds are declared before training. The epoch-1 control
is re-trained under the same internal-dev partition rather than compared to a
checkpoint selected using confirmation. InstructRec is the primary epoch-axis
case because V1 used one epoch. TEM already used a 20-epoch V1 run; any TEM
epoch extension must therefore be strictly beyond 20, not a second 20-epoch
run. All epoch cells use the same prompt, candidate slate, history budget,
optimizer, and shared evaluator.

Axis A answers only whether the V1 observation is plausibly a training-
sufficiency artifact. A plateau or overfit result closes this axis without
changing V1.

### Axis B: population extension with the epoch budget fixed

`data/standardized/kuaisearch/full_confirm_preceding40k_v11/` is constructed
from the pinned KuaiSearch Full `recall/train.jsonl` and `items/train.jsonl`.
The 2,000 V1 confirmation records and their candidate identities are copied
byte-for-byte into the new confirmation cohort. Only the training-side source
window is enlarged: 40,000 eligible requests are selected strictly before the
minimum confirmation timestamp, with a time-contained 80/20 train/internal-
dev split. The model-visible training set is therefore the earlier 32,000
requests; the 8,000-request internal-dev side is used only for checkpoint
selection. Source test rows, confirmation labels, and future records are not
used by training or scoring.

Admission fails if any training request is at or after the confirmation
minimum timestamp, if a training session overlaps the confirmation sessions,
if the copied confirmation candidate projection differs from V1, if a
confirmation record contains labels, or if any history event is not strictly
before its target request.

The V1.1 primary endpoint is graded NDCG@10 `true history - null history` on
the same four evaluator-side positive-eligible surfaces:
`target_repeat`, `target_nonrepeat_other_candidate_overlap`,
`target_nonrepeat_no_candidate_overlap`, and `target_nonrepeat_no_history`.
The all-request population and the positive-eligible population are both
reported. The same shared history-response evaluator, candidate assertion,
query-cluster bootstrap (5,000 draws, seed 20260715), activity epsilon 0.01,
and utility epsilon 0.0 are used for every model and population.

After Axis A, the epoch used in Axis B is frozen by the Axis-A internal-dev
rule and is applied unchanged to the V1 control and the 40k population. Thus
the Axis-B contrast is population size at a fixed training budget. Axis A
selected TEM 40 and InstructRec 2 for both declared seeds, so Axis B uses
TEM 40 and InstructRec 2 for every seed; confirmation results cannot select
or change these values.

## 3. Training matrix and checkpoint rule

All pre-declared seeds are reported, regardless of outcome. There is no
best-seed selection, slice selection, epoch selection from confirmation
results, or post-hoc budget increase.

| stage | model | population | seeds | epoch rule | checkpoint rule |
|---|---|---:|---:|---:|---|
| Axis A | TEM | V1 fixed | 20260717, 20260718 | V1 20 plus only pre-declared `>20` extension | `model_best.ckpt` selected by train-only internal validation MRR; P@1 is secondary |
| Axis A | InstructRec / Flan-T5-XL | V1 fixed | 20260717, 20260718 | V1 1 plus pre-declared higher-epoch cell | best epoch by train-only internal-dev graded NDCG@10; ties keep earliest epoch |
| Axis B | TEM | 40k extension | 20260717, 20260718 | fixed Axis-A epoch 40 | same train-only internal validation rule |
| Axis B | InstructRec / Flan-T5-XL | 40k extension | 20260717, 20260718 | fixed Axis-A epoch 2 | same train-only internal-dev rule |

TEM keeps its V1 implementation, item vocabulary, history budget 20, 128-d
embedding, one interaction Transformer layer, learning rate 5e-4, and 4
workers. InstructRec keeps the V1 prompt, candidate slate, history budget 6,
max source length 2048, max target length 64, bfloat16, learning rate 1e-5,
weight decay 0.01, warmup 0.1, and gradient clipping 1.0. Effective batch
size is fixed; if scheduling requires less parallelism, only device placement
or gradient accumulation may change, never the declared effective batch.

The InstructRec internal-dev partition is a deterministic request-ID hash
partition of the selected population's `records_train.jsonl` and reads only
that population's `qrels_train.jsonl`. It is never a V1 `qrels_dev`,
confirmation, or test file. The selected checkpoint is frozen before any
confirmation qrels are opened. Axis-A and Axis-B epoch decisions are logged
separately; no Axis-B confirmation result can retroactively select an epoch.

For each selected full-history checkpoint, the evaluator scores exactly the
same checkpoint under `true`, `null`, and `wrong` assignments. TEM's wrong
condition is mandatory for the provenance diagnostic; it is not used to select
a checkpoint. All score bundles must pass candidate/request/checkpoint and
finite-score assertions before the shared evaluator opens confirmation qrels.

## 4. Second population, only after both KuaiSearch axes

The JDsearch/other-dataset extension is intentionally downstream of the two
KuaiSearch axes. It repeats the frozen KuaiSearch surface analysis with an
independent population and does not explain away a KuaiSearch split. JDsearch
may first be expanded with a strict pre-confirmation train/internal-dev
window, multiple seeds, and the same checkpoint/evaluator contract; its
anonymized query boundary still limits the claim to functional replication.

KuaiSAR is not promoted to a natural-language semantic replication because
its query/caption fields are anonymized IDs. If used, it can only be reported
as functional behavioral replication. The pre-registered fallback is
`jdsearch/hash_scout10k_v3`, whose source, causal-history, candidate, and
label-isolation checks passed. JDsearch uses the exact same target-aware
surface definitions and shared evaluator, but its result is labeled
functional/anonymized rather than semantic. Amazon-C4 remains a non-binding
English stress test and cannot be used to estimate natural-search prevalence.

The existing JDsearch v3 full-history score bundle is eligible only after a
new V1.1 report re-audits its shared-evaluator metadata, manifest hashes, and
surface counts. If the source or candidate audit fails, the population is
rejected; no result slice may rescue it.

## 5. Interpretation and fallback

The KuaiSearch report first states the Axis-A training-sufficiency result and
then the Axis-B population result. Only after both are complete may the
second-population report test transfer of the same `repeat-positive /
no-overlap-not-established` observation. A disagreement is reported as a
split, with its source localized to training sufficiency, model, or
population.

More epochs that do not improve the internal-dev task metric are reported as
plateau/overfitting and the V1 checkpoint remains valid. A reliable
no-overlap gain is recorded as a V1 boundary condition: V1 remains true on
its frozen population, but the attempted prevalence extension fails. No
threshold, canonicalization, label opening, test result, or new architecture
may rescue a failed gate.
