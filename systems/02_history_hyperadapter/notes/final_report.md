# C02 final report — stopped before dev evaluation

> Terminal amendment, 2026-07-11: the subsequently authorized one-shot
> mechanical continuation repaired the empty-mask loss and completed all five
> frozen variants.  The valid train-internal gate failed 4/6 checks and closed
> C02 before dev.  See `mechanical_continuation_outcome.md` and
> `reports/pps_c02_mechanical_continuation_gate.json`.  The report below is
> retained as the historical record of the original invalid attempt.

Decision: **`stop`**.  The second and final authorized implementation attempt
failed the train-internal run on a deterministic all-no-history batch.  No C02
dev score file was produced, the shared evaluator was not called, and no dev
outcome was used to change the design, thresholds, or code.

## Mechanism and information flow

CHHT uses frozen D2t query/item text states as inputs to a compact trainable
Transformer.  A request encoder contextualizes the ordered strictly-prior
history and a shared query/candidate Transformer produces the candidate state.
For each candidate, event coordinates `a_j(q,c,h_j,e_j)` and
`b_j(q,c,h_j,e_j)` produce

```text
M(q,c,H) = sum_j rho_j a_j b_j^T / max(1, sum_j |rho_j|)
S(q,c,H) = bounded_skew(M)
R(q,c,H) = (I - S)(I + S)^-1
Delta W(q,c,H) = U [R(q,c,H) - I] U^T W.
```

The rank-8 update acts on the candidate token's final FFN output map before the
score head.  It is ephemeral and candidate/request-specific; no user adapter
is stored and no test-time optimization occurs.  With empty history the masked
sum is exactly zero, so `S=0`, `R=I`, `Delta W=0`, and the valid candidate
scores equal the frozen D2p coordinate exactly.

The three named components remained the triadic skew-kernel generator, Cayley
HyperAdapter, and training-only recurrence preservation constraint.  The five
instantiated variants all contain 322,036 trainable parameters: CHHT, static
LoRA, output gate, mean-history residual, and history-only HyperAdapter.

## Innovation audit

The original diagonal proposal, `U diag(alpha(q,c,H)) V^T`, was judged
reducible to DISeL/Ouroboros/Gated-LoRA-style input-dependent rank modulation
and was discarded before any C02 outcome.  The locked skew/Cayley operator is
not reducible to the audited diagonal/static/profile adapters at the operator
level.  The verdict is deliberately qualified: NaRA's general dynamic dense
rank core may subsume the algebraic class, so this screen could at most have
tested the value of the event-composed skew constraint, not established global
novelty.  Full evidence and degeneration ablations are in
`notes/nearest_neighbors.md`.

## Frozen implementation and environment

- proposal lock: `notes/proposal_lock.json`, implementation attempt 2/2;
- source/design root: `systems/02_history_hyperadapter/` only;
- git commit at lock: `fb2325384afa551c6c7236ee1a83531d93f2eb42`;
- environment: `/data/gkl/conda_envs/myrec-c02`, Python 3.10.20,
  PyTorch 2.6.0+cu124, CUDA runtime 12.4;
- device: physical GPU 1, exposed alone as `cuda:0`, NVIDIA A40;
- deterministic CUDA setting: `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- seed: `20260708`;
- candidate manifest SHA-256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- intended run ID:
  `20260710_kuaisearch_c02_chht_screen_s20260708`;
- observed GPU command wall time: less than 0.03 A40 GPU-hours, below the
  8-hour ceiling;
- online LLM/API calls: zero.

The worktree was already dirty and all shared changes were preserved.  The
required repository-status check exposed only the already specified sibling
top-level names; no sibling design file, code, note, run, or outcome was opened,
imported, diffed, or copied.

## Executed checks

### Unit and structural contracts

The locked candidate passed 13/13 `unittest` cases and Python compilation:

```bash
CUDA_VISIBLE_DEVICES=1 /data/gkl/conda_envs/myrec-c02/bin/python \
  -m unittest discover -s systems/02_history_hyperadapter/tests -v
```

Covered contracts include skew symmetry, rank bound, Cayley orthogonality,
finite gradients, candidate conditioning, history-only exclusion of query,
candidate, and recurrence-bit leakage, exact no-history degeneration for all
five variants, complete-history subset classification independent of the
20-event model truncation, deterministic sampling, batching, and static
evaluation-label/test-path isolation.

### Label-safe feature preparation

Feature preparation completed in 45.247 seconds:

- train: 96,939 requests, 4,632,700 candidate rows, 54,399 history-present;
- dev: 12,229 label-free records, 575,609 candidate rows, 8,119
  history-present;
- frozen D2p dev rows aligned one-for-one with packed request/item order;
- wrong-user donor violations: zero on train and dev;
- separated evaluation labels read: false;
- held-out test data read: false;
- feature manifest:
  `artifacts/c02_history_hyperadapter/features_v1/manifest.json`, SHA-256
  `59c47b45e6ff84fe63a81517836ffb04ea0dd3007d0b5b8b7ae67935272749e4`.

### Real-train GPU smoke and implementation attempts

Attempt 1 passed finite gradients and exact no-history scores, but bf16 made
the Cayley maximum orthogonality error `8.7600974e-3`.  Before any dev outcome,
attempt 2 changed only the skew core, Cayley solve, and rotation to fp32 and
relocked all source hashes.  The repeated real-train smoke then produced:

- finite loss and gradients: pass;
- no-history score equality: exact;
- maximum Cayley orthogonality error: `7.4162932e-7`;
- wrong-user maximum core change: `0.11122597`.

### Train-internal failure

The frozen attempt-2 training command was:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python \
  systems/02_history_hyperadapter/train/train_screen.py \
  --config systems/02_history_hyperadapter/configs/screen.yaml \
  --device cuda:0
```

It stopped in CHHT epoch 1 with `FloatingPointError: non-finite chht loss`.
The deterministic batch audit found 16 all-no-history batches; the first is
batch 134.  For such a batch, the corruption-valid mask is empty and
`corruption_loss` takes `.mean()` over an empty tensor, which PyTorch confirms
is NaN.  This is a mechanical empty-mask defect, not a scientific result.

Fixing it would require a post-lock source change (return a differentiable zero
when no history-bearing candidate is valid) and another training run.  Because
attempt 2/2 had already begun, doing so would be an unauthorized third
implementation attempt.  The source was therefore left unchanged and the
stop-loss was applied.

## Gate audit

| Gate item | Result | Evidence |
|---|---:|---|
| proposal/design/literature lock before C02 dev outcome | pass | two locks; final lock is attempt 2/2 |
| candidate hash, seed, environment, physical GPU | pass | frozen hash; seed 20260708; `myrec-c02`; GPU 1 only |
| unit + real-train smoke | pass | 13/13 tests; fp32 Cayley smoke above |
| matched parameter capacity | pass | 322,036 parameters for each of five variants |
| train-internal finite/decreasing loss | **fail** | empty-mask NaN in epoch 1 |
| internal non-repeat gain vs D2p | not evaluated | no valid checkpoint |
| internal repeat preservation vs item teacher | not evaluated | no valid checkpoint |
| margin over static/output/mean/history-only controls | not evaluated | controls were not trained after CHHT invalidation |
| internal wrong/shuffle/coarse/query-mask contract | not evaluated | training invalidated before internal summary |
| frozen 3,442 / 4,677 / 4,110 dev subsets | not materialized by scorer | scoring was correctly withheld |
| full no-history exact score/rank contract | not evaluated | only real-train smoke passed exact equality |
| 1,000-request deterministic dev rescore | not evaluated | no checkpoint or dev scoring |
| one shared dev evaluator call/log row | **0 calls / 0 rows** | stop-loss occurred before scoring; no duplicate or hidden call |
| test-data boundary | pass | no test record, label, score, or metric read |
| GPU budget | pass | less than 0.03 of 8 A40 GPU-hours |

Because the execution was numerically invalid and the implementation-attempt
budget was exhausted, the preregistered single dev screening was not run.  No
claim about overall, repeat, non-repeat, corruption, or control performance is
available.

## Handoff

The only scientifically hygienic next step is a newly authorized C02 protocol
that explicitly grants another implementation attempt.  Its first change can
be limited to the empty-valid-mask zero case, followed by a new source hash,
lock, train-internal run, and—only if valid—the still-unused single dev call.
No multi-seed or full-gate budget should be granted from this failed screen.
