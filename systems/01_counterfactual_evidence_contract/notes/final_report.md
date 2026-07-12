# C01 Final Report — Counterfactual Evidence-Contract Transformer

Final status: **`stop`**
Date: 2026-07-11 (Asia/Shanghai)
Candidate: `c01`
Reserved run ID: `20260710_kuaisearch_c01_cect_screen_s20260708`

## Executive decision

C01 completed its pre-outcome proposal lock, literature audit, minimal source,
unit/CPU/GPU smoke checks, two implementation attempts, and a train-internal
execution.  It did **not** produce dev scores and did **not** call the shared dev
evaluator.

The second attempt's internal numbers are invalid for the frozen C01 gate.  The
train adapter used

```text
0.6 * z(frozen-BGE cosine) + 0.4 * z(train popularity)
```

as its train/internal anchor, whereas registered D2p is

```text
0.6 * z(seed-20260708 fine-tuned D2t query-tower score)
+ 0.4 * z(legal train-only popularity).
```

This is not a harmless affine difference: the wrong anchor was present in the
ranking loss as well as the internal comparisons.  Therefore the observed
internal failures cannot be promoted to a falsification of CECT, and the two
attempt budget is exhausted.  A corrected run would require a new coordinator
authorization, a new pre-outcome lock/budget entry, and an explicitly
materialized frozen train D2p array.  No such run was attempted here.

## Locked architecture

### Information flow

For each candidate `c`, CECT builds one sequence from query `q`, candidate, and
the last 20 strictly-prior history events:

```text
[QUERY] [CANDIDATE] [EVENT_1] ... [EVENT_L]
       -> shared two-layer Triadic Event Transformer
       -> contextual event states h_i
       -> certificate energy a_i and transfer value v_i
       -> Counterfactual Quantile Contract
       -> Contracted Residual Readout -> candidate logit
```

The trainable core is a 96-dimensional, two-layer, four-head Transformer with a
192-dimensional FFN.  Frozen local BGE states are projected into the core;
query, candidate, event type, reverse age, category, and candidate-event
relation embeddings participate in its ranking information flow.

### Operator

The observational sequence and four training/diagnostic twins—wrong-user,
event-shuffled, query-masked, and coarse-only—share all encoder parameters.  On
a clicked non-exact candidate,

```text
A(o) = tau_lse * logsumexp_i(a_i(o) / tau_lse)
L_cf = relu(mu_cf - (A(true) - max_k A(twin_k))).
```

After stage 1, counterfactual event energies on a disjoint train-only
calibration slice set the finite-sample threshold

```text
Q_cf = sorted(a_cf)[ceil((n + 1) * (1 - alpha_cf)) - 1],
alpha_cf = 0.10.
```

The energy path and `Q_cf` are then frozen.  Non-exact transfer can enter the
readout only above the contract boundary.  Exact candidate recurrence is the
protected relation atom with floor

```text
3 * event_weight / sqrt(reverse_age).
```

The history-present logit follows the locked anchor form

```text
s(c) = 0.30 * z(D2p(c)) + 0.70 * z(E(c,H)).
```

When history is absent or contracted evidence is empty, inference returns D2p.
Counterfactual twins are never inference inputs.

### Mechanism fingerprint

The load-bearing primitive is an **event-level, multi-twin,
false-admission-calibrated evidence contract inside a shared Transformer**.  It
is not a fixed-score router, query-type classifier, offline LLM feature, paired
prefix logit delta, history hypernetwork, or optimal-transport module.  The
three named components are TET, CQC, and CRR; the parameter-matched control
replaces CQC by ordinary target attention without changing parameter count.

## Nearest-neighbor verdict

Pre-outcome verdict: **`distinct-with-uncertainty`**.

The closest located operators were CFT's whole-history/history-free paired
output difference and CARD's leave-one-event-out future-prediction-error
weighting.  CECT differs by calibrating a candidate-specific event certificate
against the robust maximum of four counterfactual twin families, then using a
train-only null quantile for one-pass true-input inference.  DIN, SIM/UBR, ZAM,
TEM, RTM, SASRec/BERT4Rec, CauseRec/CASR/CaseRec, CounterCLR, CLLMR, CoDeR(+),
UFRec, and final-score confidence/conformal methods were also checked.  The
full operator table and primary sources remain in `notes/nearest_neighbors.md`.

No outcome changes this literature verdict.  Because the executed probe is
invalid, C01 has not established empirical novelty or utility.

## Frozen identity and environment

- Proposal lock time: `2026-07-11T00:19:23+08:00`.
- Proposal candidate hash:
  `d7c2ba6e2b1c168ed40fa36dcfb95031bdade4f48df0e7a687998e4ea594a546`.
- Locked config SHA256:
  `3189ad10244a9b549fd699006724e5d648b6cb8101d07ee588c11eafd9c8760b`.
- Candidate-manifest SHA256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- Git commit at lock:
  `fb2325384afa551c6c7236ee1a83531d93f2eb42` (dirty user worktree preserved).
- Environment: `/data/gkl/conda_envs/myrec-c01`, Python 3.10.20,
  PyTorch 2.6.0+cu124, NumPy 1.24.3.
- Device: physical GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), NVIDIA A40.
- Seed: `20260708`.
- Deterministic cuBLAS workspace: `:4096:8`.

The proposal-lock hashes were revalidated after execution.  The design and
config files were not edited after lock.

## Files added

Design and lock evidence:

- `README.md`, `environment.txt`, `configs/screening.yaml`;
- `notes/proposal.md`, `notes/mechanism_fingerprint.md`,
  `notes/nearest_neighbors.md`, `notes/gate_protocol.md`, and
  `notes/proposal_lock.json`.

Minimal implementation:

- `model/__init__.py`, `model/cect.py`;
- `train/__init__.py`, `train/data.py`, `train/engine.py`,
  `train/integrity.py`, `train/smoke.py`, and `train/run_probe.py`;
- label-free downstream scripts `train/scoring.py`, `train/score_dev.py`,
  `train/verify_determinism.py`, `train/pre_evaluation_audit.py`, and
  `train/summarize_result.py`.  These downstream scripts were not executed
  because the internal gate never authorized dev scoring;
- `tests/__init__.py`, `tests/test_cect.py`;
- this report.

Local runtime state is confined to `artifacts/c01_cect_probe/` and
`models/c01_cect_probe/`.  No C01 run directory was created.

## Commands actually executed

Environment creation:

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda create -n myrec-c01 --clone pps-kuaisearch -y
```

Unit tests (repeated after fixes):

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda run -n myrec-c01 python -m unittest discover \
  -s systems/01_counterfactual_evidence_contract/tests -v
```

Formal attempts:

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs CUDA_VISIBLE_DEVICES=0 \
  conda run -n myrec-c01 --no-capture-output python \
  systems/01_counterfactual_evidence_contract/train/run_probe.py \
  --config systems/01_counterfactual_evidence_contract/configs/screening.yaml \
  --attempt 1

CONDA_ENVS_PATH=/data/gkl/conda_envs CUDA_VISIBLE_DEVICES=0 \
  conda run -n myrec-c01 --no-capture-output python \
  systems/01_counterfactual_evidence_contract/train/run_probe.py \
  --config systems/01_counterfactual_evidence_contract/configs/screening.yaml \
  --attempt 2
```

The dev scorer, deterministic dev rescorer, pre-evaluation audit, shared
evaluator, and result summarizer were **not** run.

## Unit and smoke results

- Candidate-local unit suite: **9/9 passed**.
- Covered: finite-sample quantile indexing, protected exact atom, exact
  empty-history fallback, twin construction, padding masks, matched parameter
  count, finite outputs, candidate-hash rejection, source isolation, and CPU
  Transformer/certificate/value gradients.
- GPU 0 smoke: finite loss `0.6307244897`; one optimizer step completed.
- GPU smoke gradient norms: Transformer attention `0.05938917`, certificate
  head `0.06498054`, value head `0.06420143`.
- A 100-request aggregation smoke traversed repeat, non-repeat, no-history, all
  corruption, bootstrap, non-collapse, and matched-control branches.
- Deterministic training replay: logged losses at every reported checkpoint in
  attempt 2 exactly matched attempt 1.
- Required first-1,000 dev deterministic rescore: **not run**, because the
  internal gate did not authorize any dev scoring.

## Attempts and budget

### Attempt 1

Training completed, then the internal reporter raised an implementation
exception by calling nonexistent `torch.flatnonzero`.  No internal gate result
or dev artifact was produced.  This is an allowed implementation-error retry.
Estimated occupied GPU wall time: `0.1524168` hours.

### Attempt 2

Training and the internal reporter completed.  Current-attempt occupied GPU
wall time through the internal report: `0.1736384` hours.  Cumulative estimated
GPU wall time: **`0.3260553 / 8.0` hours**.  Both allowed implementation
attempts were consumed.

The saved diagnostic checkpoints contain 693,603 parameters each and are
confined to the C01 model prefix.  They must not be used as a validated C01
system because of the D2p-anchor violation.

## Gate audit

The values below are preserved only to audit what attempt 2 emitted.  They are
marked **invalid**, must not enter a paper/table, and do not license a mechanism
claim.

| Frozen item | Attempt-2 observation | Valid gate verdict |
|---|---|---|
| Protected recurrence | contract − approximate item-only `-0.03077`, bootstrap CI `[-0.03673,-0.02497]`, 2,720 requests | **invalid / not established** |
| Non-repeat transfer | contract − approximate base `-0.10875`, CI `[-0.11823,-0.09966]`, 3,553 requests | **invalid / not established** |
| Counterfactual rejection | reporter returned false because true gain was non-positive; shuffle admission drop was only `0.0030`, wrong-user `0.2428` | **invalid / not established** |
| No-history contract | max score difference `0.0`, rank mismatches `0`, delta `0.0` on 3,421 internal requests | **passed structurally in invalid run; overall not sufficient** |
| Non-collapse/order | true admission `0.18589`, energy std `0.14165`, true-minus-pooled-twin energy `0.34847`, but shuffle mass drop only `0.00166` | **invalid / not established** |
| Matched plain control | equal parameter counts; contract − plain `+0.05566`, CI `[+0.04484,+0.06530]` | **invalid / not established** |

The report initially labelled four items failed and two passed, as required by
the frozen implementation.  The subsequent fairness audit supersedes any
scientific interpretation of all ranking comparisons, because both training
and comparison anchors were wrong.  No threshold, subset, module, or loss was
changed after seeing these values.

## Dev, evaluator, and determinism accounting

- Blind dev score rows produced: **0**.
- C01 dev run directory: **absent**.
- First-1,000 dev deterministic rescore: **0 calls / not authorized**.
- Shared dev evaluator calls for the reserved run ID: **0**.
- Matching `reports/dev_eval_log.jsonl` lines: **0**.
- Dev qrels read by C01 training/scoring: **no**.
- Test records/qrels/metrics read: **no**.

## Integrity and leakage audit

- Proposal candidate hash and every locked design-file hash revalidated.
- Candidate manifest hash asserted before cache preparation and model execution.
- Train, calibration, and internal slices remained label-isolated by frozen
  record ranges; wrong-history donors were constrained to their own slice.
- The dev adapter never loads candidate labels.  Its true-only inference path
  does not construct wrong-user histories; it was implemented but not run.
- Source isolation scan checked 15 Python files and found zero forbidden
  held-out/sibling path references.
- Runtime declarations record `qrel_files_read=false` and
  `test_files_read=false`.
- Shared source, evaluator, protocol, reports, data, and user worktree changes
  were not rewritten by C01.  No commit or cleanup was performed.
- Integrity failure found post-run: train/internal D2p was approximated instead
  of materialized from the registered fine-tuned D2t checkpoint.  This failure
  is explicitly outcome-invalidating and is the reason for the final stop.

## Final conclusion

**`stop`**.

C01 is neither advanced nor declared mechanistically falsified.  The probe is
implementation-invalid, the two-attempt budget is exhausted, and dev remains
untouched.  The only valid next action is a coordinator decision: either close
C01 permanently, or explicitly authorize and re-lock a corrected probe whose
train/internal adapter materializes seed-20260708 D2t scores from the frozen
query-tower checkpoint and composes registered D2p exactly before any further
outcome is inspected.  This report itself requests no additional budget.
