# C05: Candidate-Contrastive Evidence Budget Transformer

Status: **pre-run review amended; G0 and G2a train-internal probe authorized on
physical GPU 0; dev/full training remain unauthorized**.

C05 asks one question before building another large proposed system:

> On history-present requests without an exact-repeat candidate, does history
> contain query-consistent evidence that distinguishes one candidate from the
> other candidates strongly enough to improve ranking over the registered D2p
> base?

The candidate's single primitive is the **Candidate-Contrastive Evidence
Budget (CCEB)**.  It replaces one ordinary history-to-candidate attention
update inside a Transformer ranker.  For each history event, support is
centered across the current candidate set; only support outside a fixed dead
zone receives a signed, bounded residual budget.  Generic history relevance
that supports every candidate equally produces exactly zero update.

This is deliberately smaller than C01--C04:

- no certificate head or calibrated global threshold;
- no hypernetwork or modification of all FFNs;
- no optimal transport or iterative solver;
- no factual/null second LM pass;
- no mandatory order-changing projection.

An independent pre-run review found that the first protocol still mixed signal
existence with CCEB mechanism attribution.  The sequence is now stricter:

1. `G2a` first trains a minimal ordinary target-attention probe on non-repeat
   train requests only, with exact recurrence and corruption training disabled;
2. held-out evidence-fidelity audits run only if that probe beats the clean D2p
   internal coordinate;
3. CCEB centering/signed/budget mechanisms are tested only after signal
   existence survives;
4. action/recency-aware monotone repeat protection and the full Transformer are
   deferred to the final gate.

The target-attention probe and the existing CCEB wrapper are **not final
proposed systems**.  They use a registered base coordinate only to falsify
non-repeat learnability cheaply.  A full Transformer ranker may be implemented
only after the frozen signal and mechanism gates pass; its complete query/item
base and ranking head must be frozen after exact no-history parity.

## Layout

```text
configs/c05_signal_probe.yaml  pre-outcome ladder and thresholds
model/cceb.py                  Transformer attention/residual primitive
model/signal_probe.py          minimal G2a target-attention probe
train/data.py                  frozen selection and all-candidate collation
train/materialize_g0.py        clean D2p/query-state G0 materializer
train/run_g2a.py               train-internal signal experiment
train/losses.py                ranking-aligned, empty-safe objectives
notes/proposal.md              hypothesis and simple-to-complex plan
notes/mechanism_fingerprint.md exact operator and degeneration controls
notes/nearest_neighbors.md     primary-source novelty audit
notes/gate_protocol.md         G0--G4 authorization and stop conditions
notes/proposal_lock_review1.json current review-amended pre-outcome lock
tests/                         synthetic contracts only
```

## Authorized commands

```bash
python -m pytest -q systems/05_candidate_contrastive_evidence/tests

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  CONDA_ENVS_PATH=/data/gkl/conda_envs conda run -n myrec-c05 \
  python systems/05_candidate_contrastive_evidence/train/materialize_g0.py \
  --config systems/05_candidate_contrastive_evidence/configs/c05_signal_probe.yaml \
  --device cuda:0

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  CONDA_ENVS_PATH=/data/gkl/conda_envs conda run -n myrec-c05 \
  python systems/05_candidate_contrastive_evidence/train/run_g2a.py \
  --config systems/05_candidate_contrastive_evidence/configs/c05_signal_probe.yaml \
  --device cuda:0
```

Only G0 materialization and G2a are authorized with environment `myrec-c05`,
physical GPU 0, run prefix `20260711_kuaisearch_c05_`, and a cumulative 2 A40
GPU-hour ceiling.  Do not score dev, call the shared evaluator, access test, or
train CCEB/full-system code until a later gate explicitly authorizes it.
