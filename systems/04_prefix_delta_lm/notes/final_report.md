# C04 final report — Counterfactual Prefix-Delta Language Recommender

Date: 2026-07-11 (Asia/Shanghai)

Decision: **`stop`**. CPDLR failed its frozen pre-dev anchor condition before
the shared evaluator was called. The subsequent sole dev output is mechanically
valid but sequencing-invalid and is retained only as descriptive negative
evidence, not as an admissible primary-screening claim. No full gate, additional
dev call, multi-seed training, test evaluation, or rescue module is authorized.

## 1. Formula and information flow

One local BGE-small masked Transformer with one shared set of backbone, static
LoRA, embedding, and candidate-head parameters produces

\[
  h_c=f_\theta([q,H,c]),\qquad n_c=f_\theta([q,\varnothing_H,c]).
\]

For the complete fixed candidate pool, define

\[
  \Pi_n(x)=Cx-\frac{\langle Cx,Cn\rangle}{\|Cn\|_2^2+\epsilon}Cn,
  \qquad
  z=n+\Pi_n\left(\tau\tanh(\Pi_n(h-n)/\tau)\right),
\]

where `C` centers over candidates and `tau=1`. `z` is the final LM ranking
logit. Query, strictly-prior history event/type/item/text, and candidate
identity/text are tokens inside the Transformer. The candidate set is never
generated, expanded, or filtered. A frozen D2p model supplies only a
train-split KL order anchor for `n`; dev inference never reads or adds D2p.

The mechanism fingerprint is therefore: same-LM factual/null prefix logits,
candidate-order-tangent residualization, train-only null ordering anchor, and
corruption-to-zero consistency. The main model has 23,987,202 parameters,
33,282 trainable (rank-8 Q/V LoRA in the last two layers plus the scalar head).

## 2. Proposal and gate locks

| Artifact | SHA256 |
|---|---|
| `notes/proposal_lock.json` | `6bf5f14d86f44a61a461a1babe4f0c250c0d69238b6839a238dbfcd66042f920` |
| `notes/proposal_lock_attempt2.json` | `a359a5cbc70a96632bad2f72be84f68a4ee90194f0094e1d1a93ff923469d7d4` |
| initial config | `bacc243db1774af209b05b108631f83f05860b9b2039da50fc7ba94c579e91fd` |
| attempt-2 config | `fd64756f8ee1f9a25ef1faba82152b629f101aad3502e66234b9702395b27b5d` |
| candidate manifest | `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e` |
| screened scores | `9952950add845444b3fe777db0188ac4ceb3ac3d59ab86a81d64bee99a5db622` |
| screening audit | `b6d8617053b9b637b7e08200f7824173664a7e9a716e862dfd9a8e52f43bca42` |
| completion integrity audit | `509d8540110e03f17738077a4b9f3a8cec7b35982c6cccc89c198b785a1da78e` |

The initial lock covered all source, config, notes, tests, and the label-free
4,110/3,442/4,677 structural request lists before GPU outcome. It recorded zero
prior C04 dev evaluations. Attempt 1 exposed a numerical control-loss bug:
padding `-max` logits were summed and multiplied by zero when corruption was
absent, producing NaN. Attempt 2 changed only that zero construction and the
non-overwriting retry run ID; it was locked before any C04 dev outcome. The
failed attempt-1 run remains preserved and has no scientific interpretation.

## 3. Source, config, tests, and environment

- model/operator: `cpdlr/model.py`;
- fixed-prefix tokenizer: `cpdlr/tokenization.py`;
- train-only materialization/anchor: `cpdlr/materialize.py`;
- losses and execution: `cpdlr/losses.py`, `cpdlr/train.py`;
- label-free scoring and audit: `cpdlr/score.py`, `cpdlr/audit.py`;
- executable config: `configs/probe.yaml`;
- entry points: `scripts/materialize_protocol.py`, `materialize_probe.py`,
  `train_probe.py`, `score_dev.py`, `audit_screening.py`;
- tests: five files under `tests/`;
- environment: `myrec-c04` at `/data/gkl/conda_envs/myrec-c04`, Python 3.10.20,
  PyTorch 2.6.0+cu124, CUDA build 12.4, Transformers 5.12.1, pytest 9.1.1;
- hardware: physical NVIDIA A40 GPU 3 only, exposed as `cuda:0`.

The final CPU suite is **13 passed**. It covers shared-parameter deterministic
candidate logits, exact empty-prefix identity and zero delta, tangent geometry,
candidate masks, fixed/deterministic item identity tokens, label/split guards,
candidate-manifest assertion, and finite backpropagating loss. A synthetic
end-to-end load of the real D2 initialization completed paired/null/corruption
forward and backward with finite loss 0.94873.

The verified test invocation uses the candidate directory as its working
directory:

```text
cd systems/04_prefix_delta_lm
/data/gkl/conda_envs/myrec-c04/bin/python -m pytest -q -o cache_dir=.pytest_cache tests
```

The pre-outcome README's repository-root test example omits the required
candidate import path in a clean shell. It is preserved byte-for-byte under the
proposal lock; the command above is the post-outcome reproduction correction.

## 4. Commands, runs, and GPU budget

All GPU commands used `CUDA_VISIBLE_DEVICES=3`; model code used `cuda:0`.
The important command forms were:

```text
materialize_probe.py --device cuda:0
train_probe.py --mode <control|paired_delta> --device cuda:0
score_dev.py --limit-requests 1000 --no-diagnostics --output-dir <A|B>
score_dev.py
flock tmp/pps_dev_evaluator.lock python scripts/evaluate_scores.py --run-id 20260710_kuaisearch_c04_prefix_delta_screen_s20260708 ...
python scripts/compare_runs.py --request-ids <frozen subset> ...
audit_screening.py
```

PyTorch 2.6 required CUDA context initialization before its peak-memory reset;
the two deterministic and complete scoring invocations used the same locked
`score_dev` function after allocating one scalar on `cuda:0`. The failed
pre-initialization output directory is retained; it produced no scores or dev
outcome and did not change the model.

Run IDs:

- invalid numerical attempt: `20260710_kuaisearch_c04_single_pass_train_s20260708`;
- valid controls: `..._single_pass_train_s20260708_r2`,
  `..._paired_no_tangent_train_s20260708`, `..._concat_head_train_s20260708`,
  `..._static_lora_train_s20260708`, `..._identity_shortcut_train_s20260708`;
- main train/internal: `20260710_kuaisearch_c04_paired_probe_train_s20260708`;
- sole screening: `20260710_kuaisearch_c04_prefix_delta_screen_s20260708`.

Total charged GPU time, including the invalid attempt, train-only anchor,
controls, main training, both deterministic rescores, and full scoring, was
**0.207575 A40 GPU-hours**, below the frozen 8-hour cap.

## 5. Train/internal and determinism checks

The main model's final internal diagnostics were:

| Diagnostic | Result | Frozen condition |
|---|---:|---:|
| null-vs-D2p pair concordance | 0.63344 | >= 0.80 — **fail** |
| positive-negative final-logit margin | 0.02152 | descriptive |
| mean tangent delta, non-repeat | -0.001274 | positive transfer expected — **fail** |
| mean tangent delta, repeat | +0.01872 | recurrence direction positive |

The first row was explicitly frozen as a condition that had to pass *before*
dev. It did not pass, so C04's protocol-valid terminal decision was already
`stop` here. Continuing to the evaluator was a sequencing error; it cannot be
repaired by reinterpreting the later metrics or by spending the unused budget.

The small controls' positive-negative margins were 0.00771 (single-pass),
0.00643 (paired without tangent), 0.00503 (flat concat), 0.00619 (static
LoRA), and 0.00889 (identity shortcut). These controls used fewer requests and
one epoch, so their internal values are diagnostics, not a performance claim.
No control received a dev evaluator call.

Two frozen 1,000-request rescores each produced 42,968 rows with identical
SHA256 `9936186eae68c2d961820a90a4d746a3126ca2cb67d6ae7473c1113b03d45ad1`.
Thus deterministic label-free rescoring passed exactly.

## 6. Sole dev call and descriptive falsifier evidence

Despite the failed precondition above, the shared evaluator was called exactly
once. The call stayed within the global one-call maximum, but it violated the
self-frozen pre-dev sequence. Its sole log line is:

```json
{"method_id":"c04_prefix_delta_lm","ndcg@10":0.30881835019986664,"run_id":"20260710_kuaisearch_c04_prefix_delta_screen_s20260708","split":"dev","timestamp":"2026-07-10T16:57:46.128376+00:00"}
```

Aggregate metrics were NDCG@10 0.30881835, MRR 0.28649365, Recall@10
0.52999494, and purchase-NDCG@10 0.32851479 at 0.14008 coverage. The score
contract covered all 12,229 requests and 575,609 candidates.

The repository-wide static item-only waterline remains the three-seed mean
**0.3453755427**. The frozen single-seed screening used the seed-20260708
item-only value **0.3450873589** for its matched comparison. C04's 0.30881835
is below both; this distinction does not affect adjudication.

| Common/C04 falsifier | Evidence | Decision |
|---|---|---|
| overall not obviously degraded | 0.308818 < D2p 0.323816 and < item-only-minus-0.010 | **fail** |
| non-repeat transferable surface | CPDLR − D2p = -0.015395, CI [-0.023040, -0.007605], n=4,677 | **fail** |
| preserve repeat-present item-only | CPDLR − item-only = -0.082985, CI [-0.093061, -0.073091], n=3,442 | **fail** |
| no-history D2p identity | 4,098/4,110 rank mismatches; metric delta -0.020900, CI [-0.029049, -0.012772] | **fail** |
| corruption-to-zero | wrong 1.071×, shuffled 1.028×, query-masked 0.954×, coarse 0.823× factual mean-absolute delta; threshold <=0.50 | **fail all four** |
| exact recurrence only | internal non-repeat delta negative while repeat delta positive; dev both surfaces regress | shortcut-collapse warning; no cross-item mechanism supported |
| matched single-pass/tangent degeneration | internal controls completed; no extra dev call authorized | full attribution not run; primitive already fails earlier necessary conditions |
| deterministic scoring | byte-identical 1,000-request rescore | pass |
| efficiency | 20.85 ms/request including tokenization and paired passes; peak 1.542 GiB | pass |
| evaluation mechanics | candidate hash/coverage/evaluator-count/test lock all valid | pass |
| frozen pre-dev sequencing | anchor concordance 0.63344 < 0.80, then evaluator was called | **fail** |

The wrong-history diagnostic used different-user train-only donors and was
structural only; no label-derived corrupted gain is claimed. The full
freshness-matched corruption gate was not executed. Because of the sequencing
failure, all dev rows in this section are descriptive negative evidence and
must not be registered as a protocol-valid primary-screening result.

## 7. Nearest-neighbor verdict and claim boundary

Nearest-neighbor verdict remains **`uncertain`** at the design level. The raw
same-model conditional/null logit difference is algebraically reducible to
language-model classifier-free guidance and close to counterfactual logit
pairing. The candidate-order tangent was the only potentially irreducible
operator. Because CPDLR fails its necessary performance, evidence-fidelity,
no-history, and corruption conditions, there is no empirical basis to claim
that this difference is useful or novel. It must not be renamed or rescued by a
new prompt/template.

## 8. Latency/token cost, integrity, and terminal action

There are zero online API or large-LLM calls. At the observed mean candidate
count 47.069, paired scoring processes about 12,050 padded token positions per
request (`2 × candidates × 128`), followed by an `O(|C|)` projection. Complete
label-free dev scoring took 254.96 seconds, 20.85 ms/request, with peak allocated
memory 1.542 GiB on A40 GPU 3.

Candidate code never read qrels; only the shared evaluator read dev qrels. Test
records, qrels, and metrics were never accessed. Candidate hash, score coverage,
finite values, deterministic rescore, and exactly-one dev log entry all passed.
These are mechanical integrity checks only. Overall protocol integrity is
**failed** because the dev call followed a failed frozen precondition; the
scope distinction is recorded in `notes/completion_integrity_audit.json`.

**Terminal action: `stop`.** Do not advance C04 to the full gate, do not spend
multi-seed budget, do not access test, and do not add modules or relax subsets.
The admissible negative result is the train/internal anchor failure; the later
dev evidence agrees with `stop` but is not needed to establish it.
