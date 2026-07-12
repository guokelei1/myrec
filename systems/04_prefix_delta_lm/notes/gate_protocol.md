# C04 preregistered gate protocol

Date frozen: proposal-lock time recorded in `proposal_lock.json`. Seed:
`20260708`. This document is written before any C04 GPU model outcome or dev
evaluation.

## 1. Stage and evidence boundary

Authorized now: proposal lock, unit tests, one train/internal materialization,
the minimal paired model and four controls, two deterministic 1,000-request
rescores, one complete label-free dev score file, exactly one primary shared
dev evaluator call, and a final report. The run then stops.

Not authorized: extra dev calls, hyperparameter changes from dev, multi-seed,
full gate, test, Amazon/JDsearch, full training, or a post-outcome rescue
module. Scoring/training never opens qrels. Test records, qrels, and metrics are
never accessed.

## 2. Frozen data and integrity

- dataset: KuaiSearch `v0_lite`;
- train labels: `records_train.jsonl` only;
- dev scoring: label-free `records_dev.jsonl` only;
- candidate SHA256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- structural populations, materialized before model outcome:
  4,110 no-history; 3,442 repeat-present; 4,677 non-repeat history-present;
- fixed seed: 20260708;
- environment/GPU: `myrec-c04`, physical GPU 3 only, code device `cuda:0`;
- run ID prefix: `20260710_kuaisearch_c04_`.

Any candidate-hash mismatch, label leak, missing candidate, duplicate score,
non-finite logit, extra primary evaluator call, test access, or run-prefix/GPU
violation invalidates the candidate.

## 3. Frozen architecture and training budget

- local backbone: four-layer `BAAI/bge-small-zh-v1.5` masked Transformer;
- prefix length: 128; query 16 tokens; last four history events, 12 tokens each;
  candidate 40 tokens;
- PEFT: rank-8, alpha-16 static LoRA on Q/V of the last two layers;
- paired operator: candidate-order tangent in `proposal.md`, clip `tau=1.0`;
- train/internal slice: 8,000 / 1,000 frozen-train requests, at most 12 fixed
  candidates each;
- main epochs: 2; control epochs: 1 on 1,500 / 500 requests;
- main learning rate 5e-4, AdamW, weight decay .01, batch 4 requests,
  accumulation 4, bf16;
- maximum implementation attempts: 2, where attempt 2 is allowed only for a
  numerical/contract bug, not a weak outcome;
- total budget: at most 8 A40 GPU-hours;
- screening online mean latency stop-loss: 250 ms/request including tokenization
  and both paired passes; peak allocated memory stop-loss: 20 GiB;
- online API/large-LLM calls: zero.

## 4. Unit and train/internal falsifiers

Unit tests must pass for shared-parameter deterministic logits, byte-identical
empty factual/null prefixes, exact zero no-history delta, candidate masks,
fixed candidate identity, deterministic item tokens, finite scores, candidate
hash assertion, and label/split guards.

Train/internal controls are frozen as:

1. matched structured single-pass `[q,H,c]` LM;
2. paired factual/null logits without the candidate-order tangent;
3. flat query/history/candidate concatenation plus ordinary ranking head;
4. ordinary static LoRA query/candidate ranker on the null path;
5. null LM plus exact-identity shortcut.

Before dev, the main null path must have train-internal pair concordance at
least 0.80 with the train-only D2p teacher. A non-finite loss, zero parameter
movement, or failure of shared-prefix identity is a numerical failure eligible
for the one implementation retry. Weak ranking/delta diagnostics are a
scientific failure and are not retryable.

## 5. One-call primary screening thresholds

The only primary run is
`20260710_kuaisearch_c04_prefix_delta_screen_s20260708`. The shared evaluator
is invoked under `flock tmp/pps_dev_evaluator.lock` exactly once. Candidate
code does not calculate NDCG; subset deltas use `scripts/compare_runs.py` over
the shared evaluator's per-request output.

All conditions below are conjunctive:

| Screen | Frozen threshold |
|---|---:|
| overall NDCG@10 | at least seed-08 D2p `0.3238158367` and no more than `0.010` below seed-08 item-only `0.3450873589` |
| 4,677 non-repeat present vs D2p | mean delta >= 0 and CI lower >= -0.001 |
| 3,442 repeat-present vs item-only | mean delta >= -0.003 |
| 4,110 no-history | zero rank mismatches vs seed-matched D2p |
| wrong/shuffle/query-mask/coarse structural delta | each mean-absolute delta <= 0.50 of factual delta on the frozen diagnostic sample |
| 1,000-request deterministic rescore | byte-identical scores (`max_abs=0`) |
| score coverage | exactly 12,229 requests / 575,609 candidates |
| evaluator log | exactly one C04 primary line |

The screening wrong-user diagnostic uses deterministic different-user
train-only donors and is only a cheap structural falsifier; it is not the
freshness-matched full-gate control. Likewise, corruption delta attenuation is
not a label-derived gain result. Passing this table only nominates C04 for a
separately authorized full gate.

## 6. Frozen full-gate thresholds (not authorized in this run)

If screening survives and the coordinator registers new budget before any
additional outcome, the full gate must use freshness-matched wrong histories
and shared evaluation for every scored variant. It requires:

- repeat-present CPDLR minus item-only mean >= 0, non-inferiority CI lower >=
  -0.001;
- non-repeat CPDLR minus D2p mean >= 0.006 and CI lower > 0;
- wrong, shuffled, query-masked, and coarse-only gain CI upper <= 0.001;
- no-history score/rank/metric mismatches = 0;
- matched single-pass and ordinary-LoRA controls cannot reproduce the
  non-repeat gain;
- tangent removal must lose the effect, or the novelty verdict becomes
  `reducible`;
- deterministic max absolute score difference = 0.

No subset or threshold may be changed after the screening. A failed condition
produces `pivot-before-more-dev` or `stop`; it does not authorize another dev
call or a rescue module.
