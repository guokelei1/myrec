# B5o Stage A Split Decision Needed

Date: 2026-07-09

Scope: Batch 2b B5o repair attempt, external alignment only.

## What Is Fixed

The official-format materializer now exists at
`src/myrec/baselines/kuaisearch_materializer.py`.

It converts public raw KuaiSearch files into the locked official ranking loader
format:

- `rank_lite/train.jsonl` -> `data/rank.jsonl`
- `items_lite/train.jsonl` -> `data/corpus.jsonl`
- `users_lite/train.jsonl` -> `data/users.jsonl`
- `age_bucket` -> official `age`
- missing users are materialized with legal `gender`/`age` buckets so the
  official fallback index bug is not reached
- target item coverage is measured and written to the manifest
- embedding files are expected under `./data/`, matching
  `ranking/datasets.py`; no official source patch is needed for the path bug

Smoke evidence:

- command:
  `PYTHONPATH=src python -m myrec.baselines.kuaisearch_materializer --raw-dir data/raw/kuaisearch --output-root artifacts/batch2b/b5o_materializer_smoke --max-rank-rows 2000 --test-fraction 0.10 --min-target-coverage 0.999`
- output: `artifacts/batch2b/b5o_materializer_smoke/materializer_manifest.json`
- rows: 2000
- target item coverage: 1997/1997 unique target ids, rate 1.0
- users: 9/9 matched, no synthetic missing users, no invalid age/gender coercion
- official BGE process ran on the materialized files and embeddings were moved
  to `data/`
- official DNN 1-epoch smoke ran through train/valid/test without loader errors
  and reported test LogLoss 0.643912, AUC 0.377851

This smoke validates the adapter mechanics only. It is not a Table 7 alignment
run.

## Blocking Protocol Ambiguity

The public `rank_lite/train.jsonl` rows all carry `split=train` in the local
file. The locked official `ranking/datasets.py` treats only rows with
`split == "test"` as test, then randomly holds out 10% of the remaining rows as
valid.

The current local README and code do not provide the paper Table 7 test split.
The previous protocol-diff note says "paper says last day is test", but the
repo does not expose a day boundary or a split file. The public field is
`time_index`, not a calendar timestamp.

Because Table 7 alignment is judged on Logloss/AUC, the exact split can change
both class balance and AUC. Starting full DNN/DCNv2/DIN alignment with an
invented boundary would make the result hard to defend.

## Candidate Options

Option A: last-time 10% by `time_index`.

- Implementation already supported as `split_policy=last_time_fraction`.
- Pro: deterministic, temporal, close to our PPS split discipline.
- Con: not proven to be the official Table 7 split; ties at threshold can make
  the test fraction larger than exactly 10%.

Option B: explicit cutoff after confirming upstream boundary.

- Use `split_policy=last_time_cutoff --test-time-min <confirmed_min>`.
- Pro: best if the paper/authors can confirm the exact last-day boundary.
- Con: blocks full Stage A until upstream answer or hidden metadata is found.

Option C: random row split.

- Pro: matches the official loader's train/valid randomization style.
- Con: contradicts the stated last-day framing and is less defensible for a
  paper-number reproduction.

## Recommendation

Do not start the full Table 7 alignment training until the split policy is
explicitly authorized. If we need a bounded next attempt without upstream
confirmation, use Option A and label it `last-time proxy`, not exact official
reproduction.
