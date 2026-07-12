# G1 executable protocol amendment

Status: pre-outcome; this amendment resolves underspecification in `GATE.md`
without changing its seeds, sizes, controls, thresholds, or stop rule. If prose
and executable constants disagree, the execution lock must fail; neither may be
changed after a learned run begins.

## 1. Underspecification audit

The original G1 gate fixed the broad task and thresholds but did not determine:

- how topics, styles, targets, exact repeats, and supported non-repeats are
  sampled;
- how sequence order makes shuffled history causally relevant;
- independent RNG streams for data, corruptions, initialization, and batches;
- which fields each corruption may alter;
- the exact attention and extra-FFN controls or how their parameter counts and
  initial values are matched;
- the loss, batch size, optimizer, learning rate, clipping, checkpoint choice,
  score tie-break, target margin, and corruption-retention formula;
- deterministic CPU settings and one-shot output behavior.

All are fixed below before any learned G1 outcome exists.

## 2. Constants and RNG streams

The three independent experiment seeds remain `20260711`, `20260712`, and
`20260713`. For base seed `S`, every named CPU `torch.Generator` receives
`S*1000+offset`:

| Stream | Offset | Exclusive use |
|---|---:|---|
| `train_structure` | 101 | train type/topic/style/position/permutations |
| `train_values` | 102 | train candidate/history amplitudes |
| `train_schedule` | 103 | the shared 400-batch schedule |
| `eval_structure` | 201 | evaluation structure |
| `eval_values` | 202 | evaluation amplitudes |
| `wrong_history` | 301 | one nonzero cyclic donor offset |
| `shuffle_history` | 302 | per-request event permutations |
| `model_initialization` | 401 | the single shared initial state |

No operation uses global RNG except model construction immediately after
`torch.manual_seed(S*1000+401)`. Dropout is zero. Execution sets one intra-op
thread, one inter-op thread, and deterministic algorithms.

Frozen dimensions are 16 latent coordinates, 8 candidates, and 8 history
events. Each seed has 4,096 train requests and 1,024 independently generated
evaluation requests, exactly half repeat and half supported non-repeat. The type
vector is balanced first and then permuted by the split's structure stream.

## 3. Clean generator

There are four topics `t=0..3` and two styles `s in {-1,+1}`. Semantic item ID
is `2*t + 1[s=+1]`; every candidate set contains all eight IDs exactly once and
is independently permuted. Coordinates `0..3` encode topic, `4..7` encode the
topic-specific signed style, and `8..15` are zero.

For every request and semantic ID, a candidate vector has

```text
x[t]   ~ Uniform(0.90, 1.10)
x[4+t] ~ s * Uniform(0.65, 0.85)
```

and is L2-normalized. A non-repeat history vector has

```text
h[t]   ~ Uniform(0.97, 1.03)
h[4+t] ~ s * Uniform(0.50, 0.60)
```

and is L2-normalized. Independent continuous draws make accidental bitwise
candidate/history equality probability zero; this is also asserted mechanically.
The query is exactly the unit topic basis vector. A shared learned history
position embedding `0..7` is added in every control before the lower Transformer.

Each initial history contains one non-repeat vector for every semantic ID and is
permuted. Construction then depends on request type:

- **Exact repeat:** draw target style uniformly; choose the target candidate of
  the query topic and that style; draw one history position uniformly and replace
  its vector/topic/style with an exact bitwise copy of that candidate. The target
  is the unique candidate with an exact history recurrence.
- **Supported non-repeat:** make no replacement. For the two history events of
  the query topic, the event with the larger sequence position defines the
  preferred style. The target is the candidate with query topic and that latest
  style. Thus query identifies the relevant topic, history order identifies its
  preferred style, and no candidate occurs exactly in history.

Only clean requests are used for training. The synthetic target index is used by
cross-entropy; there is no repository label or evaluator.

## 4. Evaluation corruptions

Corruptions apply only to the 512 supported non-repeat evaluation requests.
Query, candidates, candidate IDs, original synthetic target, request index, and
request type remain fixed unless explicitly named.

- **Wrong history:** draw one integer offset uniformly from `[1,511]`; request
  `i` receives history `(i+offset) mod 512`. No request is its own donor.
- **Shuffled event:** draw an independent `[512,8]` matrix of `Uniform(0,1)`
  keys and stable-sort each row; apply the resulting permutation to history,
  mask, topic, and style arrays. Targets are not recomputed.
- **Query mask:** replace the 16 query coordinates with exact zeros; change
  nothing else.
- **Disjoint axes:** clean inputs occupy only coordinates `0..7`; replace every
  history vector `[a,0]` by `[0,a]`. This is the frozen disjoint-support control.
- **Empty history audit:** on the first 64 supported requests, replace history
  and history mask with exact zeros and compare RWPU to its paired query-only
  method bitwise. This is a contract audit, not a corruption margin.

## 5. Architecture and exact controls

All four methods are one `G1Ranker` class with identical named parameters,
shapes, values, lower/upper `TransformerEncoderLayer`s, role and history-position
embeddings, scoring head, optimizer, examples, and batch order. Width is 16,
heads 4, FFN width 32, dropout zero. The only Python constant that differs is the
read equation:

1. **RWPU:** the locked nonlinear `P^-1 W^-1 P W` residual.
2. **Ordinary:** the same history axes/strengths form only terminal `Wz0`; the
   candidate projects that terminal delta on the same probe axes. Probe gains and
   biases scale the two projected streams. No inverse path is used.
3. **Attention:** the same two history/probe axis projections form scaled
   dot-product logits over the eight events; masked softmax pools history axes.
   The same strengths, gains, biases, seed, and output projection participate.
4. **Pooled FFN:** masked-mean history plus query-conditioned candidate pass
   through the same two axis matrices, strengths, gains, biases, seed, and output
   projection, without an eventwise read.

No padding-only parameter exists. A pre-execution test requires every parameter
of every control to receive a finite nonzero gradient on the same 32 examples.
It also requires identical parameter names/shapes/counts and a bytewise tensor
state hash at initialization.

## 6. Training and selection

For each seed, one batch schedule is generated once and reused verbatim by all
four controls. Batch size is 64. Every epoch uses one `randperm(4096)`; its 64
contiguous batches are consumed, then another permutation is drawn until exactly
400 optimizer steps exist. There is no validation, early stopping, checkpoint
selection, retry, or sweep. Metrics use the parameters after step 400.

All methods minimize 8-way cross-entropy using AdamW with learning rate `0.003`,
betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`, and global gradient
norm clipping at `1.0`. No scheduler or mixed precision is used. NaN/Inf loss,
gradient, or score fails the seed and the one-shot run.

## 7. Metrics, ties, and frozen decision

Primary synthetic metric is top-1 accuracy on the 512 supported non-repeat
requests. Exact-repeat accuracy and mean target margin are secondary. For each
request,

```text
margin = target score - max score among the other seven candidates.
```

Scores sort descending. Exact score ties sort by ascending SHA-256 integer of

```text
S + ":eval:" + request_index + ":" + semantic_candidate_id
  + ":c08_g1_tie_v1"
```

so input order cannot decide a tie. Metrics are unweighted request means.
Corruption retention is `corrupt mean margin / clean supported mean margin`;
negative retention is allowed and passes the upper-bound condition. A zero or
negative clean margin independently fails.

For each seed, pass requires the original `GATE.md` thresholds exactly:

1. RWPU repeat accuracy `>= item_recurrence_accuracy - 0.01`;
2. RWPU supported accuracy minus the best of ordinary/attention/pooled-FFN
   `>=0.05`;
3. RWPU supported accuracy minus ordinary `>=0.03`;
4. clean supported mean margin `>0` and every corrupted/clean margin ratio
   `<=0.25`;
5. empty-history scores bitwise equal query-only;
6. reversed candidate order produces reversed scores with max error `<=1e-6`;
7. all training quantities and scores finite.

G1 passes only if all conditions pass in all three seeds. Otherwise C08 stops
before repository data, GPU, dev, or test.

## 8. One-shot execution boundary

`G1_EXECUTION_LOCK.json` hashes `PRE_OUTCOME_LOCK.json`, this amendment, the
model/protocol source, runner, independent verifier, and both test files. It also
duplicates all executable constants and binds their aggregate hash. Before
execution, both `pytest` and `verify_g1_lock.py --phase pre` must pass.

The runner accepts no seed, model, optimizer, or threshold override. It requires
`CUDA_VISIBLE_DEVICES=""` and refuses to start if its hash-derived ignored run
directory already exists. It writes a start marker before training, one raw JSON
per seed, and one completion marker. Failure writes a failure marker; it does not
authorize a rerun. Raw artifacts stay under candidate-local ignored `runs/`.
