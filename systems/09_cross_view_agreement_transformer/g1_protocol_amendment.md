# C09 G1 Executable Protocol Amendment

Status: normative amendment frozen before any trained/optimized synthetic
ranking outcome.  It resolves execution-relevant gaps in `pre_outcome_gate.md`
without changing that gate's thresholds.  If this document and the executable
runner disagree, execution must stop; neither source silently wins.

## 1. Underspecification audit

The original G1 text fixed seeds, dataset sizes, broad generator semantics,
controls, a 200-step ceiling, and pass thresholds, but did not uniquely fix:

1. RNG implementation, stream separation, or draw order;
2. the exact base and dual-causal utility constants;
3. label construction and tie handling;
4. the latent-input shared Transformer architecture;
5. the precise fusion equation for every named control;
6. whether “parameter matched” meant allocated or outcome-active capacity;
7. optimizer, batch order, clipping, numeric precision, and determinism flags;
8. where factor flips and history shuffling intervene;
9. pair-accuracy ties, corruption-retention denominators, and seed aggregation;
10. one-shot execution and artifact behavior.

Sections 2--12 close every item.  No choice remains to the runner CLI except
the already frozen output path.

## 2. Runtime and RNG

- device: CPU only; execution command binds `CUDA_VISIBLE_DEVICES=""`;
- dtype: PyTorch `float32` for model/data, Python `float64` only when aggregating
  JSON metrics;
- PyTorch deterministic algorithms: enabled with `warn_only=False`;
- intra-op and inter-op thread counts: 1;
- data RNG: `torch.Generator(device="cpu")` using MT19937 as implemented by the
  locked PyTorch runtime;
- train data stream seed: `s`;
- validation data stream seed: `s + 100000`;
- batch-order stream seed: `s + 200000`;
- shuffled-history Sattolo stream seed: `s + 300000`;
- model initialization: reset global CPU RNG to `s + 1000003` before *every*
  method, so all methods for one seed start from identical tensors;
- method execution order:

```text
base_raw, base_matched, q_only, c_only, view_mean, poe,
global_scalar, diagonal, ordinary_attention, constant_cma, cma
```

For each data split, random tensors are consumed exactly in this order:
`query`, `history`, `candidates`.  No label noise or rejection sampling occurs.

## 3. Exact generator

For each request independently draw

```text
q in R^8, h_t in R^8 for t=1..6, c_i in R^8 for i=1..8
```

with IID `N(0,1)` coordinates from the split's generator.  Let

```text
h_bar = (1/6) sum_t h_t
d = (1,-1,1,-1,1,-1,1,-1)
B_i = <q elementwise*d, c_i> / sqrt(8)
a = tanh(<q, h_bar> / sqrt(8))
g_i = tanh(<c_i, h_bar> / sqrt(8))
U_i = 0.35 B_i + 4.0 a g_i.
```

`B_i` is the query-candidate bilinear base.  `4 a g_i` is the dual-causal
history residual.  Its sign changes when either factor alone changes sign, so
neither `a` nor `g_i` determines the residual sign.  The target class is
`argmax_i U_i`; PyTorch's first-index rule decides an exact utility tie.  Pair
metrics use the continuous `U`, not the target class.

Per seed, generate exactly 2,048 training and 512 validation requests, each
with 8 candidates, latent width 8, and history length 6.

## 4. Shared latent Transformer

Every method instantiates the identical `G1SharedTransformer`:

- one shared bias-free `Linear(8,8)` input projection for `q`, `h_t`, and `c_i`;
- one learned six-position history embedding;
- one `TransformerEncoderLayer(d=8, heads=2, FF=32, GELU, dropout=0,
  batch_first=True, norm_first=True)` for history;
- one shared `MultiheadAttention(d=8, heads=2, dropout=0)` used both for the
  Q-first and C-first mediators;
- learned rank token, null-history mediator, and four role embeddings;
- one shared rank `TransformerEncoderLayer` with the same dimensions;
- one shared `LayerNorm(8)` and bias-free `Linear(8,1)` score head;
- one bias-free fusion value `Linear(8,8)` and fusion output `Linear(8,1)`;
- two learned scalar raw parameters `rho` and `tau_raw`, both initialized to 0.

The Q-first mediator attends from projected `q` to encoded history and is
candidate-blind.  The C-first mediator attends from each projected `c_i` to the
same history and is query-blind.  Base, Q-first, and C-first sequences are
`[RANK,q,c_i,mediator]` and use the same rank Transformer/head.  They yield
`b_i`, `u_i`, `v_i`, and base rank state `z_i`.

All methods compute all three rank passes.  All matched methods execute exactly
one `N x N` fusion aggregation using the same fusion value/output projections.
Thus they have identical allocated parameters, identical active parameter
groups, three rank passes, and one leading-order `O(N^2 d)` fusion.  The raw
base is intentionally labeled unmatched and is not used as the sole capacity
control.

## 5. Common notation and matched carrier

Define

```text
r_i^Q = u_i-b_i                 r_i^C = v_i-b_i
m_ij^Q = r_i^Q-r_j^Q            m_ij^C = r_i^C-r_j^C
lambda = softplus(rho)           temperature = softplus(tau_raw)+1e-4
v_ij = W_o W_V(z_i-z_j).
```

For any nonnegative off-diagonal strength matrix `S`, define null-sink
attention and aggregation

```text
A_ij(S) = S_ij / (1 + sum_l S_il),  S_ii=0
F_i(S,V) = lambda sum_j A_ij(S) V_ij.
```

The history-free matched carrier is

```text
S^B_ij = exp(clamp(temperature*(b_i-b_j), -12, 12)), i != j
C_i = F_i(S^B, v).
```

It makes value/output projections and an `N x N` aggregation outcome-active in
pointwise controls.  It is *not* added to CMA or other pair-attention controls,
which already spend the same active capacity/leading compute on their own
single aggregation.

## 6. Exact method equations

For all methods, final history correction is structurally masked with
`available = query_present AND history_present` via `torch.where`; unavailable
records return their own base `b` bit exactly.

### Proposed and pair-attention methods

```text
kappa(x,y) = x_+ y_+/(x_+ + y_+) when x_+ + y_+ > 0, else 0

cma:
  S_ij = kappa(temperature*m_ij^Q, temperature*m_ij^C)
  s_i = b_i + F_i(S,v)

q_only:
  S_ij = relu(temperature*m_ij^Q)
  s_i = b_i + F_i(S,v)

c_only:
  S_ij = relu(temperature*m_ij^C)
  s_i = b_i + F_i(S,v)

ordinary_attention:
  S_ij = exp(clamp(temperature*(m_ij^Q+m_ij^C)/2, -12, 12))
  s_i = b_i + F_i(S,v)

constant_cma:
  S is the CMA strength
  v_const = W_o W_V(rank_token), identical for every ordered pair
  s_i = b_i + F_i(S,v_const).
```

`constant_cma` preserves permission strengths but destroys cross-candidate
contrast values; it detects collapse to confidence gating.

### Pointwise methods with active matched carrier

Let `r_bar_i=(r_i^Q+r_i^C)/2` and
`alpha=sigmoid(tau_raw)`.

```text
base_matched:    s_i = b_i + C_i

view_mean:       s_i = b_i + C_i
                       + lambda*(alpha*r_i^Q + (1-alpha)*r_i^C)

poe:             s_i = b_i + C_i
                       + lambda*temperature*(r_i^Q+r_i^C)

global_scalar:   e = sqrt(mean_i r_bar_i^2 + 1e-12)
                 g = sigmoid(temperature*e)
                 s_i = b_i + C_i + lambda*g*r_bar_i

diagonal:        g_i = sigmoid(temperature*abs(r_bar_i))
                 s_i = b_i + C_i + lambda*g_i*r_bar_i.
```

`poe` is the residual-logit form of multiplying the two view distributions
after dividing out one duplicate base distribution.  `global_scalar` uses one
request scalar; `diagonal` uses a candidate-local scalar.

### Unmatched reference

```text
base_raw: s_i=b_i.
```

It still computes and trains base/Q/C passes under the common objective, but
its fusion parameters are intentionally inactive.  `base_matched`, not
`base_raw`, answers the capacity objection.

## 7. Corruptions

All corruptions retain the clean validation utilities/labels.

- `query_factor_flip`: after the shared forward, replace `r^Q` by `-r^Q`
  before CMA; `r^C` is unchanged;
- `candidate_factor_flip`: replace `r^C` by `-r^C`; `r^Q` is unchanged;
- `all_pair_disagreement`: replace `r^C` by `-r^Q`; therefore every oriented
  CMA pair is blocked exactly;
- `shuffled_history`: replace the validation history tensor by a Sattolo
  single-cycle permutation across requests, rerun the full model, and retain
  clean utilities.  Sattolo iterates `i=511..1`, draws
  `j=torch.randint(0,i)` from seed `s+300000`, and swaps positions `i,j`;
- `no_history`: keep tensors but set `history_present=False`;
- `query_masked`: replace `q` with zeros and set `query_present=False`; compare
  final logits with the base produced from that same zero query.

No corruption is used in optimization.

## 8. Optimization and batches

- optimizer: AdamW, learning rate `0.003`, betas `(0.9,0.999)`, epsilon
  `1e-8`, weight decay `1e-4`;
- batch size: 64;
- steps: exactly 200 per method and seed;
- batch stream: concatenate successive `torch.randperm(2048)` permutations
  from the batch generator; consume 64 indices per step, generating a new
  permutation only when the current epoch is exhausted;
- gradient clipping: global L2 norm at 1.0 after every backward;
- no scheduler, warmup, early stopping, checkpoint selection, or sweep;
- loss at every step:

```text
CE(s,y) + CE(b,y) + 0.5 CE(u,y) + 0.5 CE(v,y).
```

The first backward pass records gradient norms for these groups: shared input,
history Transformer, mediator MHA, rank Transformer/head, fusion value/output,
fusion scale, fusion temperature.  Every group must be nonzero and finite for
all matched methods.  All matched methods must have identical total parameter
counts.  Any mismatch is an execution-integrity failure, not a tunable result.

## 9. Metrics and ties

Primary pair accuracy examines all 28 unordered candidate pairs per request.
For utility difference `du` and score difference `ds`:

- exclude a true tie when `abs(du)<=1e-8`;
- award 1 when signs agree;
- award 0.5 when `abs(ds)<=1e-8` but `du` is non-tied;
- award 0 otherwise;
- average equally over all retained pairs and requests.

Also record top-1 accuracy, finite-loss status, parameter counts, active-group
norms, and exact fallback mismatch counts.  These are diagnostic only.

## 10. Frozen G1 decision implementation

Let accuracies be fractions and “points” mean absolute fraction differences.
For each seed:

```text
simple_best = max(base_raw, base_matched, q_only, c_only,
                  global_scalar, diagonal)
cma_simple_surplus = cma-simple_best
cma_attention_surplus = cma-ordinary_attention
clean_surplus = cma-base_raw
corruption_retention_x = (cma_x-base_raw)/clean_surplus.
```

If `clean_surplus<=0`, every corruption-retention criterion fails.

G1 passes iff all original criteria hold under these exact aggregations:

1. `cma_simple_surplus>=0.05` in at least 2/3 seeds and its three-seed mean is
   `>=0.05`;
2. `cma_attention_surplus>=0.02` in at least 2/3 seeds;
3. all-pair-disagreement correction is bit-exact zero for every request/seed;
4. query-factor-flip, candidate-factor-flip, and shuffled-history retention are
   each `<=0.25` in every seed;
5. `cma-constant_cma>=0.01` in every seed;
6. no-history and query-masked mismatch counts are zero in every seed;
7. every loss/metric/gradient is finite, every matched active group is nonzero,
   and parameter/compute contracts hold.

`view_mean` and `poe` remain reported controls; matching or beating CMA narrows
interpretation but was not an original numeric stop criterion and therefore is
not added post hoc.

Terminal decision is exactly `PASS_G1_REQUEST_D0_REVIEW` if all seven pass,
otherwise `STOP_C09_G1_FAILED`.  No rerun, threshold change, or new control is
allowed after seeing results.

## 11. One-shot execution and artifacts

The locked command is

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
python systems/09_cross_view_agreement_transformer/g1_runner.py
```

The runner first verifies `G1_EXECUTION_LOCK.json`, then transitively recomputes
every per-file and combined hash named by `PRE_OUTCOME_LOCK.json`.  It refuses
execution if either `g1_artifacts/STARTED.json` or `g1_artifacts/g1_raw.json`
exists, writes the start marker, runs the three seeds once, and writes raw JSON
to the ignored `g1_artifacts/` directory.  A concise tracked terminal decision
may be created afterward, but no normative/locked file may change.

## 12. Outcome boundary

G1 uses no repository dataset, cohort, label, qrels, dev evaluator, test split,
or GPU.  Passing is only permission to request D0 review; failing terminates
C09.  The synthetic generator is deliberately mechanism-shaped, so even a
pass is not paper evidence.
