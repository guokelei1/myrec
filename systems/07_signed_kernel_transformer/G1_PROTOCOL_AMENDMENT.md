# C07 G1 Executable Protocol Amendment

Date: 2026-07-11
Status: to be hashed by `G1_EXECUTION_LOCK.json` before any semantic run
Purpose: resolve every execution blocker without modifying the original lock or
its seven normative files

This amendment has precedence only where `pre_outcome_gate.md` was explicitly
underspecified.  It does not change its seeds, dimensions, optimizer, training
budget, thresholds, pass rules, stop rules, or authorization boundary.

## 1. Execution boundary

The only authorized command after the execution lock is verified is:

```bash
CUDA_VISIBLE_DEVICES="" python \
  systems/07_signed_kernel_transformer/g1_runner.py \
  --execution-lock systems/07_signed_kernel_transformer/G1_EXECUTION_LOCK.json \
  --output artifacts/runs/20260711_c07_pdsk_g1_cpu/result.json
```

The runner has no data-path option and imports no repository evaluator.  It
must verify the execution lock and exact output path before constructing an RNG
or model.  It refuses an existing output.  Device is literal `cpu`; one intraop
and one interop thread are used.  The semantic run is exactly the three frozen
seeds and may not resume, sweep, or select checkpoints.

## 2. Constants and tensor layout

```text
C=5, H=8, d=16, tau=0.50, kappa=1.00
train size/world/seed=4096, held-out size/world/seed=4096
Transformer layers=2, heads=4, FFN width=32, dropout=0
training dtype=float32, algebra-audit dtype=float64
evaluation batch size=256
semantic coordinates=0..4, common-mode coordinate=5
nuisance coordinates=6..15
candidate semantic amplitude=1.0, candidate common amplitude=1.0
candidate/query/history nuisance standard deviation=0.05
history common shift ~ Uniform[-1,1]
```

Candidate `i` has `candidate[i,i]=1`, `candidate[i,5]=1`, and independent
`Normal(0,.05)` nuisance coordinates.  Candidate masks are all true.  Labels
are uniform integers in `[0,4]`; the foil is sampled uniformly from the other
four identities.  All exact-match tensors are zero except where stated.

The generator's pre-model evidence audit is

\[
  g_{ij}=\sum_{r=0}^{5}q_r c_{ir}h^{key}_{jr}+e_{ij}.
\]

An event is oracle-active iff `max_i(g_ij)-min_i(g_ij) > tau` (strict).  The
audit is not a score given to any model.

Each history event has two tensors: `history_key` enters the shared Transformer
and Q/K evidence projections; `history_value` enters the shared value
projection.  This is the ordinary distinct K/V-field abstraction of one
history event and makes the frozen event/value corruption executable.  All
methods receive identical tensors.

## 3. Counter-based RNG streams

All data and schedule draws use `torch.Generator(device="cpu")`.  A stream with
integer ID `r` uses seed

```text
stream_seed = seed * 1000 + r
```

and is never shared with another stream.  Stock module initializers use the
PyTorch global **CPU** RNG reset to the stream-401 seed immediately before each
method.  IDs are fixed:

| ID | Stream |
|---:|---|
| 101/102/103/104 | train R/S/U-base/N tensors |
| 201/202/203/204 | held-out R/S/U-base/N tensors |
| 301/302/303 | R/S/N training index schedules |
| 304/305/306 | wrong-history/shuffled-event/query-masked U index schedules |
| 307 | within-batch permutations |
| 401 | model initialization, reset identically before every method |

Tensor draws occur in the exact source-code order in `make_world`.  The runner
records a SHA-256 over every generated held-out tensor per seed.  Training and
held-out streams are disjoint.  Python `random`, NumPy RNG, and CUDA RNG are not
used.

## 4. Exact worlds

For every world, first draw labels, foils, candidate nuisance, query nuisance,
history-key nuisance, history-value nuisance, and history common shifts in that
order.  Semantic coordinates not overwritten below are initially zero.

### R — exact recurrence

- Query semantic amplitudes are `q[y]=q[foil]=0.55`; `q[5]=1`.
- Draw recurrence event `r ~ Uniform{0,...,7}`.
- For all events and semantic coordinates, draw key distractors
  `Uniform[-0.30,0.30]`.
- Overwrite `history_key[r,y]=1.25`.
- Draw all semantic history values `Normal(0,0.10)` and overwrite
  `history_value[r,y]=1.0`.
- Set `exact_match[y,r]=1`.

Every request must have at least one oracle-active event.  Every non-recurrence
event must have oracle range `<=tau`; otherwise generation fails closed.

### S — supported non-repeat

- Query semantic amplitudes are `q[y]=q[foil]=0.80`; `q[5]=1`.
- Draw a uniform permutation of the eight events.  Its first two positions are
  supports, the third is the contradiction, and the remaining five are the
  sub-threshold slice.
- Draw every semantic history key `Uniform[-0.25,0.25]`.
- Overwrite each support key with `history_key[j,y]=1.25`.
- Overwrite the contradiction with `history_key[j,foil]=0.85`.
- Draw every semantic history value `Normal(0,0.10)`.
- Overwrite support values with `history_value[j,y]=1.0` and the contradiction
  value with `history_value[j,foil]=0.80`.
- Exact-match remains zero.

The target `y`, foil, and event roles are fixed before distractor/noise draws.
Both support events and the contradiction must be oracle-active; all five
sub-threshold events must have range `<=tau`.

### U — three deterministic corruptions of one S-shaped base

The U-base stream constructs records exactly as S.  Its target and foil remain
the labels for all corruptions.

1. `wrong_history`: pair each request with the first cyclic donor after a
   stream-103 random nonzero start offset whose donor target is in neither the
   target nor foil set and whose index differs.  Copy the donor's complete key,
   value, common shifts, masks, and exact tensor; keep recipient query,
   candidates, target, and foil.
2. `shuffled_event`: keep every key event fixed and replace value event `j` by
   original value event `(j+3) mod 8`.  This preserves the value multiset and
   its norms but breaks key/value pairing.  Common shifts remain on keys.
3. `query_masked`: keep history/candidates fixed.  Among shifts `1,2,3,4`, take
   the smallest whose shifted `{target,foil}` semantic set is disjoint from the
   original set, then cyclically shift query coordinates `0..4` by that amount.
   Common and nuisance coordinates are unchanged, so query norm is preserved.

For each corruption, its five designated sub-threshold event roles from its
source bundle must contain no oracle-active event under the corrupted query;
failure is fatal.  Active support events outside that slice are allowed—this is
what makes the corruptions nontrivial.

### N — no history

- Query has `q[y]=1.0`, `q[foil]=0.20`, and `q[5]=1`.
- `history_mask` is all false.
- `history_key[b,j,r]=1000*(-1)^(b+j+r)`.
- `history_value[b,j,r]=1000*(-1)^(b+j+r+1)`.
- Every exact-match cell is `1000`.

No oracle-active assertion is applied because every event is masked.  The
canaries must remain finite before and after the model call.

## 5. Common model and exact evidence score

Every method is the same `SyntheticRanker` class and therefore has the exact
same parameter count.  Reset stream 401 before constructing each method, so
all state-dict tensors are bitwise identical initially.  A mechanical assertion
checks both count and initial tensor hashes.

The shared two-layer Transformer encodes `[query, history_key, candidates]`
with the locked information-flow mask: query reads query only; history reads
query/history; candidates read query/candidates; history-to-candidate transfer
exists only in the tested branch.  No positional embeddings are used.

After contextualization:

\[
q'=\tanh(W_qq),\quad c'_i=W_cc_i,\quad h'_j=W_hh_j,
\]

\[
s_{ij}=\langle c'_i\odot q',h'_j\rangle/4
 + e_{ij}\operatorname{softplus}(\langle c'_i,q'\rangle/4).
\]

Exact-match scale is fixed to one.  Invalid history/candidate cells are set to
zero before normalization.  Values are `v_j=W_v history_value_j`.  The history
update is projected by shared `W_o`, added to candidate state, LayerNormed, and
scored by one shared bias-free linear head.

All methods own one trainable vector `theta in R^16`; it is the only role-varying
capacity and makes counts exact:

- PDSK, CENTER0, and DIFF_ATTN use the active diagonal candidate-local FFN
  `GELU(candidate_state_i) * theta` before final LayerNorm.
- ITEM_ONLY uses
  `theta * (GELU(candidate_state_i) + .1*GELU(W_h candidate_state_i))`, so its
  otherwise unnecessary history projection remains active without admitting
  cross-item history.
- BASE_FFN uses
  `theta * W_o GELU(W_c candidate_state_i + W_h candidate_state_i +
  W_v candidate_state_i + W_q query_state)`.  Thus every projection is active,
  but the branch remains candidate/query-only.
- GATED_CENTER splits it into two eight-dimensional parameter vectors for its
  active amplitude/temperature equations below.
- TARGET_NULL uses it as its learned null value.

No parameter is padding-only: a pre-lock unit test requires finite nonzero
gradients on every trainable tensor under a fixed hand-constructed loss.

## 6. Exact control equations

All candidate-axis softmaxes mask invalid candidates and all denominators use
the frozen `kappa=1`.

### PDSK

Use the locked equation with `tau=.5`, followed by global request-level L1/null
normalization over all candidate/history cells.

### CENTER0

Use exactly the PDSK equation with `tau=0`; this is the proved centered-linear
degeneration.  Use the same global L1/null normalization.

### GATED_CENTER

For valid events let `range_j=max_i(s_ij)-min_i(s_ij)`.  Form one request
summary

\[
r=\operatorname{mean}_i(c_i)\odot q'.
\]

Then

\[
A={\bf1}[\max_j range_j>.5]\,\sigma(\theta_{0:8}^Tr_{0:8}/\sqrt8),
\]

\[
T=0.10+\operatorname{softplus}(\theta_{8:16}^Tr_{8:16}/\sqrt8),
\]

\[
u_{ij}=A\left(\operatorname{softmax}_i(s_{ij}/T)-1/C\right),\quad
a=u/(1+\sum|u|).
\]

`A=0` when no history.  Amplitude and temperature are request scalars computed
from the same summary; there is no candidate-wise gate.

### TARGET_NULL

For each candidate normalize `[s_i1,...,s_iH,0]` over history plus one null.
Masked events have logit `-inf`.  History values are `v_j`; the null value is
the learned `theta`.  The update is their weighted sum.  If history is absent,
the entire update is explicitly zero after normalization.

### DIFF_ATTN

Split projected feature dimensions into `0..7` and `8..15`:

\[
s^1_{ij}=\langle c^1_i\odot q^1,h^1_j\rangle/\sqrt8
          + e_{ij}\operatorname{softplus}(\langle c'_i,q'\rangle/4),
\]

\[
s^2_{ij}=\langle c^2_i\odot q^2,h^2_j\rangle/\sqrt8,
\quad u_{ij}=\operatorname{softmax}_i(s^1_{ij})-
              \operatorname{softmax}_i(s^2_{ij}),
\]

with fixed subtraction coefficient one and global L1/null normalization.

### BASE_FFN

History update is identically zero.  The active candidate/query-local FFN uses
all projections and `theta` exactly as specified above.

### ITEM_ONLY

\[
u_{ij}=e_{ij}\operatorname{softplus}(\langle c'_i,q'\rangle/4),
\quad a=u/(1+\sum|u|).
\]

It uses the shared value/output path and its history-free active FFN.  It trains on the
same four-world batches but is evaluated only on R and N.

## 7. Training batches and order

R, S, and N each use two independently shuffled full epochs: concatenate two
`randperm(4096)` draws from their fixed index stream, then consume consecutive
chunks of 16 for 512 updates.

U contributes 16 requests/update.  Corruption counts rotate exactly:

```text
update mod 3 == 0: wrong/shuffled/query = 6/5/5
update mod 3 == 1: wrong/shuffled/query = 5/6/5
update mod 3 == 2: wrong/shuffled/query = 5/5/6
```

Each corruption has its own endless index stream made by concatenating fresh
`randperm(4096)` blocks as needed.  Concatenate R, S, U, N and apply the fixed
stream-307 `randperm(64)` for each update.  Thus the model receives no world or
corruption identifier.  All methods consume the identical saved index and
within-batch schedule.

Use exactly listwise cross-entropy, AdamW and the remaining training constants
in the original gate.  Check finiteness before/after every update.  No smoke run
is used; unit tests cover shapes.  Final update only is evaluated.

## 8. Metrics, ties, and audits

Evaluation uses consecutive held-out batches of 256.  Float32 finite logits are
required.

- Top-1 uses descending logit and, on exact equality, smaller candidate index.
- Target margin is `logit[target] - max(logit[non-target])`, averaged by
  request.
- A method's “history-induced” change compares its final logits with its own
  same-forward internal base logits before the history branch.  This is the
  only scale-valid interpretation of the fallback path.  The separately
  trained BASE_FFN accuracy is also reported descriptively but is not used for
  pointwise equality.
- U flip rate is the fraction whose final top-1 differs from its internal-base
  top-1.  Mean absolute logit change averages all requests/candidates.
- N logit mismatch is max absolute final-minus-internal-base difference;
  score-order and rank mismatch count requests whose stable full ordering
  differs.  PDSK must be exactly zero on all three; every other method is
  reported.
- Active-pair fraction on held-out S is the count of ordered valid
  `(i,k,j), i!=k` margins with `abs(s_ij-s_kj)>.5`, divided by
  `4096*8*5*4`.
- Nonzero-gradient fraction uses the first 64 held-out S records, final PDSK,
  and listwise CE.  It is the fraction of valid evidence-logit gradients with
  `abs(grad)>1e-12` among `64*5*8` cells.
- Conservation/common-mode audits cast all held-out S evidence logits to
  float64 and reapply the locked kernel.  Common shifts are
  `b_j=(-1)^j*0.37*(j+1)`.  Report maxima over all 4096 records.
- Permutation audit uses the first 256 S records, candidate permutation
  `[2,4,1,0,3]`, and history permutation `[7,0,5,2,6,1,4,3]`, permuting keys,
  values, masks, and exact-match axes together.  Report maximum aligned-logit
  error.
- S-minus-U is computed separately for all three corruptions; every difference
  must satisfy the frozen bound.

Three-seed means are arithmetic means because every seed has equal request
count.  Pass rules are evaluated in the numbered order in the original gate;
the first failure fixes the decision.  Threshold equality fails.  No metric,
rounding, or control may be changed after the execution lock.

## 9. Output and decision

The raw JSON contains locks/hashes, environment, generated-tensor hashes,
parameter assertions, final training losses, every per-seed/method/world metric,
aggregates, each boolean pass check, and the first stop reason.  It is written
atomically to the one authorized ignored artifact path.

After the run, a concise candidate-local Markdown report may summarize the raw
JSON verbatim.  It is an outcome file and therefore is intentionally excluded
from `G1_EXECUTION_LOCK.json`.
