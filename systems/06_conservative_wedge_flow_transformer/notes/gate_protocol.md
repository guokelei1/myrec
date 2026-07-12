# C06 pre-outcome gate protocol

Status: **design protocol only; execution remains unauthorized**.

## Cohort hygiene

C06 must not tune on the exposed C05 cohort.  Selection uses structural arrays
and request IDs before opening any label-shaped file.

1. Recompute and blacklist the C05 selection whose manifest SHA256 is
   `e7e48c03e0e84ed779494beb1365fdcd6945c09cabd772553b3e281af6666eef`.
2. Define non-repeat from complete packed history, not the model truncation.
3. Rank remaining IDs by
   `sha256("c06-relative-v1\0" + role + "\0" + request_id)`.
4. Freeze:
   - 12,000 fit requests from the unused pre-cut non-repeat pool;
   - 1,200 `internal_A` requests from the unused post-cut non-repeat pool;
   - 600 delayed `internal_B` requests;
   - 515 escrow requests that no C06 process may open;
   - 512 post-cut no-history requests for label-free parity only.
5. Fit, A, B, escrow, no-history and every C05 ID must be pairwise disjoint.

The registered D2 epoch/alpha previously used the broader calibration partition,
so A/B are architecture falsifiers relative to D2p, not paper-level untouched
evaluation.  Dev remains the first possible positive-claim split and is not
authorized here.

## G0 - input and coordinate integrity

No optimizer exists.  Required:

- candidate manifest and every train/token input match registered hashes;
- selection is frozen before labels and has zero C05 overlap;
- every method uses full candidate sets;
- `(request_id, candidate_item_id)` alignment rejects duplicate, missing,
  unknown and non-finite rows;
- shuffled-row realignment is bit-identical with zero base rank mismatches;
- `internal_B` labels and escrow records are not materialized;
- current design, controls, thresholds, source hashes and environment are
  locked before G1.

## G1 - architecture contracts

Synthetic data and fit-only real states must establish:

1. `F=-F^T`, trusted `T=-T^T`, `0<=t_ij<=1`, valid `sum(delta)=0`, and
   `max|delta|<=rho_max+1e-7`;
2. candidate-common factors, no history, true `H=0`, and one candidate yield
   exact zero correction;
3. no-history scores equal the base bit-for-bit;
4. candidate permutation has max error `1e-6` and zero rank mismatches;
5. masked NaN/Inf cannot contaminate valid rows;
6. low-rank potential plus every candidate's gradient/cycle row energy match
   explicit FP64 edge materialization, including nearly coincident factors;
7. two real optimizer steps open scale then reach query/candidate/history/factor
   paths, with deterministic checkpoint reload;
8. a real multi-layer Transformer under the information barrier gives exactly
   zero history gradient to the base head and nonzero history gradient only
   through the opened wedge path;
9. a potential flow has `t≈1`; a pure cycle has `t=0`; adding a cycle at fixed
   raw potential lowers incident local trust; flipping only the cycle sign
   leaves trusted logits unchanged;
10. nested, duplicate and distractor pools remain finite and score-bounded;
    their changes are reported because no subset-independence or local
    influence bound is claimed.

An implementation failure may be repaired only before any A/B score exists.
The v1 Hodge-path fits exercised this allowance once: review1 permits only a
mathematically equivalent, primitive-error-bounded FP64 row fallback for three
failed variants, with distinct retry ledgers. It freezes the parent config/G0,
preserves the completed centered v1 control, and changes no scientific setting.

## G2 - one-shot architecture gate

Train exactly two fixed epochs, seed `20260708`, final checkpoint only, no early
stopping, no corruption training and no hyperparameter grid.  All variants use
the same 12,000 fit requests, full candidates, unified schema, listwise loss and
optimizer steps.

G2 is staged so the local-trust question is answered before paying for the
larger nearest-neighbor suite.

The minimal mechanism gate trains only:

- C06 candidate-local Hodge-trusted projected flow;
- the same projected-flow architecture with `t=1` everywhere;
- a parameter-matched direct learned candidate/event gate on the same projected
  potential;
- ordinary centered candidate-to-history cross-attention with the same final
  score trust region.

The rejected global event-level Hodge scalar is a same-factor, same-checkpoint
counterfactual and needs no additional training. Failure against any of these
four closes C06 before further controls.

CPU source status: `t=1`, global-event and direct learned-gate modes share the
same factor generator and low-rank energy path as the primary. The direct gate
adds only 32 parameters at the registered dimensions (well below 2%). The
centered cross-attention control has exactly the primary parameter count. Its
four load-bearing tied projection rounds exactly match the primary's frozen
dominant `C*H*r^2` accounting; GPU smoke still reports wall time and memory.
Neither symbolic matching nor smoke alone authorizes the real gate.

Only a complete minimal pass authorizes the deferred nearest-neighbor gate:

- history-null groupwise Transformer;
- pairwise Transformer with additive/Borda aggregation;
- potential-flow restriction `F_ik=u_i-u_k`;
- MIR/SetRank-style history-aware groupwise block.

Trainable parameters differ by at most 2%, theoretical FLOPs by at most 10%,
and dormant parameters cannot be used for matching. The review implementation
pre-registers G0/local/audit on GPU 0 and the three independently trained
controls on GPUs 1--3, but the environment, source lock, stage authorizations
and time budget still require separate approval before G0/G2.

### G2-A0 - label-free order audit

Score A without opening its labels.  Pass only if:

- two deterministic rescoring passes are bit-identical;
- common-mode ratio is at most `1e-5` per request;
- at least 10% of requests have
  `range(delta)>1e-3*rho_max`;
- at least 5% change some candidate order and 1% change top-10 membership;
- constructed candidate-common factors still yield exact zero correction;
- every candidate/event trust is finite and in `[0,1]`;
- a same-factor counterfactual with `t=1` changes score deltas on at least 5%
  of requests and changes some candidate order on at least 1%;
- replacing candidate-local `t_ij` by one scalar per event also changes score
  deltas on at least 5% of requests; otherwise locality is not load-bearing;
- all candidates remain present and every score is finite.

Nested, duplicated and distractor-augmented candidate pools are also scored
label-free and summarized beside every matched control. They are a required
diagnostic, not a pass threshold and not evidence of subset independence.

Failure stops before A labels are opened. Parameter movement or edge mass is
not a substitute.

### G2-A1 - shared ranking gate

Only after A0 may the shared metric implementation open A labels.  Use
request-equal NDCG@10, the registered tie-break and 10,000 paired bootstrap
samples.  C06 must simultaneously:

- improve over D2p by at least `+0.001`, CI lower bound above zero, with all
  three request-hash folds positive;
- improve over every matched control by at least `+0.0005`, paired CI lower
  bound above zero;
- have positive clicked-minus-unclicked score-delta mean and CI lower bound.

Failure closes the primitive.  A cannot tune `rho_max`, flow rank, learning
rate, epochs, history length or architecture.

### G2-B - delayed replication and authenticity

Only a complete A pass opens the frozen 600-request B cohort.  The unchanged
checkpoint must again beat D2p by `+0.001` with positive CI.  True history gain
must separately exceed held-out wrong-user, matched event replacement and
query-mask gains with paired CI lower bounds above zero.  Temporal shuffle is
binding only if learned time/order tokens are active.  These twins never enter
training.  The 515 escrow requests cannot rescue a failure.

## G3 - exact recurrence, only after G2

Add exact recurrence as a monotone conservative edge field; exclude exact
events from free semantic flow.  Verify autodiff/finite-difference monotonicity
and non-inferiority to the registered item-only control.  Passing G2 merely
permits a new G3 lock; it does not authorize dev, full training or test.

## Current authorization

| Stage | Authorized |
|---|---:|
| Design and CPU synthetic contracts | yes |
| Cohort materialization / G0 execution | coordinator-authorized; source lock pending |
| GPU smoke / four two-epoch fits / A0-A1 | coordinator-authorized; source lock pending |
| Delayed B / escrow | no |
| Dev evaluator calls | 0 |
| Full LM training | no |
| Test | no |
