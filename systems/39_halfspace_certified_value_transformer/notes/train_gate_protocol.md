# C39 frozen Amazon train-internal gate

This protocol is frozen before C39 opens any real fit label or produces any
internal-A score. Numeric thresholds, modes, data roles, optimizer settings,
and stop rules may not change after the proposal lock.

## Evidence and outcome isolation

- Dataset: standardized Amazon-C4 temporal-history train only.
- C0 and C1 must pass and remain hash-bound.
- Fit: exactly the 6,000 C38 fit requests. They are reused only as training
  data; no C38 A score is reused by C39.
- Internal-A: 1,200 requests selected by a new request-key hash exclusively
  from C38's 1,599 `unused_indices`. They were not part of C38 feature
  collection, scoring, label opening, delayed-B, or escrow.
- Reserve: the remaining 399 C38-unused requests. Features, scores, and labels
  are not authorized.
- Wrong histories: another user in the same retained-history-length bin,
  selected without labels. Target category is unavailable.
- Upstream dev/test records, labels, qrels, and evaluator are forbidden.

The proposal lock binds the design-gate report, C38 terminal report and
selection, C39 selection, standardized files, C0/C1, model/config/source,
shared metrics, and BGE snapshot. Feature collection and BGE encoding read
only `records_train_blind.jsonl`. G0 may open fit labels only after the proposal
lock. Training begins only after an execution lock binds every feature array,
embedding, base score, fit-label array, and G0 report.

## Frozen model and controls

All five modes have 197,376 trainable parameters: shared 384-to-64 `Q/K/V`,
64-to-384 `W_O`, four heads, and a shared 384-to-128-to-384 Transformer FFN.
The BGE-small-en backbone is frozen. Every mode uses the same 6,000 requests,
all candidates, request order, paired seed initialization, one epoch, AdamW
learning rate `0.001`, weight decay `0.0001`, gradient clip `1.0`, and equal
listwise/direction loss weights. Candidate sampling is forbidden.

Modes are:

1. `eventwise_halfspace` primary;
2. `eventwise_raw` without value projection;
3. `postpool_halfspace` with projection moved after aggregation;
4. `ray_only` with the same immediate nonnegative score component but no
   score-neutral value representation;
5. `global_only` with only the query-attended unprojected history write.

The three seeds are 20261601/02/03 on physical GPUs 0/1/2. Correction scale is
fixed at 2.0 for every mode. Exact recurrence uses the common base boost 3.0
and suppresses all cross-item correction for that request.

## G0

All must pass:

1. design-gate report and C0/C1 pass with frozen hashes;
2. C39 fit equals C38 fit and C39 A comes only from C38 unused, with zero
   overlap with C38 internal-A/delayed-B/escrow;
3. all selected histories are nonempty and strictly prior to request time;
4. the target item is absent from every released history;
5. wrong donors have full coverage, zero same-user assignments, and at least
   95% same-length-bin matching;
6. each fit request has exactly one positive; internal-A labels remain closed.

## A0 label-free mechanism gate

All checks are binding before A labels may open:

- identical parameter count and paired initialization for all modes, with
  seed-specific initial states;
- finite training, updated parameters, and nonzero gradients through the
  primary `Q/K/V/W_O` and FFN;
- deterministic error exactly zero and candidate-permutation error at most
  `1e-6`;
- no-history, query-masked, and repeat-present corrections exactly zero;
- primary eventwise score-halfspace violation at most `1e-6`;
- at least 5% of active negative raw edges are changed by projection;
- at least 5% of candidate/event/head support edges are exact zero, and at
  least 50% of active requests contain both supported and rejected edges;
- primary changes at least 5% of complete orders and 1% of top-10 sets versus
  base;
- true versus wrong history changes at least 2%/0.5% of complete/top-10 order;
- primary differs from each of the four controls on at least 2% of complete
  orders and 0.5% of top-10 sets;
- no candidate scalar head, tangent projection, dataset/category/query-type
  branch, or dev/test/qrels access.

Any A0 failure is terminal and A labels remain closed.

## A1 utility and mechanism-rent gate

If and only if A0 passes, open the 1,200 A labels and use the shared NDCG@10
implementation with 10,000 paired bootstrap samples and three request-hash
folds. All must pass:

1. primary minus frozen BGE/exact-repeat base mean at least `+0.002`, every
   seed and fold positive, and 95% CI lower bound above zero;
2. primary minus `global_only` mean at least `+0.002`, every seed and fold
   positive, and CI lower bound above zero;
3. primary minus each of `eventwise_raw`, `postpool_halfspace`, and `ray_only`
   mean at least `+0.0005`, every seed nonnegative, and CI lower bound above
   zero;
4. true-history primary minus wrong-history primary CI lower bound above zero;
5. the candidate-specific correction `primary - global_only` has clicked-minus-
   negative mean correction CI lower bound above zero.

Any failure is terminal for C39. There is no threshold, projection strength,
head count, width, scale, loss, epoch, cohort, encoder, or candidate-pool
rescue. Even a complete A1 pass authorizes only a separately frozen KuaiSearch
confirmation; it does not authorize reserve/dev/test or a proposed-system
claim.
