# 19 - Fine-tuned Non-personalized Control Protocol

Status: locked after D1 adjudication and before D2 training or any D2 dev
evaluation on 2026-07-10.

## 1. Purpose and disclosure

D1 establishes that a supervised adapter over frozen text embeddings improves
the zero-shot query control but remains below the static query/history mixture.
Its encoder is frozen, so it is not a strong enough basis for a broad claim
about train-fitted non-personalized ranking. D2 is a post-D1 robustness control:
it fine-tunes a compact language-model query tower while keeping the data,
candidate set, evaluator, and item-text artifact fixed.

This extension is intentionally transparent. Its design was fixed after seeing
D1, but all D2 epochs, mixture weights, and retries are selected on train only.
No D2 dev metric may change this protocol.

## 2. Data and isolation

- Reuse the 96,939 positive train requests and 12,229 label-free dev requests
  in the hashed D1 packed artifact.
- Read click labels only from the packed train arrays. Requests without a
  clicked candidate remain excluded for the positive-listwise objective.
- Tokenize the aligned raw query text with the locked BGE tokenizer at maximum
  length 32. No retrieval instruction prefix is added, matching the official
  KuaiSearch embedding artifact.
- Reuse frozen CLS item-title embeddings from
  `BAAI/bge-small-zh-v1.5`. Dev scoring never reads qrels.
- Test remains untouched.

## 3. Models

### D2t - fine-tuned text ranker

Initialize the four-layer `BAAI/bge-small-zh-v1.5` query encoder from its public
checkpoint and fine-tune all query-tower parameters. Candidate titles use the
frozen official CLS embeddings followed by one trainable, identity-initialized
linear adapter. L2-normalized query and candidate vectors are compared by dot
product with a bounded learned logit scale. D2t uses no user ID, history,
candidate ID embedding, click-count feature, or dev label.

### D2p - non-personalized text/popularity mixture

After D2t calibration, select one global alpha on the train-only validation
split from `{0.0, 0.1, ..., 1.0}`:

`score = alpha * z(D2t) + (1 - alpha) * z(train-only log-click count)`.

Ties select the largest alpha, preferring more text and less popularity. The
same alpha is frozen for all final seeds. D2p remains non-personalized and
contains no history. It tests whether a stronger query model plus a legal item
prior can explain B7 without behavioral evidence.

## 4. Train-only calibration

Use the same first-90%/last-10% retained-train split as D1. Seed 20260708 may
train for at most four epochs with patience one and `min_delta=0.0001`, selected
by internal NDCG@10. Optimizer and scheduler are fixed in the hashed config.
One retry is allowed only for a non-finite loss/gradient or inability to move
any trainable parameter; it cannot be triggered by a low dev result.

The selected epoch and D2p alpha are written to a separate final config before
any D2 dev scoring. Final D2t models retrain on all retained train requests with
seeds 20260708, 20260709, and 20260710.

## 5. Evaluation

Six fixed dev evaluations are authorized: D2t and D2p for three seeds. Use the
shared evaluator, frozen candidate hash, and seed 20260708 for paired bootstrap
against B2z, D1q, B0b, and B7. Report every seed direction and mean/variation.
No dev-selected retry or architecture change is permitted.

## 6. Locked interpretation

- If D2p reaches or exceeds B7, withdraw the claim that the static waterline
  requires behavioral evidence and promote D2p to the non-personalized
  baseline-to-beat. Identity-specific C3-R evidence remains separate.
- If D2t/D2p improve D1q but remain significantly below B7, the bounded result
  becomes stronger: a fine-tuned non-personalized text ranker and legal item
  prior still do not recover the static query/history combination.
- If D2 does not improve D1q, report the optimization result as negative and do
  not use it to claim query saturation.
- No outcome establishes that all query-only methods fail. The allowed claim is
  limited to the tested frozen, fine-tuned dual-encoder, and zero-shot
  cross-encoder families under the fixed candidate protocol.

## 7. Boundary

D2 is a motivation control, not the proposed system. It cannot use history and
does not authorize test evaluation. Proposed-system design begins only after
the D2 result is registered and the introduction/design documents are updated
under the corresponding locked branch.
