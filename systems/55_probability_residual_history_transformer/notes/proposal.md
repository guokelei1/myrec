# C55 proposal — probability-residual history signal

Status: pre-outcome fit-internal signal gate.  This is not a C54 rescue or an
architecture novelty claim.

## Why

C54 removed the direct history-free value path, yet its Kuai correction still
correlated 0.85--0.87 with D2p.  Ordinary final-score listwise training can
therefore spend capacity reproducing relevance already captured by the base.
Amazon activity was additionally inflated by incompatible score units.

## Frozen target

For every request, standardize the frozen base logits over valid candidates:

```text
b_i = (s_i - mean(s)) / std(s)
p_i = softmax(b)_i
y_i = click_i / sum_k click_k
t_i = y_i - p_i.
```

`t` is the exact zero-sum probability error of the base.  The C54
history-carrier Transformer is trained by per-request masked MSE to predict
`t`, not by final-score listwise loss.  At evaluation its predicted residual
is added once to standardized base logits; no coefficient is tuned.

## Controls and isolation

Each seed trains two models from paired initialization and identical request
order:

- `history_carrier`: C54 factual-minus-null history values with candidate
  competition;
- `raw_candidate`: identical capacity and optimizer, but its value path reads
  only candidate/base states and is exactly invariant to history content.

The history model is also scored with frozen matched wrong-user histories.
The existing C53 fit role is hash-split without labels into train and 1,200
holdout requests per domain.  Kuai wrong donors are inherited from the frozen
C34 fit mapping; Amazon donors are inherited from C38.  C53-A and every later
role remain closed.

## Gate

In both domains, the ensemble must reduce residual MSE over zero, raw, and
wrong controls by the frozen relative margins; improve NDCG@10 over base/raw/
wrong with positive bootstrap intervals; and preserve the same direction in
every seed.  Any domain failure closes probability-residual training on this
representation family.  There is no epoch/loss/scale/seed/domain rescue.
