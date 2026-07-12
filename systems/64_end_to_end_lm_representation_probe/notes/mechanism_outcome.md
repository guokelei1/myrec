# C64 representation-probe outcome

Status: **failed terminal at label-free A0**.  The 1,200 validation labels,
all fresh roles, dev, test, and qrels remained closed.

Label-free G0 passed completely.  The deterministic 4,800/1,200 split and
candidate hashes were valid; only BGE layers 2--3 were trainable; gradients
reached both adaptive layers, the joint Transformer, and the output head;
true/wrong history changed internal candidate states; candidate permutation
error was `2.98e-7`; and no-history/repeat error was zero.

Three GPUs then trained all three fixed modes for 600 steps each.  Every loss
decreased and every gradient/finite check passed.  End-to-end adaptation was
substantially rank-active: primary changed `40.25%/55.17%/46.08%` of complete
orders and `2.33%/3.83%/5.17%` of Top-10 sets versus the strong base.

The history-specific gate failed.  Wrong-history replacement changed
`27.42%/18.50%/37.50%` of complete orders, but only `7/5/60` of 1,200 Top-10
sets.  The frozen rule required at least 12 in every seed, so seeds 20264501 and
20264502 failed independently of any utility label.  In addition, bf16
candidate-set scoring produced `0.00392--0.00394` permutation error against a
`2e-6` tolerance.

An fp32 or canonical-order rescore could repair only the numerical contract;
it cannot repair the two binding Top-10 failures and is therefore forbidden.
C64 closes without validation labels or Amazon continuation.  This result
shows that unfreezing pretrained token layers makes the ranker active, but does
not make correct history consistently load-bearing at the decision boundary.
