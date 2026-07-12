# C43 cross-dataset compatibility audit

## Verdict

Exact C42 checkpoint transfer is invalid because Amazon uses 384-dimensional
`BAAI/bge-small-en-v1.5` states while the frozen KuaiSearch protocol uses
512-dimensional `BAAI/bge-small-zh-v1.5` states. Adding a learned or fixed
384/512 bridge would introduce a new untested primitive and confound transfer.

C43 therefore freezes the architecture and training rule, reconstructs only
the dimension-dependent tensors at width 512, and trains all paired modes on a
label-isolated KuaiSearch fit split. This is the smallest valid cross-domain
test.

## Frozen versus mechanical changes

Frozen:

- metric-coupled head equation and all matched reductions;
- four heads, rank 16, temperature 0.1, profile scale 1, correction scale 2;
- initialization std 0.01, AdamW, learning rate 0.001, weight decay 0.0001;
- one epoch, batch size 16, full-candidate listwise plus direction loss;
- 6,000 fit requests, three seeds, and conjunctive seed/fold/CI gates.

Mechanical/domain-protocol changes:

- hidden width 384 to 512 because the registered frozen LM differs;
- Amazon frozen-BGE base to the already frozen KuaiSearch D2p base;
- exact recurrence returns the registered KuaiSearch item-only fallback;
- C43 uses the KuaiSearch packed train interface and train-internal labels only.

No dataset branch occurs inside the model. Evidence absence is expressed only
through query/history/repeat masks.

## Outcome isolation

C43 fit is a label-free deterministic 6,000-request subset of C37 fit. C43-A
is exactly all 600 C37 delayed-B plus all 600 C37 escrow requests. C37 reports
record both roles as feature/score/label unopened. Their labels may open only
after all C43 modes score and A0 passes. Dev/test and qrels never open.
