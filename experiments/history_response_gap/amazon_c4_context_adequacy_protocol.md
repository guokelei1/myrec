# Amazon-C4 context-adequacy check

Frozen on 2026-07-14 after the two-event result and before any eight-event model
outcome.

## Question

Did the initial history budget of two make Amazon-C4 look directionally easy by
omitting the longer, potentially conflicting preference history?

## Single permitted check

Train one matched QC/FULL v2-m3 pair with the same pair examples, seed,
objective, optimizer and update budget as the completed Amazon run. Change only
the context capacity to maximum length 1024 and the FULL history budget to the
eight most recent events. Label-free real-tokenizer audits already establish
that the complete query-plus-history first sequence fits every train and dev
request; `only_second` may truncate only candidate metadata.

Score FULL under true, null and the already frozen wrong-history assignments.
Evaluate all and strict-nonrepeat requests using the same direction, utility,
QC comparison and fixed-delta intervention contract. Do not add further history
budgets after seeing this result.

## Interpretation

- If candidate-relative direction remains clearly above 0.65 and true history
  retains positive utility over null/wrong, the absence of the KuaiSearch gap is
  not explained by the two-event budget.
- If direction falls to approximately chance while response remains active and
  utility becomes unstable, the original Amazon boundary was input-limited and
  must be revised.
- FULL versus QC decides net system benefit separately. A large true-null gain
  that only repairs FULL-null degradation is not evidence that personalization
  beats a strong non-personalized ranker.

This check cannot resolve dataset scale because the available Amazon-C4/history
release contains only 14,983 aligned requests. Scale is assessed through power,
baseline adequacy and confidence intervals, and—if a different Amazon test is
later needed—through the conventional Amazon PPS construction as a separately
defined dataset, not by silently changing this one.
