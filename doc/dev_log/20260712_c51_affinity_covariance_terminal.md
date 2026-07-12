# 2026-07-12 — C51 affinity covariance terminal

C51 tested a dataset-independent observable relation rather than another
prediction-error value.  For each candidate it centered, across the user's
history events, query-to-event and candidate-to-event affinities and used their
covariance as the correction.  This removes user-level common semantic
affinity and asks whether query and candidate select the same events.

All pre-outcome structural and numerical checks passed.  On exposed KuaiSearch
the operator improved over query base by `+0.002675` with a positive interval,
but it was worse than both the uncentered moment and plain KRR; clicked
direction and specificity were unstable.  On exposed Amazon-C4 it improved
over base by only `+0.001551` with a zero-crossing interval and was about
`0.022` below posterior-supported ridge and fixed softmax.  True history did
not beat wrong history stably.

The two-domain disagreement rules out a scale or normalization rescue.  More
importantly, C47--C51 now jointly show that candidate support, sign consensus,
prequential error, orthogonal dual memory, and centered affinity covariance do
not turn frozen pooled semantic states into reliable evidence.  The search
therefore leaves this abstraction level.  The next candidate must alter token
representation formation and make candidate competition load-bearing before
the final score, while using the failed pooled operators as nearest controls.
