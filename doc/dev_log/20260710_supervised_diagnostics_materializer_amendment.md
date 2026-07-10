# 2026-07-10 - Supervised Diagnostic Materializer Amendment

The first compact-data materialization stopped before training because some
`records_train` requests have no clicked candidate. The locked objective is a
multi-positive listwise softmax, which has no positive log-partition for those
requests.

Decision before any calibration or dev evaluation:

- exclude no-click requests from the train optimization/calibration arrays;
- do not alter candidate rows for retained requests;
- report the excluded request count in the materialization manifest;
- leave the 12,229-request dev evaluation scope unchanged.

This is an implementation-domain correction, not a result-dependent protocol
change. The original materialization exited with an assertion; no checkpoint,
score file, metric, or dev-log entry was created.
