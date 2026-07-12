# C66 pre-implementation review

Decision: authorize one label-free G0 and, only if it passes, the exact C65
three-seed training protocol.

- C65 opened no labels and trained no model, so no outcome can inform C66
  optimization.
- Stable item-key sorting is label-free and preserves candidate sets exactly.
- The wrapper adds zero parameters and must restore caller order for all output
  tensors.
- Every scientific configuration value is copied verbatim from C65.
- No second numerical continuation is authorized.
