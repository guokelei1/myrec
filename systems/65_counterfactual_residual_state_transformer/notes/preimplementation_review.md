# C65 pre-implementation review

Decision: authorize implementation, label-free G0, and a three-seed exposed-fit
gate only after source/data lock.

- The primitive responds directly to C64's generic-reranker shortcut and does
  not increase LM layers, width, epochs, or candidates.
- The end-to-end LM remains the ranking core; the strong base is an anchor
  coordinate, not a routed mixture.
- Hidden residual, wrong-neutral objective, factual-state, and logit-difference
  contributions are separated by binding controls.
- C64's bf16 permutation issue is handled prospectively by frozen fp32 full-
  candidate scoring.  This is a numerical contract, not an outcome rescue.
- Reusing the C64 split is permitted because validation labels never opened.
  C64 score activity is known, so C65 results remain a formulation gate rather
  than fresh paper evidence.
- No validation/fresh/dev/test label may open until C65 A0 passes every seed.
