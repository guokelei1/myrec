# C50 proposal — semantic-protected dual memory

C49 failed because replacing raw semantic KRR values with prequential
innovations removed a strong semantic path and produced the wrong specificity
on Amazon.  C50 tests a different information-flow law: raw semantic memory is
retained exactly along its read direction, while behavioral innovation may
change only the orthogonal complement.

For raw KRR read `u_s` and innovation read `u_b`,

`u_b_perp = u_b - u_s <u_s,u_b> / (||u_s||^2 + eps)`,

`u = u_s + u_b_perp`.

This is not a scale rescue: the unprojected `u_s+u_b`, raw `u_s`, innovation
alone, and C47's best fixed score are binding controls.  The current test
reuses the frozen C49 checkpoints and exposed C47-A only.  C50 must beat raw,
unprojected, and C47-best with positive intervals plus every seed/fold sign on
both domains, while retaining true/wrong and clicked specificity.  Otherwise
it closes before training or fresh reserve.
