# C34 pre-lock implementation review

Decision: provisionally accept for the minimal train-only gate.

The primary changes the attention/write law rather than a dataset rule.  It
removes the candidate-shared state that limited C31--C33 and fixes the sign of
admissible evidence geometrically, addressing C28's free-comparator gauge.  It
is not a query/category router, score mixture, independent candidate head, or
generic pair MLP.  Exact zero support is observable before labels.

The novelty boundary is narrow: rectified attention, target-aware history
attention, spherical projection, and tangent residuals all have prior art.
Therefore the cone law earns no claim merely by being active.  It must beat
both parameter-identical nearest reductions on a wholly fresh fit and A cohort.
A KuaiSearch pass would still require Amazon-C4/JDsearch confirmation.
