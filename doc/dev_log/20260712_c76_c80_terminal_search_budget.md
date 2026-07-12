# C76--C80 terminal architecture-search budget

Date: 2026-07-12
Status: binding user instruction

The architecture search now has a hard terminal boundary.

1. Complete and judge the currently frozen C76 primitive.
2. If C76 fails, permit at most three further **mechanism-level** architecture
   updates.  No proposed candidate may be numbered beyond C80.
3. Mechanical repairs, canonicalization, threshold changes, hyperparameter
   changes, renamed copies, or same-primitive rescues do not create additional
   scientific budget and may not be used to evade the boundary.
4. Stop early if a candidate satisfies the registered architecture, utility,
   specificity, safety, matched-control, and cross-domain requirements.
5. If C80 is reached without such a candidate, stop architecture search.  Do
   not issue C81.  Produce a repository-level C01--C80 retrospective answering
   why repeated iteration did not yield a CCF-A-level motivation-to-design
   contribution.

The terminal retrospective must allocate causal responsibility rather than
list failures.  At minimum it must assess:

- whether the motivation established a constructive and sufficiently general
  target, or mostly negative constraints;
- whether design principles overconstrained useful models or rewarded formal
  non-reduction over predictive adequacy;
- whether early pooling/data interfaces removed the signal before architecture
  search;
- whether synthetic/mechanical gates had construct validity for real ranking;
- whether repeated single-candidate local pivots caused specification and
  benchmark overfitting;
- whether label-role fragmentation, tiny fresh cohorts, and test locks left
  enough statistical power for learning and selection;
- whether the available datasets support one common architecture claim;
- which conclusions are robust enough to retain for a paper even if the
  proposed architecture is negative.

This boundary refines the existing active goal; it does not declare that goal
complete.  It is also reflected in the live execution plan.
