# Current direction decision

Status: active scope note, 2026-07-15.

The project remains Query-conditioned Personalized Product Ranking and
LLM4Rec. The universal direction-gap hypothesis in doc 34 was useful but did
not survive unchanged: Amazon-C4 has strongly correct history direction and
JDsearch strict-nonrepeat direction is reliably above chance.

The active motivation is the controlled-history-composition problem in doc 35:

> ordinary full-token LLM4Rec rankers can read history, but do not reliably
> preserve query-candidate base capability while adding a high-efficiency,
> candidate-relative history update. Positive true-over-null utility can repay
> a base deficit created by joint history training, while recurrence masks the
> problem overall.

## Evidence tracks

- **Main natural-language source:** KuaiSearch Full. Its strict-nonrepeat
  surface localizes the direction-allocation failure; Lite supplies
  cross-family and cross-objective exploratory evidence only.
- **Independent functional replication:** JDsearch v3. Its label-free hash
  candidate order and anonymized terms support ranking/accounting evidence,
  not a pretrained-semantic claim.
- **English semantic positive boundary:** Amazon-C4 plus Reviews-2023 history.
  Its constructed target-revealing query falsifies universal direction failure
  while exposing the base/history tradeoff under a longer history budget.

These are exploratory development populations, not independent confirmation.
KuaiSAR is no longer required for the completed motivation decision; it may be
considered later only under a new pre-outcome confirmation protocol.

## Authorization

The next authorized step is a frozen Failure Card testing standard repairs,
train-only direction recoverability, independent-family replication, and
confirmation. No C81, C80 rescue, old R0 round, proposed architecture source,
or architecture GPU training is authorized by this decision.
