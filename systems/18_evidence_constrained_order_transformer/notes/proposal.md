# C18 proposal

## Observation

The closed portfolio exposes two different failures that ordinary history
residuals conflate:

1. exact candidate recurrence is the only stable history component in C5-R3;
2. candidate-conditioned attention/FFN/value/flow writes repeatedly became
   candidate-common, vanishing, saturated, or corruption-insensitive.

The architecture consequence is not another value function.  Reliable evidence
must constrain the **admissible final ordering**, while speculative semantic
transfer may move scores only inside that admissible set.

## Single primitive

ECOT's primitive is an evidence-conditioned Euclidean projection of a
Transformer score proposal onto a recurrence-anchored order cone.

Let `b_i` be the query/candidate Transformer logit, `r_i` indicate that
candidate `i` occurs exactly in strictly prior history, and

```text
a_i = b_i + beta r_i,                   beta > 0
y_i = a_i + rho * Center_i(tanh(u)),    u = HistoryTransformer(q,H,C).
```

The synthetic gate freezes `beta` before outcome; any later real gate must
separately lock how a nonnegative in-model coefficient is learned.  The semantic
history path receives event content, order and behavior but not
the exact item-identity equality bit.  Identity evidence acts only through the
constraint set

```text
K(a,r) = {s :
  mean(s)=mean(a),
  s_i-s_k >= a_i-a_k for every r_i=1, r_k=0 }.

s* = argmin_{s in K(a,r)} 0.5 ||s-y||_2^2.
```

When history is absent, the entire personalized path is skipped and `s*=b`
bitwise.  When no exact recurrence is present, `K` has no order inequalities
and the bounded semantic proposal remains unchanged.  Thus projection protects
repeat evidence without routing request types to separate scorers.

## Why this attacks the observed failure

- common score translations are removed before projection;
- the semantic write has a nonzero bounded radius and must pass a label-free
  order-change lower bound;
- exact recurrence is a final-margin invariant rather than an internal feature
  that later normalization can dilute;
- the projection is the minimum score change satisfying the evidence contract,
  so unsupported transfer cannot move repeat comparisons more than necessary;
- corruption sensitivity remains a learned-property gate for non-repeat
  transfer and cannot be claimed from the hard invariant.

## Falsification

C18 fails before real data unless a learned three-seed synthetic probe shows
all of the following simultaneously: exact-repeat non-degradation, useful
supported non-repeat transfer, hard-corruption collapse, nontrivial order
changes, exact no-history fallback, and advantage in worst-subset utility over
the same Transformer trained with direct residual or soft constraint penalty.

Passing the synthetic gate would establish only operator viability under the
frozen construction.  It would not establish KuaiSearch signal, dev gain, or
global novelty.
