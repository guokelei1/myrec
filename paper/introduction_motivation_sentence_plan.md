# Introduction Front-Half Sentence Plan

Date: 2026-07-10

Status: **motivation complete; architecture/protocol formulation is authorized,
while implementation and training remain gated by a new design-specific
pre-outcome falsifier**.

## Writing Goal

The front half should establish this chain:

```text
fixed candidate pools are already query-conditioned
  -> supervised text + train-only popularity yields a strong D2p base
  -> full D2s appears to benefit from correct rolling history
  -> temporal identity specificity is not stable on the same-query control
  -> exact pre-outcome decomposition separates item recurrence and category affinity
  -> item-only is strongly positive and becomes the 0.3454 static waterline
  -> category-only is indistinguishable from D2p
  -> adding category affinity significantly weakens item-only in every seed
  -> tested history evidence has unequal fidelity: exact recurrence is reliable,
     while uncalibrated cross-item/category transfer is not
  -> C5-R3 TERMINAL_FAIL closes only the doc/23 item/category recovery ladder
     and validates neither of its two candidate primitives
  -> motivation closes with a bounded design problem and permits
     architecture/protocol formulation, not implementation or training
```

## Global Guardrails

- Say "in the tested B0b/D2s bundle," not "all personalization."
- Say "exact repeat-item recurrence," not generic semantic user preference.
- Treat C5-R2 as a failed identity gate and C5-R3 TERMINAL_FAIL as terminal
  only for the preregistered doc/23 item/category recovery ladder.
- Do not use M3/M4 oracle values as headroom or router evidence.
- Do not claim category semantics, query attention, or identity specificity.
- Report trainable controls by three-seed mean, never best seed.
- Use item-only D2s (mean 0.3453755), not full D2s/D2h/B7, as the current
  static baseline-to-beat.
- Require exact D2p fallback when history is absent.
- State that test is untouched and unavailable for motivation selection.
- A bounded transition into architecture/protocol formulation is allowed, but
  do not claim that C5-R3 validated either candidate primitive or a new system.
- Require a new design-specific pre-outcome falsifier before implementation or
  training begins.

## Paragraph 1: Task

1. Define fixed-candidate personalized product ranking under explicit query
   intent and strictly-prior history.
2. Explain that recall has already query-filtered the candidate set.
3. State the central audit question: does a history gain reflect semantic
   preference use or a narrower recurrence shortcut?

## Paragraph 2: Protocol Discipline

1. One candidate manifest, one evaluator, and identical tie-breaking apply to
   every method.
2. Raw `recently_*` fields are rejected; histories are rebuilt from events
   strictly before the request.
3. Scoring cannot read dev/test qrels; test remains untouched.
4. Every claim control is frozen before its outcome.

## Observation 1: Strong Non-personalized Base

1. Query-shuffle and catalog-reservoir checks establish query-conditioned
   candidates.
2. D2t improves zero-shot BGE.
3. D2p combines D2t and train-only popularity for a 0.3239501 mean.
4. Bound the claim to tested lexical/semantic/non-personalized controls.

## Observation 2: Bundled History Gain Is Ambiguous

1. Full D2s reaches 0.3416290 and beats D2p.
2. The historical train-frozen wrong-user control is temporally confounded.
3. C5-R2 preserves aggregate value but fails same-query significance (1/3 vs
   the frozen 2/3 requirement).
4. Therefore do not call the bundled gain established identity-specific
   personalization.

## Observation 3: C5-R3 Explains the Gain

1. State that item/category decomposition and the sole fallback were locked
   before component outcomes.
2. Report exact decomposition: 575,609 rows, max error `7.1e-15`, zero
   tolerance violations.
3. Item-only vs D2p on history-present requests:
   +0.03204/+0.03214/+0.03263, all CIs positive.
4. Category-only vs D2p:
   +0.00059/+0.00053/-0.00003, all CIs cross zero.
5. Full minus item-only:
   -0.00538/-0.00521/-0.00634, all intervals negative.
6. Item-only overall mean is 0.3453755 and becomes the static waterline.
7. All 4,110 no-history requests are rank/metric equivalent to D2p.

## Observation 4: Negative Mechanism Evidence

1. D1 mean and query-attentive residuals do not stably beat their base.
2. Representative sequence/CTR/PPS neighbors do not exceed the current static
   result; preserve their documented alignment caveats.
3. Random-channel canaries invalidate M3/M4 oracle/router motivation.
4. These negatives are consistent with, but do not prove, the recurrence
   shortcut interpretation.

## Closing Paragraph

1. State the supported insight: apparent history value can be dominated by
   exact repeated items rather than semantic preference transfer.
2. State the benchmark implication: future methods must beat item-only, not
   merely D2p or full D2s.
3. State the scientific boundary: the C5-R3 primary and fallback failed, so
   neither candidate primitive in doc/23 is validated and that recovery ladder
   is closed.
4. State the bounded design transition: formulate an architecture and protocol
   that preserve reliable exact recurrence while calibrating or rejecting
   unsupported cross-item/category transfer.
5. State the execution boundary: a new design-specific pre-outcome falsifier
   must be frozen and passed before implementation or training.

## Forbidden Sentences

- "We prove that personalization is useful."
- "User identity causes the gain."
- "Category preference is an effective semantic memory."
- "Query-conditioned attention is established by the motivation."
- "D2s is the strongest static baseline."
- "C5-R3 proves that our proposed architecture works."
- "Both C5-R3 candidate primitives are validated."
- "Test confirms the insight."
