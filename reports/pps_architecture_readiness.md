# PPS Architecture Readiness

Date: 2026-07-10

Status: **ready for design formulation; not yet ready for full
implementation/training**.

The motivation stage is complete. C5-R3 correctly falsified its preregistered
multi-granular and coarse-category claims, but that failure was previously
overextended into a prohibition on any design work. The valid interpretation is
narrower: C5-R3 establishes that history evidence has unequal empirical
fidelity. Exact candidate recurrence is reliable; the tested category transfer
is not, and mixing it with recurrence reduces ranking quality. This contrast is
a design problem even though it is not yet a validated architecture.

## Current Bounded Insight

```text
Observation: history evidence has unequal empirical fidelity. Exact candidate
             recurrence is stable, while tested coarse category transfer is
             non-informative and dilutes the stronger item-memory ranking.

Design hypothesis: one candidate-conditioned evidence-fidelity calibration
             primitive should preserve reliable recurrence and admit a
             transferable personalized residual only when joint
             query/history/candidate evidence supports it.

Falsification: before full implementation, the design must preserve item-only
             behavior on repeat-present requests, add stable value over D2p on
             the 4,677 non-repeat history-present requests, reject
             coarse/wrong/shuffled/query-masked evidence, and fall back exactly
             to D2p without history.
```

The observation is established. The primitive is a formulation target, not an
empirical claim. It must not be presented as “category memory works,” “query
attention works,” or “semantic transfer is already proven.”

## Empirical Basis

| Evidence | Result | Design implication |
|---|---:|---|
| Item-only vs D2p, history-present | +0.03204 / +0.03214 / +0.03263; 3/3 significant | reliable recurrence must be preserved |
| Category-only vs D2p | +0.00059 / +0.00053 / -0.00003; 0/3 significant | coarse transfer cannot be trusted by default |
| Full D2s minus item-only | -0.00538 / -0.00521 / -0.00634; 3/3 significantly negative | undifferentiated mixing dilutes strong evidence |
| Non-repeat history-present surface | 4,677 requests | remaining surface for transferable personalization |
| No-history surface | 4,110 requests | exact D2p fallback contract |
| D1 query-attentive residual | no stable gain | generic attention is not sufficient evidence |

The item-only control remains the binding static waterline at mean NDCG@10
**0.3453755427**. If a later protocol retains a 2% full-claim margin, the
reference is approximately **0.3522831**.

## What the Motivation Establishes About Existing Systems

Within this dataset, fixed-candidate protocol, and tested method set, the
existing systems are insufficient for the target problem. Official RecBole
SASRec averages 0.2972; proxy-aligned KuaiSearch DNN/DCNv2 average
0.3063/0.3054; provisional ZAM/TEM adapters average 0.2986/0.2940 with their
documented provenance and cold-product caveats; and the D1 query-attentive
residual does not stably improve its supervised base. None exceeds the 0.3454
item-only control, and full D2s is significantly worse than item-only on the
frozen history-present comparison in all three seeds.

The bounded conclusion is therefore not “history is useless.” It is that the
tested systems do not reliably extract transferable history value beyond a
narrow recurrence shortcut, and some undifferentiated history use dilutes that
shortcut. This is sufficient motivation for a new design problem, but not a
universal claim about every personalized ranker or proof that the proposed
primitive will succeed.

## Stage Authorization

| Stage | State | Authorized work |
|---|---|---|
| Motivation | complete | preserve the audited evidence and claim boundaries |
| Design formulation | **ready now** | architecture proposal, one primitive, information flow, nearest-neighbor comparison, control matrix, efficiency budget, execution protocol |
| Pre-implementation design gate | pending | freeze and execute the repeat/non-repeat evidence-fidelity falsifier |
| Full implementation/training | blocked until gate passes | model training, dev tuning, multi-seed expansion |
| Test/finals | locked | one run only after the complete system/config is frozen |

## Required Design Deliverables

Design formulation may begin immediately, but it must produce all of the
following before any full training:

1. a mathematical definition of one evidence-fidelity calibration primitive,
   not a router over fixed scores;
2. a proof-by-information-flow that it is not reducible to DIN target attention,
   ZAM/TEM query attention, generic history retrieval, or a fixed mixture;
3. a parameter/compute-matched backbone control;
4. a frozen cheap gate covering repeat-present, non-repeat history-present,
   wrong-user, event-shuffle, query-mask, coarse-only, and no-history cases;
5. a unified mask-based interface with no dataset-specific architecture branch;
6. at most three named components, each with a rent-paying ablation;
7. an efficiency budget with zero online LLM calls.

## Claim Boundary

Design formulation is justified because the experiments expose a concrete
failure mode in tested systems and heuristics. It does **not** mean that:

- semantic preference transfer already exists in this benchmark;
- user identity causally explains the aggregate gain;
- a query-aware Transformer will necessarily work;
- exact recurrence itself is the paper's architecture primitive;
- C5-R3's failed primary or fallback has been reinterpreted as passed.

Authoritative numerical result:
`reports/pps_c5r3_candidate_history_alignment.json`. Current stage decision:
`reports/pps_c5_insight_audit.json`. Test remains untouched.
