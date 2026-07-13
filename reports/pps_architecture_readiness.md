# PPS Architecture Readiness

Date: 2026-07-13

Status: **NOT READY FOR ARCHITECTURE FORMULATION OR TRAINING. R0 problem
discovery is authorized.**

Current authority:
`doc/31_problem_discovery_and_architecture_iteration_protocol.md`; continuous
execution and whole-pipeline end states follow
`doc/32_autonomous_pipeline_controller.md`.

## Terminal Boundary

- C01--C80 architecture search is closed.
- C80 failed its frozen pre-label event-permutation mechanical contract.
- C80's 365 fresh labels remain unopened; C80 utility is unknown.
- There is no C81 and no precision, canonicalization, threshold, label-opening,
  dev, or test rescue for C80.
- `doc/24_parallel_llm4rec_design_protocol.md` is historical and authorizes no
  current run.

The authoritative terminal evidence is
`reports/pps_c80_amazon_real_gate.json` and
`doc/dev_log/20260712_c01_c80_terminal_retrospective.md`.

## What Is Established

| Evidence | Supported conclusion |
|---|---|
| KuaiSearch C5-R3 | exact item recurrence is strong; coarse category transfer is not independently established; bundled history dilutes item-only |
| Amazon pooled HSO | pooled text exposes only a small strict-non-repeat direction (`+0.001661`) |
| Amazon full-token HSO | ordinary joint Transformer exposes practical true-null (`+0.025298`) and true-wrong (`+0.035944`) value |
| KuaiSearch R0 full-token | current family exposes three-seed true-null (`+0.020936`) and true-wrong (`+0.031812`) value, but is not an adequate strong base |
| Frozen edge attribution | bidirectional Q--H, C--H, and history-read-context paths are load-bearing |
| True vs shuffled history | current evidence does not establish event order as load-bearing |
| C01--C80 portfolio | activity, safety, algebraic distinctness, or corruption sensitivity alone does not establish ranking utility or unique mechanism rent |

## Missing Architecture Premise

No experiment currently establishes a reproducible defect of a normally tuned
ordinary full-token Transformer that requires a new architecture. The strongest
positive result supports repairing the representation interface first.

The previous candidate-conditioned evidence-fidelity formulation is too broad
to select a mechanism. It remains background motivation, not an architecture
authorization.

## R0 Round 1 Review

Round 1 established a real Transformer asset on KuaiSearch: three-seed
`true-null +0.020936` and `true-wrong +0.031812`. It did not establish a
strong ordinary Transformer base. The same family has true NDCG@10
`0.331028` versus item-only `0.345376`, and no-history degradation
`-0.013066` with a fully negative replication-fold interval. Both frozen
Failure Atlas ideas were falsified.

Accordingly the revised contribution level is L1 observability/measurement,
not L2 shared blind spot or L3 native Transformer shortfall. The next gate is
R0-M Motivation Brief plus R0-C0 model-family adequacy, not architecture.

## R0 Authorization

Authorized work:

1. KuaiSearch/Amazon/JDsearch information-object and holdout/power audit.
2. KuaiSearch/Amazon full-token equivalence review; shuffle remains report-only
   unless an order claim is proposed.
3. A quantitative R0-M Motivation Brief with shared-blind-spot and paper-payoff
   kill criteria.
4. R0-C0 comparison of a ranking-pretrained cross-encoder, same-backbone
   query-candidate base, current recipe, and strongest eligible baselines.
5. Within-family tuning only after base adequacy, followed by a Motivation-
   aligned Failure Atlas.
6. A replicated Failure Card that rules out simple/nearest repairs and
   localizes one ranking-relevant model failure.
7. At most two CPU/tiny-data disposable probes selected from at most three
   active failure ideas; prototypes remain under ignored `tmp/r0_prototypes/`.

The pipeline is currently user-paused; this scientific scope does not authorize
execution until explicit resume.

Not authorized:

- a new tracked `systems/<architecture>/` source tree;
- architecture GPU training;
- C80 fresh-label opening or reuse as rescue;
- a new confirmation run;
- test access.

## Architecture Entry Checklist

A future architecture hypothesis becomes ready only when all items are true:

- [x] The Round 1 sentence-encoder family received its declared dev tuning budget.
- [ ] A ranking-appropriate model family passes no-history/base-preservation and
      strongest-baseline adequacy.
- [ ] A quantitative Motivation Brief establishes prevalence, severity, shared
      blind spot, Transformer asset, and recoverable paper-level payoff.
- [ ] The failure has a ranking-utility consequence, not only internal activity.
- [ ] The failure replicated on two independent splits or a comparable second dataset.
- [ ] The effect exceeds a precomputed MDE with adequate confirmation power.
- [ ] Capacity, context length, ordinary optimization, and existing mechanisms do not repair it.
- [ ] An intervention localizes a representation, attention, objective, or ranking locus.
- [ ] A cheap falsifier and nearest simpler control are frozen.
- [ ] Dataset information objects support the intended cross-domain claim.
- [ ] A sufficiently large independent confirmation holdout exists.

Until this checklist passes, the correct output is a baseline/measurement study
or a scoped Failure Card, not a proposed architecture.

## Future Gate Order

```text
mechanics
  -> learnability and normal tuning
  -> development utility
  -> claim-specific specificity
  -> mechanism attribution and rent
  -> frozen independent confirmation
  -> Tier-2 audit
  -> one-shot test
```

Mechanical, numerical, learnability, utility, specificity, and novelty failures
must be reported separately. Event permutation is binding only for an explicit
order/set-invariance claim; wrong-user history only for provenance; no-history
exactness only for a base-preserving claim.
