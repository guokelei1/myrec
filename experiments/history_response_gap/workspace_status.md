# Workspace readiness — 2026-07-15

## Ready now

- dataset-independent standardized-record and label-isolation validation;
- shared candidate-hash evaluator and paired bootstrap comparison;
- true/null/wrong counterfactual identity checks;
- candidate-relative response, signed direction, clustered uncertainty, and
  fixed-response attribution diagnostics with hand-computed unit tests;
- label-free request surfaces and causal history assignments for KuaiSearch,
  Amazon-C4, and JDsearch;
- KuaiSearch Lite cross-family/cross-objective exploration;
- an independent KuaiSearch Full-source population scout and a controlled
  prior-query restoration probe;
- Amazon-C4 two-event and eight-event history-budget probes;
- admitted JDsearch v3 data with label-free candidate order, matched QC/FULL
  training, true/null/wrong scoring, and a QC/FULL by serialization factorial;
- the cross-dataset controlled-history-composition motivation in
  `doc/35_controlled_history_composition_motivation.md`;
- preserved legacy source and outputs for read-only, selective reference.
- a frozen representative matrix (Qwen, HSTU, LLM-SRec, plus BGE anchor), the
  Apache-2.0 HSTU source snapshot, and a shared label-free sequence adapter that
  passes all current standardized-source audits.
- complete Lite exploratory HSTU/SASRec QC/FULL bundles and a complete
  paper-mechanism LLM-SRec true/null/wrong bundle with a frozen train-only
  SASRec teacher.

## Current scientific decision

Exploratory motivation is established. The universal claim that Transformers
cannot direct history is rejected. The surviving problem is that ordinary
full-token rankers do not jointly control base retention and candidate-relative
history utility: KuaiSearch exposes low direction conversion, JDsearch exposes
low nonrepeat conversion plus training-induced base erosion, and Amazon-C4 shows
that even strongly correct history direction can be outweighed by larger base
erosion.

## Still blocking architecture and binding confirmation

- no separately frozen whole-population confirmation split or endpoint/MDE lock;
- no adequate independent-family replication of the full base/history
  accounting on JDsearch or Amazon-C4;
- no successful train-only recoverability witness for strict-nonrepeat direction;
- no standard repair control showing whether anchoring, history dropout, or an
  ordinary objective can close the joint tradeoff;
- no localization to a specific Transformer computation rather than objective,
  optimization, or interface.
- HSTU/SASRec adapters and runs exist, but their Lite QC remains below BM25 and
  the HSTU repeat positive control is not yet binding; LLM-SRec needs a bounded
  normal Amazon-C4 run and an explicit adequacy/QC boundary before its negative
  outcome can support a shared-failure claim.

## Readiness decision

The repository is ready to freeze the next Failure Card and its cheapest
discriminating controls. It is not ready for a proposed architecture source tree
or architecture GPU training. No wholesale archive restoration is needed; use
`archive_reuse_policy.md` only when a named old utility is required.
