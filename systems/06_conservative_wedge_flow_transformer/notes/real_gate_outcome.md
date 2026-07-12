# C06 real-gate outcome

Status: **terminal A0 failure; C06 is closed**.

This note records the locked train-internal mechanism gate only. It is not a
dev/test result or a positive paper claim. No threshold, model setting, data
selection, seed, optimizer, batch rule, or epoch count was changed after the
review1 lock.

## Repair1 fit completion

The three variants eligible for the pre-A numeric repair completed their one
and only repair attempt on 12,000 fit requests for the frozen two epochs:

| Variant | Run ID | Epoch fallback rows | Total fallback rows | Report SHA256 | Checkpoint SHA256 |
|---|---|---:|---:|---|---|
| local Hodge | `20260711_kuaisearch_c06_local_hodge_repair1_s20260708` | `198`, `16` | `214` | `c68a4d509ec58822c4b1b70ab326b90e505b1aeea24becd3940aaf1b6ec12627` | `8db9a667672c76dca94f7ab50ff56b79b931e43168c421906efa876fe100f489` |
| untrusted `t=1` | `20260711_kuaisearch_c06_untrusted_repair1_s20260708` | `31`, `506` | `537` | `7193d2d804cf952e984bb11e6348fb2af2d4f1ae0ef58eb5dcbc017e7aef86e7` | `230efa268c8007fab9cf4d52ba42b38b7aed34df1169b30b05fc62affbc61665` |
| direct learned gate | `20260711_kuaisearch_c06_direct_gate_repair1_s20260708` | `785`, `138` | `923` | `293a30882728d0aef7552ea201e2e67036e0cb61100f520271af41284252857d` | `ea8d256fed2969efd65e17accd7484834a74464010068d46f0485a8b320863fd` |

The corresponding raw report paths are
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/training/local_hodge.json`,
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/training/untrusted.json`,
and
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/training/direct_learned.json`
in table order. Their checkpoint paths are
`models/c06_conservative_wedge_flow_transformer/real_gate_v1/local_hodge.pt`,
`models/c06_conservative_wedge_flow_transformer/real_gate_v1/untrusted.pt`, and
`models/c06_conservative_wedge_flow_transformer/real_gate_v1/direct_learned.pt`;
the table gives each exact report/checkpoint hash.

Every repair ledger completed while recording
`internal_A_features_scored=false`, `internal_A_labels_opened=false`, and no
delayed-B/escrow access. Their raw ledgers are:

- `artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/formal_attempt_repair1_local_hodge.json` — `72ee21fded6e3f6465ad8c87351ce01defab9b6ac090903f2dfe9fd7319cdbb4`;
- `artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/formal_attempt_repair1_untrusted.json` — `f3087c9354f95d3abc480e2efd1f0d8985e8877298c8e4923e375dd1affa04a6`;
- `artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/formal_attempt_repair1_direct_learned.json` — `757784cf0ae1e6838cfbcac605ba471eaec26fce354559f1884942cb15c65c40`.

The completed centered v1 control was preserved rather than rerun. Its report
remains
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/training/centered_cross_attention.json`
at `b464a99a9d679c64d8388e43a3fe801dcbbd9f798a936c852415f5e29c5252e2`.

## A0 label-free terminal result

A0 scored the frozen 1,200-request internal-A candidate sets without labels.
All methods rescored bitwise deterministically; scores were finite; local trust
was finite in `[0,1]`; no-history scores were bitwise base; candidate-common
factors were exact zero; conservation, score bounds, and pool-intervention
numeric contracts passed. All first-rescore fallback counts were zero.

The local correction was nevertheless too small to affect ranking:

- `delta_range_fraction_of_bound = 0.0`;
- requests with any local-vs-base order change: `0 / 1200 = 0.0`;
- requests with top-10 membership change: `0 / 1200 = 0.0`;
- maximum absolute conservative delta:
  `0.000018312828615307808`;
- maximum absolute candidate-sum delta:
  `2.7284841053187847e-12`;
- maximum common-mode ratio: `3.87183344938117e-8`.

The same-checkpoint controls showed that the trust operators changed numeric
deltas but still did not change an order:

- local-to-`t=1` requests with different deltas:
  `0.9991666666666666`; order-change fraction `0.0`;
- local-to-global-Hodge requests with different deltas:
  `0.9991666666666666`; order-change fraction `0.0`.

Exactly four preregistered A0 checks failed:

| Failed check | Observed | Required |
|---|---:|---:|
| `enough_nontrivial_delta_ranges` | `0.0` | at least `0.10` of requests with range above `0.001` of the bound |
| `enough_order_changes` | `0.0` | at least `0.05` |
| `enough_top10_changes` | `0.0` | at least `0.01` |
| `t_one_changes_orders` | `0.0` | at least `0.01` |

The other A0 checks, including `global_changes_deltas` and
`t_one_changes_deltas`, passed. The exact A0 artifact is
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/a0_label_free_audit.json`
with SHA256
`b989bdcd5166b13dee45fb30a7c150fc20de7520a9586e2e3e489ee89145f9ff`.

## Evidence boundary and decision

The terminal report status is `failed_A0` and its locked decision is:
`stop before internal-A labels; close the C06 primitive`.

- Internal-A features were scored only for the label-free A0 audit.
- Internal-A labels were not opened.
- `a1` is `null`.
- Delayed B and escrow were not opened and will not be run.
- Qrels, dev records, test records, dev evaluator calls, and test access were
  all zero/false.
- No failed threshold will be relaxed and no further C06 tuning or rescue run
  is permitted.

The terminal raw report is
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/real_gate_report.json`
with SHA256
`4c9765c189df009c40cd91509b66ebd0bec81d08ff92b7b5ad400116054660a3`.
Its audit ledger is
`artifacts/c06_conservative_wedge_flow_transformer/real_gate_v1/formal_attempt_audit.json`
with SHA256
`06bc9f4cac8138b737fd013e4abc78a002b73c2af247a9684cce959736a7a83f`.
The immutable G0 and review1-lock hashes are respectively
`7430dae9c56b257cb64a9c75e3e0cbf932856d9419e296b64fa1e9cc81a0af1e`
and `565d3133ed5b098e3b8722e6a031ca5d979365857dd74b2ce1aecdf37b178f14`.

## Conclusion

The repair established that C06 can train stably under its exact numeric
contracts. It did not establish a useful architecture: candidate-local Hodge
trust changed very small score deltas but produced no candidate-order change on
the entire frozen A cohort. The preregistered label-free gate therefore closes
C06 before labels, A1, delayed B, escrow, dev, or test. Future architecture
work must use a separately proposed mechanism rather than tuning this gate.
