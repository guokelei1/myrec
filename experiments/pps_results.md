# Current PPS result register

Status: Current Motivation first-round (V1.2) single-seed result freeze, 2026-07-17.
These are preliminary results at the pre-registered pilot seed `20260714`, not
a multi-seed or cross-family robustness claim.

## Evidence boundary

- Protocol: `experiments/motivation/protocol.yaml`, SHA256
  `6788d27cce8186be02dae4595129157fcca5032b49c1107ec83fdd2f9ecf8e43`.
- Post-selection method/config/checkpoint lock:
  `runs/20260717_kuaisearch_motivation_v12_post_selection_release_lock/post_selection_release_lock.json`,
  SHA256 `a4ae744d78e084685a1c14f6703fe9d4f3f05805da8f5ae2ee8d423f2a3e9d3f`.
- New confirmation dataset: `full_confirm_preceding40k_newholdout4k_v12`,
  manifest SHA256
  `21f4c45b4e796a3808cb0db9de066f1e3fbf8e50a2eff8a4bdf3f1bf17d8f3bb`;
  4,000 requests and 77,836 candidate rows.
- Candidate/request/records SHA256 values are respectively
  `8b2c859bcd35400bed58b6df2cad4911e043a2c5ba2cac19c243392a3fff4c29`,
  `5586653149a4a17fd617f0beabae842170e996713539e4481eb0a08c67352db2`,
  and `828127bb611e1b7429e596a1a66977854cb9bbaf64ae443d00f1b1c32c203e8f`.
- Every result below was produced by the shared graded-NDCG@10 evaluator after
  a passing label-free score audit. Confidence intervals use 5,000
  normalized-query cluster-bootstrap draws with seed `20260715`.
- The new confirmation is a recipe-locked **retrospective** source-train
  population earlier than the reused 32k training population. It has zero
  request/session overlap and strict timestamp boundaries, but is not a
  forward-temporal or user/item/query-isolated holdout; later training
  histories can expose earlier holdout events. Source test was not opened.

## Registered new-4k LLM table

The four rows were fixed before the new holdout was materialized. `Q0` is a
specialized Qwen reranker anchor; `Q1--Q3` share the same general
`Qwen3-0.6B` initialization, so the four rows are not a fully
matched-pretraining comparison. The table reports exact evaluator values for
`full - null` NDCG@10. Surface counts are recurrence `n=288`, strict transfer
`n=1,079`, other overlap `n=290`, no history `n=266`, and no observed positive
`n=2,077`.

| Method | Full NDCG@10 | Overall delta | Recurrence delta | Strict-transfer delta | Other-overlap delta |
|---|---:|---:|---:|---:|---:|
| Q0 Qwen3-Reranker-0.6B | `0.20573326504337267` | `0.013625261070750105` | `0.21541945369819093` | `0.00999694721584022` | `-0.0631947049240348` |
| Q1 InstructRec-GeneralQwen | `0.19734729236636003` | `0.014123912418449323` | `0.20732291670519742` | `0.006951766512099469` | `-0.03694588415122389` |
| Q2 RecRanker-GeneralQwen | `0.2043267926137048` | `0.015465424592572212` | `0.20970611550579965` | `0.009516324454170182` | `-0.030350955108382902` |
| Q3 TALLRec-GeneralQwen | `0.19738287091324844` | `0.013397721086875656` | `0.25671034382115254` | `-0.0016415494295340663` | `-0.06403607875352409` |

Exact normalized-query cluster 95% intervals for the same comparison:

| Method | Overall CI | Recurrence CI | Strict-transfer CI | Other-overlap CI |
|---|---|---|---|---|
| Q0 | `[0.008585758471656066, 0.0188243141895706]` | `[0.17658410467783425, 0.25389801001085194]` | `[-0.0014226097828673988, 0.021827846382170314]` | `[-0.09347730767829673, -0.03293656380468988]` |
| Q1 | `[0.009921341696861647, 0.018232393550873822]` | `[0.17373685364584326, 0.24300382439577684]` | `[-0.0015418938589130769, 0.015339119289369653]` | `[-0.058725742070418946, -0.0158061235633418]` |
| Q2 | `[0.010483405650533174, 0.020654943631812722]` | `[0.17460730937961189, 0.24402766067364545]` | `[-0.002520590118462954, 0.021743223484844726]` | `[-0.05882968082468504, -0.002062225171980034]` |
| Q3 | `[0.007573495478635253, 0.019325905195198615]` | `[0.21936711585782717, 0.29387101584686787]` | `[-0.016512324003094388, 0.01362658126103521]` | `[-0.09083281598304425, -0.03661546524480298]` |

All-request population-weighted contributions reconstruct each overall
`full - null` mean up to floating-point rounding:

| Method | Recurrence contribution | Strict-transfer contribution | Other-overlap contribution |
|---|---:|---:|---:|
| Q0 | `0.015510200666269746` | `0.002696676511472899` | `-0.004581616106992523` |
| Q1 | `0.014927250002774213` | `0.0018752390166388315` | `-0.002678576600963732` |
| Q2 | `0.015098840316417573` | `0.0025670285215124065` | `-0.0022004442453577603` |
| Q3 | `0.01848314475512298` | `-0.0004428079586168144` | `-0.004642615709630496` |

The registered wrong-user check gives the following exact `full - wrong-user`
means and intervals:

| Method | Overall | Recurrence | Strict transfer | Other overlap |
|---|---|---|---|---|
| Q0 | `0.015104568775387546 [0.010284813990763034, 0.020132376205437535]` | `0.23010090910575645 [0.19198713586341745, 0.268193710535082]` | `0.007387398063219378 [-0.0038791168233913683, 0.018474426629785604]` | `-0.047661342176280756 [-0.07568408126475597, -0.019815961696382316]` |
| Q1 | `0.01454466181881836 [0.010252650127105451, 0.01896632911184316]` | `0.22014757039879884 [0.1839745392489331, 0.25703874794867937]` | `0.006940650284670393 [-0.0015500786629586418, 0.015556879217526257]` | `-0.043837291919793006 [-0.06585202532049966, -0.02266448657060138]` |
| Q2 | `0.013482166268188192 [0.008747485175728343, 0.018324702491882186]` | `0.203193015091957 [0.1677423095886798, 0.23911907686034412]` | `0.006837252467971225 [-0.0036938973425100813, 0.017475651821925474]` | `-0.04127006443679918 [-0.06858047668778204, -0.015101648189220912]` |
| Q3 | `0.012273110045841404 [0.0068757173538035446, 0.017831361772381385]` | `0.24066423277534843 [0.20394081520997506, 0.27648082987764466]` | `-0.0019597541948919725 [-0.015353562078387357, 0.011190270438140694]` | `-0.06242856579188358 [-0.0914824807596403, -0.03513760302527815]` |

The wrong-user assignment is a diagnostic control: 2,745 of 3,571
history-present requests use the registered deterministic global-other-user
fallback. `full - wrong-user` therefore cannot by itself prove a causal or
provenance-matched user-specific effect.

## W0 diagnostic witness

W0 is a non-LLM structural witness outside the four-method main table.

| Comparison | Overall | Recurrence | Strict transfer | Other overlap |
|---|---|---|---|---|
| Full - null | `0.0012045907613015866 [-0.0006629350660166736, 0.0033676086516870184]` | `0.020144867701267992 [0.006080808381219785, 0.03875184007131775]` | `0.0003419396533854259 [-0.004734883418913395, 0.0052779133327130415]` | `-0.004663143926764502 [-0.011834899888699358, 0.0022178812386841277]` |
| Full - wrong user | `0.001969685479784197 [-0.00002484961103321487, 0.004121158449592508]` | `0.024695961134070018 [0.011862521020079862, 0.03902332074401092]` | `0.001026855136928894 [-0.0051849075545605545, 0.007237626229890884]` | `-0.001178177862833269 [-0.010606499460596955, 0.00804240221986728]` |

Its full NDCG@10 is `0.18855729717122086`; its `full - null` recurrence,
strict-transfer, and other-overlap weighted contributions are respectively
`0.0014504304744912954`, `0.00009223822150071863`, and
`-0.00033807793469042634`. It establishes a small recurrence response, not
recoverable strict-transfer headroom.

## Development and legacy compatibility

These supporting rows are `full - null` means. The strict-transfer interval is
included because Q1, Q2, and W0 had positive dev-only signals that did not
survive both compatibility checks.

| Population | Method | Overall | Recurrence | Strict transfer [95% CI] | Other overlap |
|---|---|---:|---:|---|---:|
| internal-dev 8k | Q0 | `0.014378955349174376` | `0.2526908378133236` | `0.0020539876506979527 [-0.006180203588345609, 0.010244866221595832]` | `-0.035227111485110284` |
| internal-dev 8k | Q1 | `0.012187754286809094` | `0.18752387972229603` | `0.012217456398350682 [0.005946514698689448, 0.018572529464017152]` | `-0.04358137075889147` |
| internal-dev 8k | Q2 | `0.016388550314547785` | `0.23915818855689963` | `0.01090560628887709 [0.00292602218934925, 0.018949003502578426]` | `-0.02939219514730125` |
| internal-dev 8k | Q3 | `0.01337311030271858` | `0.25179600025955384` | `0.003425002801329801 [-0.007292032059139483, 0.014370117427022919]` | `-0.05150967209879179` |
| internal-dev 8k | W0 | `0.0024854490360094713` | `0.02274336586580737` | `0.0033416220733538753 [0.00013405929572264018, 0.006434003159225006]` | `0.0008311302460272614` |
| legacy 2k | Q0 | `0.013898471832629149` | `0.24649372926283922` | `0.001652688923651536 [-0.015815603039338974, 0.01919705331664935]` | `-0.020344643055173074` |
| legacy 2k | Q1 | `0.010242288792387632` | `0.15010753154627055` | `0.010311194092041483 [-0.0026109292340018315, 0.023306485702935607]` | `-0.025030943495861324` |
| legacy 2k | Q2 | `0.011578058132705855` | `0.22642198523399432` | `-0.0012078299927468113 [-0.019308639971648497, 0.016641996482569775]` | `-0.024067938041652703` |
| legacy 2k | Q3 | `0.01632012219478505` | `0.27121399234328` | `-0.0018945833278077781 [-0.022736020460913865, 0.019225179775626565]` | `0.006299311651818516` |
| legacy 2k | W0 | `0.0009799184209398173` | `0.01741925892481521` | `-0.0027640270524358465 [-0.008648077317781205, 0.0029595305597670124]` | `0.009937212629215358` |

All ten analysis run IDs appear exactly once in `reports/dev_eval_log.jsonl`;
the five new-4k analysis run IDs also appear exactly once.

## Frozen run identities

`Full NDCG@10` and overall deltas above are copied verbatim from each shared
evaluator `metrics.json`; surface intervals and contributions are copied from
the colocated `motivation_v12_evidence.json`.

| Method | Frozen checkpoint | Config SHA256 | New-4k analysis run | Evidence SHA256 | Metrics SHA256 |
|---|---|---|---|---|---|
| Q0 | `q0_qwen3_reranker_06b@654f929996f7eb09f7b2` | `6e9f7d93dadf2c6946049ba7290912e753ab8a6c9297116e211926748429dd66` | `20260717_kuaisearch_q0_qwen3_reranker_06b_newholdout4k_analysis` | `659292cc76d408578d322569fc7118cd1300c1b91f141c5d0b87a6541977fbd5` | `5ee836cb60f4aceacd03213fb1a1bebff12cc942d091f9374d48ab35a21bb206` |
| Q1 | `q1_instructrec_generalqwen@9625a8c5a36327cec65f` | `1b68a3f4e79807862a0c1da369caa7b2d12218de1fe0dbf6ab2dfe18a5f15493` | `20260717_kuaisearch_q1_instructrec_generalqwen_newholdout4k_analysis` | `02dc4e85066778a44746af67d04f7db58c484dfeba75e56a09b52feb21f9da67` | `5dfdac5d7cbc217a27da466d34179cd8804999736abf8540a1d954caa5a31d4c` |
| Q2 | `q2_recranker_generalqwen@e207d2213741c16f997a` | `88a463fe48e5a884e99bf72cc3522a82031194f13cdd4b98966b160378e9a11e` | `20260717_kuaisearch_q2_recranker_generalqwen_newholdout4k_analysis` | `155bda5c735435d08642df3f8156aaf49739db840bfd5a100be8c5576780d8ae` | `93d16a99bd8db172b3bd0eaf195b253b9a4565d6dcf2b62db7109454f2563fda` |
| Q3 | `q3_tallrec_generalqwen@ea46a89671b63741ada8` | `ea8e0fb2d3421408cc51ecc216bfcfc7c7a0524e14a594d24009c9678235bd91` | `20260717_kuaisearch_q3_tallrec_generalqwen_newholdout4k_analysis` | `6359cd51a075642ed3c4b6821705659001e5c693008068503aa8e1ca34a617f9` | `3625966876884eba96d267d5e1c6bb55e361cea2934aac8fcd6c7e3ade43056a` |
| W0 | `w0_copps_style_transfer_witness@ddee4f219794be9e77f5` | `70c6a0290259f0ef1cef73b3505c111ebac7e8cddeba77c6eb6abb893709d784` | `20260717_kuaisearch_w0_copps_style_transfer_witness_newholdout4k_analysis` | `a168403ea2be1be34422e8cf27c244c7fd3779bdd88d842ffd93dcfb034ce6e2` | `080fb2a1e804733f6c9753e0e41e56c600a0c85af97f81b0198d6c79f8388856` |

## Current conclusion and non-results

At this seed, all four Q methods have reliable overall and recurrence gains
against both null and wrong-user histories. None establishes strict-transfer
gain on the new holdout under either comparison: every strict-transfer
interval crosses zero, and Q3 has a negative point estimate. Numerically, each
method's all-request recurrence contribution exceeds its strict-transfer
contribution, but no direct paired recurrence-minus-transfer test was
registered, so this is a decomposition statement rather than a significance
claim between surfaces. All four Q methods also show reliable negative
other-overlap response on the new holdout.

The supported first-round observation is therefore that these frozen LLM
ranker variants show recurrence-dominant history use. The expected W0 recovery
of strict-transfer headroom is not supported. This does not prove that
transferable signal is absent, that any official upstream method fails, or
that the result generalizes across seeds, model families, datasets, or time.

One pre-fix W0 scorer stopped after 15,880 rows because it hashed a normalized
encoder query instead of the raw-query request identity. It published no
metadata, read no qrels, and is permanently a mechanical non-result,
superseded by the v2 checkpoint and score runs above. Q2/Q3 resume canaries and
all smoke runs are engineering diagnostics, not transfer results. Q3 has the
same two fully tied requests under all three conditions, but all 4,000 requests
and 77,836 candidate rows are finite and identity-complete; this is an audited
score detail rather than a coverage failure. No valid method remains
under-converged or pending.

The concise machine-readable freeze is
`reports/motivation_current_summary.json`. The registered first-round stopping
rule was satisfied. The user has since authorized mechanism analysis under
`experiments/motivation/mechanism_analysis_plan.md`; this table remains frozen
and is not reused as a mechanism-stage development ledger.
