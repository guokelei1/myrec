# R0 Round 1 流程复盘与修订决定

日期：2026-07-13
状态：只读复盘；pipeline 仍为 `PAUSED`，不授权 Round 2、训练或 evaluator call。

## 结论

Round 1 有真实正面进展，但按当前产出不能形成 CCF-A 级 proposed-architecture 论文。
它完成的是可信 instrumentation、主轨 full-token observability 和两个错误问题的排除；
尚未形成强方法共享的 blind spot、ordinary Transformer 的 native shortfall 或可支付
unique rent 的 architecture consequence。

最关键的证据是一个“资产与缺陷混杂”状态：

- Transformer 资产成立：三 seed `true-null +0.020936`、`true-wrong +0.031812`；
- 当前 recipe 的基础排名能力不成立：true NDCG@10 `0.331028`，低于 item-only
  `0.345376`；
- no-history base degradation 为 `-0.013066`，replication fold 95% CI
  `[-0.025734,-0.006001]`；
- R0-FIDEA-01/02 均被强对照或方向结果证伪。

因此 Round 1 达到 revised contribution ladder 的 L1，而非 L2/L3。流程不能把
“history signal 可见”误写成“新架构必要”。

## 分步骤审计

| Step | 实际产出 | 判断 | 流程修正 |
|---|---|---|---|
| R0-A scope audit | 三轨信息对象、holdout 和 label boundary 可信 | 必要且有效 | 保留；跨轨旧结果必须另做 equivalence card |
| wrong-user control | 6 次 donor/serialization 修复后 coverage `0.982141` | 工程质量提升，不是六次科学迭代 | engineering iteration 与 scientific round 分开计数 |
| R0-B observability | KuaiSearch true/null/wrong 信号稳定；shuffle 不承重 | Round 1 最重要的正面结果 | 明确为 Transformer asset，不直接授权 architecture |
| R0-C tuning | 只比较一个 sentence-encoder family 的 token budget、LR、dropout；24 evaluator calls 覆盖 4 config + 2 seeds | family 太窄，且过早做多 seed；“normally tuned”不等于“strong” | 先做 model-family adequacy，再做局部 tuning；config trials 与 evaluator calls 分账 |
| T004 selection | 相对 T002 `+0.000907` 且 CI 跨零，仍按最高点估计选中 | 容易追逐噪声 winner | 小于 MDE/CI 跨零时选更简单、base-preserving 配置 |
| R0-D atlas | 两个 idea 被证伪；发现 no-history degradation | 正确地没有造 architecture，但 atlas 开始过早 | base degradation 应在 adequacy gate 阻断 atlas并路由 R0-C0 |
| Failure idea quality | repeat-conflict 连 item-only 都能解决；query-aligned nonrepeat 上 full-token 已有正点估计 | 没有 shared blind spot，也缺少可恢复整体价值 | 每个 idea 必须来自 Motivation Brief，并报告 prevalence × severity、强 baseline matrix 和 paper payoff |

## 为什么现流程容易原地打转

1. **缺少 problem-value gate。** 流程要求 failure 可复现，却没有先要求它值得成为论文
   核心。可构造的小 slice 会自然压过更重要但更难定义的高层问题。
2. **strong baseline 定义过弱。** 稳定、正常调参和有 history effect 都不能替代基础
   排名竞争力。一个 no-history 显著退化的模型会制造大量伪“Transformer failure”。
3. **搜索粒度倒置。** Round 1 先调 token budget/LR/dropout，尚未比较 ranking-
   pretrained cross-encoder、same-backbone query-candidate base 和 objective family。
4. **预算单位混淆。** 四个 scenario 让一次配置试验消耗四次 evaluator call；结果是
   16-call ceiling 只覆盖四个配置，却又在 motivation survivor 出现前投入多 seed。
5. **motivation 与 novelty 太晚。** nearest prior/simple answer 和 reviewer-facing paper
   thesis 直到 Failure Card/Hxx 后才变得 binding，容易先得到局部现象再后验找故事。
6. **进展计数偏工程活动。** Round 1 有 16 个记录，其中大量是 control materialization
   与锁修复；这些值得记录，但不应制造“研究已经走了很多轮”的错觉。

## 修订后的研究顺序

```text
R0-A/B: information object + observability
  -> R0-M: quantitative Motivation Brief
  -> R0-C0: model-family/base adequacy
  -> R0-C1: within-family tuning
  -> R0-D: shared-blind-spot probes
  -> Failure Card: native shortfall + paper value + nearest work
  -> Hxx: one small structural repair
```

高层与细节不是二选一。先用高层问题决定“为什么值得做”和“哪些强方法共同失败”，
再用具体 intervention 定位到 representation/attention/objective locus，最后才探索细小
结构。过早 high-level 会变成口号；过早 low-level 会变成没有论文意义的局部调参。

## CCF-A 路径的最低叙事闭环

后续只有同时建立以下链条，才有 CCF-A proposed-method 潜力：

1. 一个覆盖足够请求、造成可量化整体损失的真实 ranking problem；
2. 强 non-Transformer/static/upstream 方法与 adequate ordinary Transformer 共享盲点；
3. ordinary Transformer 已建立的 token/context asset 在该问题上仍有必要；
4. native shortfall 被定位，ranking pretraining、容量、context、objective 等简单修复无效；
5. 一个 primitive 修复 targeted failure，同时保留 query relevance、history utility 与
   no-history base；
6. targeted gain、overall gain、specificity 和 unique rent 在 independent confirmation
   中同时成立。

流程能提高得到这条链的概率，但不能保证 CCF-A。若 L2/L3 仍无法建立，及时转为
cross-dataset measurement/negative-design paper 或换题，比继续枚举 Transformer 模块
更有研究价值。

## 已修改的权威规则

- `doc/31`：加入 contribution ladder、R0-M、R0-C0/C1、strong-base adequacy、
  problem-value fields、早期 nearest-work/reviewer audit 和 revised next-round order。
- `doc/32`：scientific round 与 engineering iteration 分离；默认最多 3 个 scientific
  rounds；增加 motivation/base-adequacy transition，并把 config trials 与 evaluator
  invocations 分账。
- `doc/15`：固定“高层问题 → shared blind spot → native shortfall → 小结构修复”的
  粒度顺序，并登记 Round 1 资产与 base weakness。
