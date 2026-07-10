# Batch 2b 论证逻辑备忘：为什么要官方 baseline、为什么 B7-best 是参照点

日期：2026-07-08。来源：Batch 2 复审讨论。目的：论文写作和 Batch 2b
执行时防止叙述越界。doc 14 §0 的结论层级是本文的执行版。

2026-07-10 审计更新：本文保留为当时的决策记录，但 M3 "上方 headroom"
解释已被 `reports/pps_m3_m4_random_canary_audit.json` 否定；B8 也不是
query-only 方法。当前只保留下方静态水位线与 baseline 比较逻辑。
同日后续 C3-R 用先锁定的 matched wrong-user history 控制补齐了身份特异性：
true B7 在 history-present / same-query donor 子集分别平均高 +0.0431 / +0.0321。
该身份解释后来因时间不对称被 C5-R2 supersede；当前 `doc/22` gate 未通过，
不存在正向设计授权。见 `reports/pps_c5r2_temporal_symmetric_identity.json`。
同日 D1/D2/D2h/D2s 强化后，完整静态对照 D2s（D2p + causal B0b）以
0.3416 显著超过遗漏 popularity 的 D2h 0.3352，已成为当前静态水位线；
本文其余 B7 数字只解释 Batch 2b 当时为何需要官方 baseline。
最终 C5-R3 component audit 又显示 item-only mean 0.3453755 高于 full D2s，
category-only 无独立显著增益，故 item-only 才是当前静态水线；C5-R3
primary/fallback 均失败，故该 recovery ladder 按冻结规则终止。当前解释不是
“没有 design insight”，而是更窄的 evidence-fidelity failure：exact recurrence
可靠，未经校准的跨 item/category transfer 不可靠。它允许 design formulation，
但不验证任何具体 primitive，也不授权在新 design-specific gate 前正式训练。

## 1. Strawman 问题（Batch 2 adapter 负结果的边界）

B4/B5/B6 是 hashed 特征 + 在线 logistic 的简化 adapter：

- B4 vs 真 SASRec：无 self-attention、无学习 embedding，只有 item-id
  转移统计；
- B5 vs 真 DIN：无 attention 软选择，只有手工 overlap 特征；
- B6 vs 真 HEM/ZAM/TEM：无联合表示学习，只有 gated overlap 特征。

它们输给 B0b（recency 启发式）本身说明容量不足。因此：

> "我的简化 adapter 输了" ⇏ "SASRec/DIN/HEM 这类方法在此数据上会输"

论文若写"经典个性化方法不行"，审稿人一句"你比较的是 strawman"即可拒稿。
这也是提议系统（可训练架构）主实验成立的前提：同类（可训练、用 history）
对照必须是真实强度，否则"新架构打败 baseline"无说服力。

## 2. B7-best 不是上界，是参照点（三明治结构）

B7 是刻意最笨的双通道方法：z-score × 全局固定 α，全 dev 只选一次 α。
分析上界是 M3 oracle。B7-best 的价值在于"笨却打不过"：

```text
M3 oracle   0.4232   ← 历史登记值；Random oracle 更高，不能作 headroom 证据
B7-best     0.3305   ← 零智能静态加权的水位线
官方个性化方法  ≤ ?   ← 若连它都打不过（"不好"的下界证据）
```

三个用途：

1. **绝对刻度不存在**：NDCG 0.33 本身无意义，"效果不好"只能相对定义。
   当前只有"下面被静态方法压住"这条比较可用；"上面有空间"须重做
   noise-controlled gate。
2. **击碎"history 信号没用"的反驳**：B7-bge 显著高于 B2z，且 matched
   wrong-user history 显著弱于 true history，证明 correct-user history 有
   identity-specific 增量信号。所以现有方法
   输不是信号不存在，而是利用不如固定权重加法。motivation 的精确形态：
   "个性化证据有价值（B7 > query-only），但现有方法用不好
   （官方 baseline ≤ B7）"。
3. **两个关键数字的锚**：静态 waterline 定义系统必须超过的数值门槛；
   matched-history 差定义身份证据必须保留的机制门槛。若未来出现更强
   baseline，数值靶子换人，但 wrong-history/no-history 归因规则不变。

## 3. 已确证 vs 未确证（Batch 2 收尾时的状态）

| 结论 | 证据 | 状态 |
|---|---|---|
| 候选池 query-conditioned，已测 query-only scorer 边际有限 | B1/B2z/B3 聚集；query/candidate canary 通过。B8 不属于此链 | 有边界地成立 |
| correct-user history 有 identity-specific predictive value | matched wrong-history 三 seed + same-query subset | 成立 |
| 代表性个性化方法未超过 B7 | B4o/B5o/B9 正式结果及 caveat | 有边界地成立 |

## 4. 措辞上限（Batch 2b 即使全部结局 A 也不能越过）

1. 不能说"所有现有方法"，只能说"代表性方法"并列出测过的类别谱系；
2. 结论限定在此 dataset + 此 setting（固定候选池、blind records、
   ≤50 历史）；跨数据集泛化等 C4/C5；
3. B6+（2022 后的近年 PPS 方法）仍是缺口，投稿前需补至少一个；
4. Batch2 多通道 oracle +65.4% 与三通道 +28.0% 都是
   selection-over-noise 失败诊断，不得作定性异质性或 headroom 证据；
5. B8 full-dev 数字（含 B7 fallback）不进主表，B8 对比只用同 subset 报告；
6. B4/B5/B6 adapter 负结果只可出现在 appendix/实现说明。

## 5. 提议系统的定位备忘

- 提议系统 = query-anchored personalized-residual transformer，端到端学习
  query-candidate base 与 masked target-history residual；**不是路由系统**。
  该 insight 的 premise 已通过 C3-R/C5-R，但具体架构仍须靠消融归因。
- 对比表建议含一个廉价 learned router over 现有通道作为中间对照：
  打不过它 = 架构没学到超出通道选择的东西；打过 = 架构价值的直接证据。
- 系统开发可与 Batch 2b 并行：协议已冻结，baseline-to-beat 事后替换
  不影响系统代码。motivation 一节的最终措辞必须等
  `reports/pps_batch2b_decision_summary.md` 出来后按 doc 14 §0 填。
