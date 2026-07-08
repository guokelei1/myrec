# Batch 2b 论证逻辑备忘：为什么要官方 baseline、为什么 B7-best 是参照点

日期：2026-07-08。来源：Batch 2 复审讨论。目的：论文写作和 Batch 2b
执行时防止叙述越界。doc 14 §0 的结论层级是本文的执行版。

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
M3 oracle   0.4232   ← 上方 +28% headroom（"不好"的上界证据）
B7-best     0.3305   ← 零智能静态加权的水位线
官方个性化方法  ≤ ?   ← 若连它都打不过（"不好"的下界证据）
```

三个用途：

1. **绝对刻度不存在**：NDCG 0.33 本身无意义，"效果不好"只能相对定义。
   "上面有空间没吃到 + 下面被笨方法压住"两条证据都以 B7-best 为参照。
2. **击碎"history 信号没用"的反驳**：B7-bge 显著高于 B2z
   （0.3305 vs 0.3056），证明 history 通道有真实增量信号。所以现有方法
   输不是信号不存在，而是利用不如固定权重加法。motivation 的精确形态：
   "个性化证据有价值（B7 > query-only），但现有方法用不好
   （官方 baseline ≤ B7）"。
3. **两个关键数字的锚**：M3 headroom 以"最强单方法"为分母；提议系统的
   贡献 = 系统 − 最强 baseline。最强者认错，两个数字都错。
   若 Batch 2b 出现更强方法，靶子换人，参照逻辑不变。

## 3. 已确证 vs 未确证（Batch 2 收尾时的状态）

| 结论 | 证据 | 状态 |
|---|---|---|
| query-only 在候选池内饱和 | B3≈B2z；B8a(7B) ≤ B7-bge；词法→bi-enc→cross-enc→LLM 四级无增益 | 扎实 |
| 最强 baseline 之上有大空间 | M3 oracle +28.0%，CI 下界 +27.2%，split-half 同向 | 扎实 |
| 现有个性化方法做得不好 | 仅弱 adapter 负结果 | **未确证，Batch 2b 的目标** |

## 4. 措辞上限（Batch 2b 即使全部结局 A 也不能越过）

1. 不能说"所有现有方法"，只能说"代表性方法"并列出测过的类别谱系；
2. 结论限定在此 dataset + 此 setting（固定候选池、blind records、
   ≤50 历史）；跨数据集泛化等 C4/C5；
3. B6+（2022 后的近年 PPS 方法）仍是缺口，投稿前需补至少一个；
4. Batch2 多通道 oracle +65.4% 是 selection-over-noise 膨胀值，只作
   定性异质性证据，主文引 +28% 三通道版本；
5. B8 full-dev 数字（含 B7 fallback）不进主表，B8 对比只用同 subset 报告；
6. B4/B5/B6 adapter 负结果只可出现在 appendix/实现说明。

## 5. 提议系统的定位备忘

- 提议系统 = 全新可训练 transformer 类架构，端到端学习个性化证据的
  per-request 利用；**不是路由系统**。oracle headroom 证明的是"不同请求
  需要不同证据利用方式"，attention 即最自然的 per-request 证据加权机制；
  insight 框架："何时以及如何利用个性化证据"。
- 对比表建议含一个廉价 learned router over 现有通道作为中间对照：
  打不过它 = 架构没学到超出通道选择的东西；打过 = 架构价值的直接证据。
- 系统开发可与 Batch 2b 并行：协议已冻结，baseline-to-beat 事后替换
  不影响系统代码。motivation 一节的最终措辞必须等
  `reports/pps_batch2b_decision_summary.md` 出来后按 doc 14 §0 填。
