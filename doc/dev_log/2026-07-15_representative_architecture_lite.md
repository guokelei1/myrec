# 代表性架构 Lite 落实记录

日期：2026-07-15

本轮只做 motivation baseline，不提出或训练 proposed architecture。核心矩阵是 ordinary Qwen、
rec-native HSTU、KDD 2025 LLM-SRec；BGE 作为已有 encoder anchor，SASRec 是 HSTU 的同接口
matched control。

## 已完成的工程闭环

- 固定并 vendor 了 Apache-2.0 的 Generative Recommenders/HSTU 官方源码，commit 为
  `6135bc30398f97e5786674192558d91f2ef2fa90`；官方 HSTU 和同仓 SASRec 的 GPU
  forward/backward、结构参数有限差分检查均通过。
- 建立统一的 causal sequence adapter：只读 standardized visible records，当前 query 是最后
  一个 causal token，true/null/wrong 不改变候选身份和顺序。
- 发现 dev item OOV 很高，因此所有序列模型绑定冻结文本内容向量，不能靠纯 ID lookup。
- 第一版错误地把 reranker 的单文本 CLS 当作双塔向量，HSTU 几乎只产生共模位移；该 recipe
  被降级为不充分 baseline。第二版换为本地冻结 `bge-small-zh-v1.5` CLS 双塔向量，HSTU 与
  SASRec 使用完全相同的外层 scorer、容量、目标和更新预算。
- 根据 LLM-SRec 论文公式独立实现 retrieval、MSE distillation、uniformity、CF embedding
  injection 和 `UserOut/ItemOut` 表征抽取；没有复制无许可证的官方源码。
- 用 train-only SASRec checkpoint 物化冻结 teacher item/user 表征；物化过程不读 dev qrels。
  LLM-SRec 完整 Lite 训练中 Qwen 与 SASRec 均冻结，只更新论文允许的轻量模块。

## 结果判断

第二版 HSTU 在几乎所有 history-present 请求上产生实质的 candidate-relative response；在
strict-nonrepeat 上方向点估计低于随机、区间覆盖随机，true-null 排序收益为负且区间跨零。
matched SASRec 也表现为大面积响应、strict-nonrepeat 方向接近随机、收益不稳定。LLM-SRec
在一轮完整 Lite 训练后同样没有把该症状消除：true 低于 null 的 overall 点估计，
strict-nonrepeat 的方向仍与随机相容。

这与已有 Qwen/BGE 结果的形状一致，因此是有价值的跨范式支持性证据。但它还不是 binding
shared-failure 结论：HSTU-QC 和 SASRec-QC 均低于 BM25，HSTU 的 repeat utility 正控制也未
显著；LLM-SRec 没有独立 QC，只做了一轮完整训练，而且 KuaiSearch 不是其最接近原论文的
Amazon 条件。

因此本轮最重要的结论不是“三个都失败”，而是：**代表矩阵已经真实可运行，症状在三种
sequence-oriented 实现上都出现；但序列 baseline adequacy 仍是提升 claim 等级的主要阻塞。**

## 下一步

先完善普通 baseline，而不是开始新架构：为 HSTU/SASRec 建立更强的 query-candidate search
task adapter 或标准 sequential pretraining，使 QC 至少与强 base control 竞争；LLM-SRec 按
协议优先迁移到 Amazon-C4 做 origin-compatible 正常训练。只有 adequate survivor 再做第二
seed 和 KuaiSearch Full。

汇总判断见 `reports/history_response_gap_lite_representative_architecture_decision.json`。
