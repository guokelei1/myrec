# N14 history-embedding-stage operator plan

状态：2026-07-20，独立于冻结 D0--D7 family 的新增入口级 Transformer 诊断。

## 问题

现有输入干预改变 token 内容，不能区分“历史语义本身变化”和“同一历史 embedding
进入 Transformer 后的幅度/方向变化”。N14 在固定 token IDs、history span、mask、position
和 candidate boundary 下，只变换 `embed_tokens` 的 history rows，检验 signal 是否在
Transformer 入口就已被削弱或反向。

## 固定操作

- full/null 两条路径保持原有 token IDs 或冻结 content-neutral IDs；
- history embedding rows 只做 identity、0.5x、2x、sign-flip、zero；
- query、candidate 和 padding embedding rows 不变；所有 Transformer blocks 与 native readout
  保持原生；Q2/Q3 各覆盖 8,000 internal-dev requests。

## 门禁与解释

scorer 不读 qrels；identity、embedding hook exactly-once、Q3 shared prompt path、完整 finite
coverage 和 baseline hash 必须先通过。evaluator 只在 pre-qrels audit 后打开 qrels，固定
normalized-query cluster bootstrap、strict-transfer surface 和 BH family。embedding-scale
敏感性不是 preference attribution，也不说明应该直接缩放 embedding；只有与后续层级的
projection、attention、MLP、residual/readout 证据结合时，才可约束入口瓶颈假设。
