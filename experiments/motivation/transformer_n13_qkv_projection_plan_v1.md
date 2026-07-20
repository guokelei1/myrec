# N13 Q/K/V projection-stage operator plan

状态：2026-07-20，独立于冻结 D0--D7 family 的新增 Transformer 内部诊断。

## 问题

N3/N11 已经分别观察或干预 attention history edge 与合并的 pre-softmax QK logits，
但这两种结果仍不能区分三条内部路径：readout query 的形成（Q）、history key 的形成
（K），以及 history value 的传输（V）。N13 在同一 block、同一 token span 和同一
full/null 请求上只改其中一条投影路径，回答“改变发生在 projection stage 的哪一条边”，
而不是把某个层或投影直接提升为设计。

## 固定操作

- Q：只在 native candidate/answer readout rows 对 `q_proj` 输出做 identity、0.5x、2x、
  sign flip；随后仍经过原生 RoPE、attention、o-proj 与 residual。
- K：只在冻结 history span 对 `k_proj` 输出做同样四种模式；随后仍经过原生 RoPE、mask、
  softmax 和 value transport。
- V：只在冻结 history span 对 `v_proj` 输出做同样四种模式；Q/K、softmax 与 o-proj 保持原生。
- full/null 均覆盖；不选择 head、neuron、层或 outcome slice。blocks 固定为 13/20/27，
  Q2/Q3 各 8,000 internal-dev requests。

## 条件与门禁

保留 full/null baseline、每个 Q/K/V 的 full/null identity、每个 component 的 half/double/
sign active conditions。identity 要求所有六个 identity cell 的 scorer delta ≤1e-5，projection
hook 必须恰触发一次，Q3 Yes/No prompt path 必须一致，所有请求/candidate score finite 且完整。
eligible content-control 仍沿用冻结 N11 parent；不合格请求写回冻结 full/null baseline，不打开
qrels。shared evaluator 在完整 coverage、hash、identity 和 implementation digest 审计后才读取
qrels，并使用 normalized-query cluster bootstrap 与固定 BH family。

## 解释边界

N13 的 component×mode 变化是 operator sensitivity，不等于 causal attribution、preference
direction 或可加和贡献。只有与 N8/N9/N12、matched wrong-user、cross-request 和结构负控结合，
并在 Q2/Q3 方向一致时，才可把 Q/K/V path 作为后续设计候选约束；即便如此，诊断 patch 仍不
是论文方法。机械失败、CPU smoke 和不完整 bundle 都记为 non-result。
