# 2026-07-11 C25 anchored Möbius 终局

结论：**负面；当前仍没有经过正面验证的好架构。**

C25 是第一次从 residual 中结构性删除 D2p、recurrence scalar 与 raw candidate
bypass，只允许共享势函数的三阶离散导数进入 Transformer。12 个等参数、等势函数
求值模型均完成，说明工程与优化链路成立。但 primary 只改变 2/1,200 个 top-10；
wrong history 虽改变所有浮点 correction，却在三个 seed 只改变 15/19/22 条排序，
top-10 全为 0。它仍没有把候选相关历史变成承重排序状态。

fp32 Möbius 抵消与候选中心化分别有 `6.35e-6`、`3.54e-5` 的严格容差失败；即使把
这两项视为可修的数值实现，两个核心 activity/corruption 门仍独立失败，因此不得以
高精度重算为由重开 C25。internal-A、delayed-B、escrow、dev/test 均未打开。

C23--C25 的连续证据现在足以关闭“在同一 pooled D2 state 上继续换 sequence、set
或高阶代数算子”的搜索支路。下一个合理的架构位置是 token-level representation
formation：保留 query token、candidate title token 与 history title token 的细粒度
桥接，再问跨商品证据是否存在；若普通强 token-level 近邻也失败，就应进一步收缩
跨商品 personalization 主张，而不是再发明 pooled-state 算子。
