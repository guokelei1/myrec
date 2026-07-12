# 2026-07-11 C26 token bridge 终局

结论：**负面；当前仍没有经过正面验证的好架构。**

C26 首次把搜索位置从 pooled D2 表示切到 query/candidate/history 的 WordPiece
token 层，并把 candidate-only late interaction、generic token triadic 与 pooled
history 放进同参数同算力对照。12 个终态全部完成，说明 token 数据链路和内部
Transformer 训练成立。

但 shared-query-token bridge 只改变 66/1,200 条完整排序和 1/1,200 个 top-10。
wrong history 在三个 seed 改变 92%--100% 的浮点 correction，却只改变
1/2/128 条排序，top-10 均为 0。这比 C25 多了一点 rank activity，但仍是典型的
sub-margin signal：历史进入了数值状态，却没有稳定承担候选决策。第三 seed 的
10.67% order change 不能覆盖前两个 seed 的 0.08%/0.17%，因此不能选 seed 讲故事。

`2.27e-5` 的中心化容差失败可工程修复，但两个核心门独立失败，禁止用 fp64、
放大 residual 或调 score cap 重开 C26。internal-A、delayed-B、escrow、dev/test
均未打开。

下一步应改 Transformer 内部的排序作用位置：让 token 证据控制候选间竞争或
listwise readout margin，并以同一 token bridge 的 additive residual 作为最近邻
对照。这里的新假设是“交互位置导致承重差异”，不是“更大 correction 会更好”。
