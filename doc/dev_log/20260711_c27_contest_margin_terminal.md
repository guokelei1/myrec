# 2026-07-11 C27 pairwise contest 终局

结论：**仍是负面，但第一次接近通过结构门；当前还不能称为好架构。**

C27 用奇函数 comparator 和 soft-Borda contest 替换 C26 的独立 scalar residual。
结果验证了作用位置很重要：相对 D2p 改变 42.17% 完整排序与 5.5% top-10，wrong
history 在三 seed 改变 12.33%/14.33%/22.17% 排序；反对称、互补概率、neutral
base identity、置换与全部回退合同均通过。

唯一失败是 wrong-history top-10 的 all-seed 门：冻结要求每 seed 至少 3/600，实际
为 2/1/7。17/18 不是“约等于通过”；不得把阈值降到 1 条、挑第三 seed 或加大
`pair_delta_max`。internal-A、delayed-B、escrow、dev/test 均未打开，因此没有 utility
结论。

失败结构也给出比继续换 evidence 表示更具体的下一步：uniform all-pair Borda 把
大量远离决策边界的 pair 一起平均，历史变化虽能改低位排序，却较少穿过 top set
边界。后继应测试与 D2p margin 距离有关的通用局部 contest graph，并以 C27 uniform
contest、candidate-only、generic 与 additive 作为匹配控制。局部性必须基于连续
margin，而不是硬编码 top-10 或数据集类别。
