# 2026-07-11 C24 多复现候选竞争终局

结论：**负面，C24 在 label-free A0 终止，当前仍没有好架构。**

C24 不是最终创新声明，而是一个信号存在性门：若普通 set Transformer 的
candidate-candidate 边能稳定改变排序，才值得继续发明 recurrence-specific
competition operator。冻结运行完成 3 seed × 3 等参数模式 × 2 epoch。模型整体相对
item-only 改变 38.0% 排序和 15.5% top-10，说明训练并非完全无效；但同 checkpoint
删除跨候选边后，三个 seed 均为 0/600 排序变化。跨边只改变很小的数值校正，未承担
任何排序决策。

这把 C23/C24 的共同失败定位得更清楚：只要静态 exact-recurrence anchor 和每候选
独立特征同时可用，Transformer 会走独立校准捷径；把 suffix 或候选集合接入 attention
并不会让这些信息自动变成 load-bearing state。下一步不得在 C24 上调层数、扩大残差
或加辅助损失，也不得把 generic set attention 包装成创新。新候选应先改变
representation formation，使目标状态无法由独立 recurrence scalar 重建，再用同参数
shortcut control 和表示级干预验证该状态真的进入排序。

内部 A、escrow、dev/test 均未打开。权威报告为
`reports/pps_c24_train_gate.json`。
