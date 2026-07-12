# 2026-07-11 C28 margin-local contest 终局

结论：**结构门正面，效用门负面；仍没有可称为“好架构”的模型。**

C28 只改变 C27 的候选比较图：按冻结的 request-zscored D2p margin 使用
`exp(-|gap|)` 连续局部权重，token evidence、奇函数 comparator、训练预算和
matched controls 保持一致。三 seed、五模式 GPU 运行在 781.68 秒内完成。

无标签 A0 首次 18/18 全过：相对 D2p 改变 70.0% 完整排序和 12.5% top-10；wrong
history 在三 seed 改变 30.8%/36.3%/21.8% 完整排序及 8/8/4 个 top-10。说明连续
margin locality 确实修复了 C27 的 top-set 稀释，不是简单放大 residual。

但 A1 打开 600 个 internal-A 标签后 11/11 utility checks 失败。primary-D2p 为
-0.000052 NDCG@10，CI 跨零；wrong history 略好于 clean，clicked correction
方向也为负且不显著。三个 seed 的 correction 几乎不相关，幅度相差三个数量级；
固定反号只帮助 seed 1/3，严重伤害 seed 2。因此不是统一符号写反，而是自由 odd
comparator 的方向/尺度在有限训练下不可识别、不可泛化。

不得挑 seed 2、事后反号、调 kernel scale 或用 delayed-B 救援。下一候选必须把
方向校准写入 Transformer 内部证据定律，而不是继续调整局部图或数据阈值；C28
delayed-B、escrow、dev/test 仍保持关闭。
