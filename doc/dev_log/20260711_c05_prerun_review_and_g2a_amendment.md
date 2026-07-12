# C05 pre-run review and G2a amendment

日期：2026-07-11

在任何 C05 data-fit outcome 产生前，三项独立 review（架构、实现、数据链）一致
判定原 CCEB 不能直接进入 G2。主要原因是 signal existence 与 mechanism attribution
混在一起、shuffle gate 与集合算子矛盾、exact atom 未保护最终 logit、attention
mass 未约束实际残差，以及原配置的 internal D2p scope 不合法。

因此 C05 改为先执行 architecture-nearest 的 G2a：只在冻结 train-internal 的
non-repeat 请求上训练普通 target attention，exact/centering/dead-zone/budget 和
corruption loss 全部关闭。它只回答冻结 D2 表示下是否存在廉价可学习的 cross-item
ranking signal。失败不外推为所有 LM/cross-item transfer 不可能。

G0 改用 D2 calibration checkpoint、internal-train popularity、alpha 0.6、FP32 和
完整候选集。selection 在读取 labels 前按 request-ID hash 冻结。G2a 通过后才允许
held-out hard-twin audit；CCEB、repeat path、dev 和 full training 均未授权。

用户在 review 后明确要求开始实验验证。物理 GPU 0 从已关闭 C01 释放并分配给
C05；环境 `myrec-c05`、prefix `20260711_kuaisearch_c05_`、G0+G2a 累计不超过 2
A40 GPU-hours，dev evaluator calls 为 0，test 继续锁定。
