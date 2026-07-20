# Transformer deep-dive interpretation: attenuation before reversal

状态：2026-07-18，用户在 D1 与 Q3 native-position gate 已封口、D2 全层因果
sweep 仍在运行时指出：更自然的机制可能是中层历史信号被某层削弱或抹去，而不是偏好
信号本身发生方向反转。本说明是结果解释边界，不修改冻结计划、family、层、条件、队列或
stopping point。

## 1. 术语收紧

当前证据只支持：`same_full_to_null - null` 对严格迁移 target margin 的因果效应在
block 13 为正、block 27 为负。后续报告固定称为 **endpoint effect sign transition**，不能称为
activation/preference vector reversal，除非预注册的 direction/scale controls 直接支持后者。

block-27 full-state patch 复现有害 full behavior，说明晚层状态足以携带该行为的全部成分；它不
区分以下机制：

1. 有益历史信号被衰减，其他 query/popularity/common-offset 分量接管；
2. 历史信号仍在，但被旋转到 native readout 不再利用或错误利用的子空间；
3. attention/MLP 新增一个幅度更大的 anti-transfer 分量，覆盖仍存在的有益分量；
4. preference direction 本身发生几何反向。

## 2. 已完成证据更符合衰减解释

D1 candidate-readout category `real-minus-random` region estimates：

| Model | blocks 0--6 | 7--13 | 14--20 | 21--27 |
|---|---:|---:|---:|---:|
| Q2 | 0.7208 | 0.6940 | 0.6658 | 0.4826 |
| Q3 | 0.6378 | 0.5229 | 0.4245 | 0.4009 |

D1 candidate-readout category `full-minus-null excess`：

| Model | blocks 0--6 | 7--13 | 14--20 | 21--27 |
|---|---:|---:|---:|---:|
| Q2 | 0.0254 | 0.1650 | 0.1328 | 0.0250 |
| Q3 | 0.0331 | 0.2245 | 0.1454 | 0.0930 |

Q2 的末区间 CI 跨零且未通过 BH；Q3 末区间仍为正但相对中层明显减弱。这些是表示性证据，
支持 attenuation/overwriting 作为当前首要解释，但不能单独定位因果层或分支。

这里的 attenuation 只指 **task-aligned、candidate-relative preference structure**，不能解释为
所有 history-conditioned hidden-state difference 的范数都在晚层归零。完整 strict-transfer
candidate-readout geometry 恰好给出相反的粗粒度现象：

| Model | state 13 full-null L2/sqrt(hidden) | state 28 | ratio |
|---|---:|---:|---:|
| Q2 | 0.1413 | 3.7846 | 26.8x |
| Q3 | 0.1057 | 1.2632 | 12.0x |

两模型的 full/null RMS ratio 在这些位置仍接近 1。因而当前更精确的工作假设是：总的历史诱导
位移继续累积，但其中可泛化、可用于候选相对排序的偏好分量被稀释、旋转到 native readout
不用的子空间，或被更大的 request-common/anti-transfer 分量覆盖。粗粒度 L2 增长本身不能证明
偏好保留，也不能证明反转；D2 的相邻层因果曲线和层内分支分解仍是定位依据。

一个结果前冻结、qrels-blind 的高维样本给出了更直接但仍属描述性的几何检查。对 512 个
`candidate not in original history` 行（482 requests），把每个 state 的 full-minus-null delta 投影到
train-only real-label 与 random-label probe 的判别行空间。candidate-readout category 在 state 13
的 real/random 投影能量占比为 Q2 `0.002674/0.001613`、Q3 `0.003900/0.001639`；到 state 28
变为 Q2 `0.001385/0.002349`、Q3 `0.001268/0.002560`。固定区间的 real-minus-random 投影份额
在两 fold 中方向一致：Q2 `blocks 7--13` 为 `+0.000465`，`14--20/21--27` 为
`-0.000551/-0.000416`；Q3 相应为 `+0.000356/+0.000630/-0.000736`。brand 的区间 excess
则都接近零，没有同样清晰的 category 模式。

这项几何审计没有读取 qrels，也没有选择最佳层，并保留全部 29 states；但样本量只有固定 512
candidate rows，线性 probe 子空间也不等于 native causal preference path。因此它不能指定擦除层，
也不能证明 state 28 的信息为零。它只增强一个较窄的解释：category-aligned full-null 分量在中层
相对随机方向更特殊，到末层这种特殊性消失，而非整个 history delta 发生字面反转。具体覆盖点
仍必须由 D2 相邻层 patch、独立 fold 和 attention/MLP branch rescue 确认。

同一组 train-only probe 权重还允许检查“是不是判别坐标系本身在后层越来越乱”。将各 state 的
standardized ridge 系数还原到 raw residual coordinates，比较相邻 state 判别子空间的 principal
angles 后，candidate-readout category 的 real-label 平均平方 canonical cosine 反而随固定区间上升：
Q2 为 `0.251/0.296/0.512/0.679`，Q3 为 `0.207/0.278/0.398/0.601`；random-label 控制也呈相近
深度趋势（Q2 `0.220/0.277/0.483/0.655`，Q3 `0.198/0.270/0.377/0.580`）。例如
state `12 -> 13` 的 real similarity 虽低（Q2/Q3 `0.156/0.248`），random control 也同样低
（`0.165/0.261`），不构成 preference-specific 断裂。

所以目前不支持“末层 preference decoder 坐标系逐层崩散”这一简单解释。更一致的描述是：后层
category 判别子空间趋于稳定，但 history-conditioned delta 对真实子空间的特殊对齐下降，或 native
readout 对其中有用分量使用失准。独立 ridge probe 的 principal angle 仍可能受估计稳定性与层间
尺度影响，不能替代 activation patch；它只是把搜索重点从全局 subspace chaos 进一步推向具体
attention/MLP 覆写、candidate-relative composition 与 final readout calibration。

位置对照又把 Q2 的描述性定位收紧了一步。Q2 category real-minus-random subspace excess 在
`history_summary_end` 的四区间为 `-0.000467/+0.000513/+0.000145/+0.001553`，晚区间 fold 0/1
分别为 `+0.001564/+0.001542`；同一模型的 `candidate_readout` 则为
`-0.000109/+0.000465/-0.000551/-0.000416`，晚区间两 fold 均为负
（`-0.000410/-0.000422`）。因而 Q2 的 category-aligned 成分不像在所有 token 位置一同消失：
它在晚层 history-summary 位置仍相对 random probe 特殊，却没有同样到达 candidate readout。
这提高了 history-to-candidate transport/recipient composition 作为 Q2 候选瓶颈的优先级，并给
D3 history→readout edge 干预一个清晰预测。位置专属 probe 不能直接作可加路径分解，而且 Q3
没有复现同一模式，所以在 D3 causal result 前仍只称 model-specific descriptive localization。

但同一固定样本的 raw residual cross-position geometry 排除了“Q2 总 history delta 根本传不到
candidate”的更简单版本。Q2 `history_summary_end` 与 `candidate_readout` full-null delta 的平均
cosine 按固定区间从 `0.023/0.046/0.195` 升到晚区间 `0.557`，candidate/history RMS ratio 从
`0.065` 升到 `0.340`；state 27/28 的 cosine 更达到 `0.646/0.700`。两 fold 的区间值几乎相同。
Q3 的总跨位置对齐则始终弱得多，四区间 cosine 仅
`0.012/0.017/0.041/0.089`，state 28 的 signed mean 又降到 `0.006`（absolute mean `0.076`）。

因此 Q2 的优先问题不是 history transport 的总量不足，而是 **transport selectivity**：越来越多的
总体 history-conditioned state 到达 candidate，同时上一段显示 category-specific alignment 在晚层
下降。这与“有用偏好分量被更大的通用/无关历史分量淹没”一致，也警告不能用增加 history
attention mass 作为默认修复。Q3 可能更接近弱 transport、强重编码或不同 recipient composition，
不能套用 Q2 机制。D3 必须把 logits-mask、value-edge zero、neutral K/V 和 head/GQA observations
结合起来，回答传输内容与选择性，而不只回答 attention edge 是否非零。

对冻结 512-row 控制覆盖的 482 个请求扩展到其全部 `20,357` 个候选后，request-common 与
candidate-relative 分解把 Q2 的“淹没”描述直接量化了。Q2 四区间的 common energy fraction 为
`0.941/0.924/0.918/0.977`，candidate-relative residual/common RMS ratio 为
`0.244/0.281/0.287/0.149`；common RMS 从 `0.036` 增到 `2.262`，residual RMS 也从
`0.0087` 增到 `0.341`。所以 residual 没有在绝对幅度上被擦成零，而是 common 分量增长得快得多。
同时 residual 的 category real-minus-random probe 投影 excess 从中后区间 `+0.000441` 降到晚
区间 `+0.000017`，晚区间 fold 0/1 为 `+0.000046/-0.000014`；common 与 history-summary delta
的 cosine 则从早区间 `0.023` 升到晚区间 `0.562`。这支持“越来越强的、主要为 request-common
的历史传输淹没 candidate-relative factor selectivity”，而不是字面反转或总信号消失。由于 shared
linear readout 下 common activation 才严格 rank-invariant，这仍是几何诊断，不能替代 score/branch
因果结果。

逐 block 的精确 residual-stream 能量恒等式又排除了“Q2 晚层整体抹掉 candidate-relative delta”。
在 blocks 21--27，Q2 的 482 个冻结请求没有任何一次 candidate-relative output energy 低于 block
input；该区间已有 residual 与新增 update 的平均 cosine 为 `+0.057`，不是负向覆盖。但 common
energy change 均值为 `1.9651`，candidate-relative 仅 `0.03788`，相差 `51.9x`，且 block-update
energy 的 `97.35%` 属于 common 分量；fold 0/1 的 common/residual change、cosine 和 share 几乎
一致。唯一在总体均值上明显收缩 candidate-relative energy 的是早期 block 4（Q2/Q3 分别
`99.8%/99.6%` 请求下降），Q3 block 7 还有一个接近零的小收缩；它们同时出现在两模型且早于
category 中层峰值，不能解释 Q2 特有的晚层现象。因此“弄没”的更精确含义应是任务相关方向被
大量 common/off-task update 相对稀释或旋出，而不是 residual norm 在晚层被消灭。D2 selected-node
和 D3/D4 仍需定位这种 composition 改变来自 attention 还是 MLP。

Q3 构成重要反例：其 common energy fraction 从 `0.939` 降到 `0.848`，residual/common RMS ratio
从 `0.249` 升到 `0.418`，晚区间 residual category excess 反而为 `+0.000302`。因此 Q2 的
common-dominance 不能升级为 LLM4Rec 的统一层深机制；Q3 仍需由 attention content、MLP 与 native
readout 干预区分弱 transport、重编码和输出组合问题。

同一 482-request、20,357-candidate qrels-blind anchor 的 activation-channel 审计进一步区分了
“common 变大”和“跨请求收敛到通用方向”。按每请求等权，把 full-minus-null candidate delta
逐 channel 分成 common 与 residual energy 后，Q2 晚区间 common/residual/history-summary 的
归一化 channel participation ratio 为 `0.237/0.237/0.0888`，Q3 为
`0.353/0.407/0.204`；Q2 history delta 明显更集中。Q2 history unit vector 的跨请求平均
pairwise cosine 在晚区间达到 `0.808`，其 channel-energy profile 与 candidate-common 的 cosine
为 `0.768`；Q3 对应只有 `0.736/0.474`。Q2 fold 0/1 的 history participation
`0.0880/0.0895`、pairwise cosine `0.811/0.806`、common-history profile cosine
`0.766/0.770`，不是单一 query fold 驱动。

state 28 的差异更集中在 history path：Q2 history participation 仅 `0.0462`、跨请求 cosine
`0.866`、与 candidate-common channel profile cosine `0.863`；Q3 为
`0.136/0.754/0.540`。这与 Q2 晚层把不同用户的 history-conditioned state 压向较通用、较少
channel、并同步传到 candidate-common path 的解释一致，比“每个用户各自无关地 common 变大”
更具体。它也与前述 history-summary 到 candidate 总 delta 的高 cosine 相容：Q2 的 transport
不是弱，而是越来越由跨请求共享的方向/通道主导。

但 channel concentration 本身仍不是故障标志。state 28 candidate-residual participation 几乎相同
（Q2/Q3 `0.113/0.108`），Q3 candidate-common participation 甚至比 Q2 更低
（`0.031/0.154`，即更集中）。这些量依赖 residual-stream 坐标基，global alignment 也可能包含
prompt/template 共性，不能指定 neuron、head 或 MLP group。当前只提高一个更窄的 D3/D4 预测：
Q2 若存在主要机制，应表现为 history content 被 attention/MLP 组合成通用 candidate-common
更新，而不是“少数 channel”本身；必须由固定 head/edge、neutral K/V 与 MLP-group 因果结果验证。

query-end causal-floor 负控排除了“Q2 通用方向只是分批/低精度/位置公共偏移误差”的主要替代解释。
query endpoint 严格位于 history 之前，exact causal arithmetic 下 full/null 应完全相同；实际 FP16
snapshot 的小差异只作为机械底噪，不从科学 delta 中相减。Q2 晚区间 candidate-common/query-floor
RMS ratio 为 `17.29`，candidate residual/query-floor 也有 `2.63`；common 与 query-floor 的
energy-weighted direction cosine 仅 `0.0100`，common 的 `99.97%` 能量在该控制方向的正交补中。
state 28 相应为 `18.23x`、cosine `0.0194`、正交份额 `99.96%`，fold 0/1 的 common SNR
`18.06/18.39`、cosine `0.0189/0.0198`。

更关键的是，同一 state 28 Q2 candidate-common 与 history-summary 的 signed cosine 为 `0.711`，
而 common/query-floor 只有 `0.019`；Q3 common/history 为 `-0.00017`、common/query-floor
`-0.0015`。所以 Q2 的 late history→candidate-common 同向收敛不是由 query causal-invariance
误差带出的假象，并保持 model-specific。history/query-floor cosine 在两模型也接近零，history
RMS 却高于 query floor 约 `43--51x`。

该负控同时修正了 channel-profile 证据的权重：Q3 state 28 candidate-common 与 query-floor 的
channel-energy profile cosine 高达 `0.961`，但 signed direction cosine 仍接近零；同一组高增益
channels 可以承载完全不同方向。因此 channel participation/profile 只能描述容量使用，不能证明
语义 transport。后续机制判断优先采用 signed activation alignment 与因果 edge/branch effect，
channel profile 只作支持性背景。

这种变化还不是跨模型、跨偏好因子统一的“擦除”：Q2 candidate-readout brand
`real-minus-random` 从中区间 `0.4188` 降到晚区间 `0.0646`；Q3 brand 却从早区间
`0.2903` 增至晚区间 `0.4591`，同时其 category 从 `0.6378` 降到 `0.4009`。因此最终诊断必须
区分 model、preference factor 与 candidate-relative causal use；不能从单一 category 曲线推断
所有偏好消失，也不能据此提出无差别保留全部 history state 的控制。

把四个宽区间展开为 29-state 完整描述曲线后，变化也不是单调的逐层衰减。state 1--28 的
`real-minus-random` 相邻步中，Q2 brand/category 分别有 11/16 与 16/11 个上升/下降步，Q3
brand/category 分别有 12/14（另 1 个相等）与 12/15 个上升/下降步；相邻步方向切换达到
11--18 次。Q2 category 例如在 state 10 下降约 `0.1116`，state 11 又回升约 `0.1694`。
full-minus-null excess 曲线同样频繁换向。因而 D1 不支持从 probe 曲线指定一个确认性的“擦除
层”：偏好代理可能被反复重编码、暂时减弱后恢复，或受不同分支竞争。层责任仍必须由冻结的
D2 causal-margin adjacent transition 和独立 fold 确认，D1 的任何最大下降都只作全量描述。

位置负控进一步限制了“可解码即偏好”的说法。`query_end` 位于 history 之前；其
full-minus-null excess 在 Q2/Q3 × brand/category × 四区间 16 个点中的最大绝对值只有
`0.00461`，而 candidate-readout category excess 峰值为 Q2 `0.1650`、Q3 `0.2245`。
query_end 仍有较高 category `real-minus-random`，说明该 probe 也会捕获 query/population
相关性。因而架构上需要保护或分离的是 **history 新增到 candidate-relative state 的部分**，而非
任意位置上可解码的 category/brand 方向；query_end excess 近零同时构成 causal ordering 与
full/null 配对的内部负控。

embedding 边界也已经明确：Q2/Q3 的 candidate-readout `state 0` full-null
L2/sqrt(hidden) 都精确为 `0`，full/null RMS ratio 为 1。相同 candidate token 的静态 embedding
不携带 history-conditioned personalization；差异只在 Transformer 上下文化后出现。这个结果不
否认 history token embedding 的内容作用，但排除了把当前候选相对问题主要归因于静态 candidate
embedding table 的解释，也降低了只调整 item/candidate embedding 作为防止方案的优先级。

overall 与 strict-transfer 的同一 candidate-readout category full-minus-null 曲线也很接近：Q2
overall 四区间为 `0.0208/0.1603/0.1029/0.0285`，strict 为
`0.0254/0.1650/0.1328/0.0250`；Q3 overall 为
`0.0355/0.1999/0.1162/0.0775`，strict 为 `0.0331/0.2245/0.1454/0.0930`。
strict 通常不弱于 overall。因而 D1 的中层峰值和晚层衰减不是只在 strict-transfer 才出现，
也不像由 recurrence/overlap 混入 overall 后制造的平均假象。任何 surface-aware gate 的优先级
仍必须由 D2/D3 的因果 margin 与 recurrence/strict surface 结果决定，不能仅据表示曲线升级。

brand 的 history 新增成分更显示 factor/model 异质性：Q2 candidate-readout full-minus-null 四区间
为 `-0.0908/-0.0011/-0.1067/-0.1063`，Q3 为
`-0.1439/-0.0170/+0.1795/+0.0580`。Q3 的 brand probe 是中后层才被 history 明显增强，
Q2 没有同样转折。这个解码 contrast 仍不是因果使用证据，但它排除了把 category 的中层峰值
外推成统一 preference preservation objective 的依据。

Q3 native causal gate 的 block-13 margin effect 为 `+0.0028487`，block-27 为
`-0.0147846`。这建立 endpoint transition，但不提供 hidden-vector 夹角证据。

已完成 readout/gate 结果还能排除两个较窄的解释。Q2 在 `final_rmsnorm_input` 与
`final_rmsnorm_output` 的注册 patch 结果相同，因而该 full-to-null 转移的新增符号变化不能归因于
最后一次 RMSNorm；这不等于 RMSNorm 在所有干预下普遍无作用。Q3 的
`all_native_positions - first_position_only` margin 差约 `1e-8` 且未拒绝，说明首轮只 patch
第一个 native position 不是 b13/b27 端点差异的解释。Q3 完整 native readout 分解仍待 D6 正式
bundle，不能因此提前排除 lm-head/log-prob composition。

对 29 个 norm sites 的离线重放把 RMSNorm 边界扩展到所有 block。state 0--27 分别是各 block
input RMSNorm 的输入，state 28 是 final RMSNorm 输入；使用冻结 norm 权重在 FP16 activation
snapshot 上以 float64 重放。Q2 晚区间对 full-null common/residual delta 的 RMS gain 均值为
`0.548/0.569`，residual/common gain ratio 为 `1.014`，common energy fraction 只改变
`-0.00091`。fold 0/1 的 ratio 为 `1.0138/1.0142`，结果几乎相同。也就是说，Q2 input
RMSNorm 虽整体重标度 branch input，却没有相对压掉 candidate residual，反而极轻微偏向 residual。

Q3 构成反向控制：其晚区间 common/residual gain 为 `0.974/0.905`，ratio `0.943`，common
fraction 增加 `+0.01497`，比 Q2 更表现为 norm 后 common 相对占优；两 fold 增量为
`+0.0168/+0.0133`。因此“input RMSNorm 普遍压掉 residual，造成 Q2 common dominance”与
跨模型证据相反。input RMSNorm 在 pre-norm block 中只馈入 attention/MLP 分支，原 residual
bypass 不被它替换；这些 local gain 也不能沿层相乘成 residual-stream 擦除量。

Q2 final RMSNorm 单点确实更偏向 common：common/residual gain 为 `0.1009/0.0884`，ratio
`0.876`，common fraction 增加 `+0.00464`；但 residual pre/post cosine 仍为 `0.873`，不是方向
反转，而且前述注册 causal patch 已显示 final-norm input/output 对 endpoint effect 相同。Q3 final
RMSNorm 的 residual/common ratio 反而为 `1.037`，同时 total direction cosine 更低
（Q2/Q3 `0.906/0.739`），再次说明 norm 几何变化本身不是 Q2 失败的充分标志。所有 block 的
norm-weight RMS 随深度上升主要是 Qwen base 共有模式，Q3 LoRA adapter 不含任何 norm 参数；
Q2 的对应权重也几乎相同。RMSNorm 因而降为 branch-conditioning 次级因素，selected-node 因果
审计仍保留，但当前优先级低于 attention content 与 MLP composition。

Q2 final-state 的两个评价端点还显示“部分有用但目标错配”，不是所有排序信息都被抹去：
same-minus-null NDCG 为 `+0.01173`（95% CI `[0.00373,0.01925]`，BH q=`0.00540`），
target margin 却为 `-0.03137`（95% CI `[-0.04079,-0.02183]`，BH q=`0.00080`），且两 fold
分别在各自方向一致。晚层 history-conditioned state 因而能改善一部分 slate utility，同时把注册
target 相对最佳低增益竞争者推向错误方向。防止机制不能只做无条件 layerwise preservation；即使
中层状态被保住，也仍需要 candidate-wise signed residual、兼容性 gate 或可审计 abstention，确保
历史贡献的符号与具体候选匹配。

上述双端点是 population mean，不能误写成所有请求内的必然 trade-off。对已有
per-request bundle 作不新增显著性检验的全量描述：2,194 个 target-margin 有限的 strict 请求中，
NDCG/margin 符号 3×3 计数（NDCG 行 `positive/zero/negative`，margin 列
`positive/zero/negative`）为 `[[438,22,235],[379,50,467],[102,16,485]]`；两端点 Pearson
相关为 `+0.432`。NDCG 正且 margin 负只占 `10.7%`，两者都正占 `20.0%`，margin 负请求总计
`54.1%`。fold 0/1 的 NDCG正-margin负比例分别为 `11.2%/10.2%`。所以均值符号冲突主要反映
request heterogeneity 与效应幅度，而不是普遍的同请求 antagonism；这只是探索性描述，但使
request-level gate/abstention 比一个全局固定系数更值得在诊断后验证。

同时，Q2 same-minus-cross 的 NDCG 为 `+0.03149`、target margin 为 `+0.10222`，两者 BH
q 均为 `0.00080`。正确 request state 明显优于错误 donor state，却在上述 target margin 上仍劣于
null。故问题也不能简化成“用户特异性消失”：晚层保留了 request-specific information，但其相对
query-only/null 基线的候选贡献发生了符号或强度校准错误。一个未来 gate 必须同时满足
same > wrong-user 和 gated contribution >= null-oriented safety criterion；只优化前一个条件会保留
个性化身份，却不保证严格迁移。

两个条件的 per-request 联合分布证明它们不是同一件事。对 2,194 个 margin 有限的 strict 请求，
以 same-minus-null 为行、same-minus-wrong-user 为列的 `positive/zero/negative` 3×3 计数为
`[[594,15,310],[54,1,33],[473,15,699]]`。其中 `473` 个（`21.6%`）是
same > wrong-user 但 same < null 的 **specific-but-harmful** 请求，只有 `594` 个（`27.1%`）
同时通过 specificity 与 null-safety；两比例在 fold 0/1 分别为 `21.0%/22.2%` 与
`27.1%/27.0%`。NDCG 上对应比例为 `10.7%` 与 `21.1%`。这是探索性全量描述，不新增
推断结论，但明确规定未来 gate 的两个可独立失败条件：wrong-user discrimination 不能替代
relative-to-null usefulness/abstention。

specific-but-harmful 也不主要是用 NDCG 收益交换 margin：上述 473 个请求的 NDCG
same-minus-null 均值为 `-0.0740`，符号计数为 `91/165/217`（正/零/负），fold 0/1 均值为
`-0.0796/-0.0686`。只有 `19.2%` 在 margin 有害时仍有正 NDCG；相对地，594 个
margin-safe-specific 请求的 NDCG same-minus-null 均值为 `+0.1214`。因此多数
specific-but-harmful 请求在两个端点上都适合 abstain，约五分之一才构成真正多端点权衡。这个
探索性分组不能作为训练标签或新评价 slice，但说明双 gate 不必天然牺牲总体 utility，并要求
最终机会报告单独列出剩余 trade-off 子群而非宣称普遍可解。

Q2 score decomposition 对 common-offset 解释还有一个必要边界：注册分解
`score_ij = common_i + candidate_relative_ij` 的最大重组误差为 `0`。same-minus-null 的平均
common shift 虽为 `0.53898`，但它在同一 request 的排序和 target-versus-competitor margin 中严格
抵消，不能直接造成有害端点；真正改变排名的是 candidate-relative shift（RMS `0.19171`）。
显式分离 common path 的价值在于审计、容量约束与训练信号归因，单纯在现有 score 上减去 common
不会改变任何排名，不能被包装成修复。需要约束的是 candidate-relative residual 的符号与校准。

对同一封口 bundle 作不新增推断 family 的 direction-vs-scale 描述：2,194 个 strict 请求的
candidate-relative full-null shift RMS 与 signed target-margin effect 的 Pearson 相关只有 `-0.058`
（fold 0/1 为 `-0.111/-0.006`），但与 `|margin effect|` 的相关为 `+0.558`（fold 0/1 为
`+0.602/+0.519`）。common shift 与 signed margin 的相关为 `+0.0016`。因此 scale 较稳定地决定
效应“有多大”，却几乎不决定方向好坏；统一缩小 residual 会同时削弱正负请求和已观察到的
NDCG 增益。正式 direction/scale 因果结论仍等待 D2 factorial，当前结果只支持把 scale-only
calibration 放在 signed compatibility gate 之后，而不是用它代替方向判断。

safe-specific 与 specific-but-harmful 的简单 readout diagnostics 也没有显示一个明显 norm 阈值：
两组 candidate-relative RMS 均值为 `0.1507/0.1311`（safe 反而更大；标准化均值差约 `0.26`），
candidate count、full input norm、full output norm 与 input-output cosine 的标准化均值差绝对值均
小于 `0.18`，且两 fold 的组均值方向稳定。边际差异小不能证明这些特征在联合模型中无预测力，
但它否定了“历史位移过大即有害”的简单阈值故事；usefulness gate 更应基于有符号的
query-history-candidate content compatibility，并接受严格 held-out 验证，而非按当前 dev outcome
选择 norm cutoff。

训练侧的已完成结果也不支持“优化过程中把整体方向翻过去”的简单解释。Q2 的每请求
RankNet/ListNet gradient cosine 在 base/final、recurrence/strict-transfer/other-overlap 六个单元中均
为强正值（约 `0.8920--0.9840`），没有达到负向冲突 SESOI。Q3 的 56 条 q/v LoRA 路径在
step 500 到 frozen final 的 function-space `delta_w_cosine` 均为正，均值 `0.9505`、最小值
`0.8802`。这些结果不排除局部参数/样本效应，但把主搜索方向进一步推向 inference-time
attention/MLP/residual/readout composition，而不是全局 loss conflict 或 LoRA trajectory reversal。

LoRA 容量侧同样没有出现字面意义上的秩或范数塌缩：step 500 与 frozen final 的 56 条
q/v 路径在注册阈值下均保持有效秩 8，final 相对 step 500 的更新范数平均还增长约
`11.4%`（q）和 `9.7%`（v）。final 的能量确实集中在较少方向（熵秩均值约
`4.56/4.20`），且 mid-late/late 的最小/最大奇异值比更低，但这只能描述为后层谱更
各向异性，不能证明容量瓶颈或信号擦除。因此当前证据不支持全局增加 LoRA rank；只有
组件级因果证据指向特定后层后，才值得检验 block-local 的方向分配或容量保护。

Q2 的训练目标也没有在粗粒度参数族上隐藏明显的资源分配分歧。对 base/final × 三个 surface ×
observed/label-shuffle 共 12 个完整单元的 96-request 梯度能量份额作描述性审计，observed 单元中
RankNet 与 ListNet 的参数族平均份额差最大绝对值只有 `0.00303`，每请求参数族份额 total
variation 的单元均值最大为 `0.01287`，没有任何参数族达到预先保留的 `0.05` 描述带。训练前后
最多约 `0.0345` 的参数族重分配也在两个目标中近乎同步。这个结果只排除“两个目标把总体梯度
能量系统性投向不同 Q/K/V/O、MLP、RMSNorm 或 embedding/readout 族”的简单故事；份额没有
族内 dot product，仍不能排除特定层/参数内的有符号冲突，后者继续等待 exact step-501
optimizer replay 与组件因果结果。

Q2 frozen checkpoint 相对声明的 BF16 base 的 596,049,920 参数全量差分，也没有显示晚层或单一
大组件异常吞噬训练更新。四个等参数量 Transformer 区间的 update-energy share 为
`29.2%/24.3%/20.8%/25.7%`，逐层 update RMS 的 coefficient of variation 仅 `7.35%`，最大/最小
为 `1.33x`；Q/K/V/O 与三支 MLP 的 per-parameter update RMS 都在约
`1.61e-4--1.72e-4`。MLP 三支合计能量较高主要随参数量增长，tied embedding/readout 的 update
RMS 反而只有 `5.18e-5`。晚区间 input RMSNorm 的单参数 update RMS 为 `8.48e-4`，但其
Transformer update-energy share 只有 `0.043%`，只能作为 D2 norm-node 的候选线索。逐层参数
update RMS 与 common-energy change/common share 的 Pearson 也仅为 `0.25--0.42`。因此当前不支持
“晚层被全参数训练得特别狠”或“某个大参数族容量分配失衡”的简单训练侧解释；BF16 base 还限制
了量化步长以下的解释，最终仍以 optimizer replay 和干预结果为准。

总量均匀不等于通道分配均匀。对相同 Q2 参数差分按 attention head、SwiGLU intermediate channel
和 norm channel 分组后，attention-head update 仍较分散：归一化 participation ratio 从早区间
`0.964` 仅降到晚区间 `0.914`。MLP intermediate update 则明显更各向异性，participation ratio 从
`0.945` 降到 `0.749`，top-10% channel energy share 从 `0.146` 升到 `0.228`，max/mean energy
ratio 从 `3.16` 升到 `8.01`；gate/up/down 三支的晚/早 participation ratio 均约
`0.77--0.81`。attention-head participation 与 block common-update share 的逐层 Pearson 为
`-0.476`，但只是 post-hoc 几何共现。部分 norm tensor 更集中，却只占极小更新能量并受 BF16
差分边界影响。这个线索把 D4 MLP channel/group 干预置于高优先级，但 Q3 后层 LoRA 谱也更
各向异性，所以不能把 update concentration 本身写成 Q2 因果机制或容量塌缩。

同口径的 Q3 LoRA function-space head audit 进一步构成反例。Q3 final q-head participation ratio 从
早区间 `0.892` 降到晚区间 `0.750`，比 Q2 full-parameter q-head 的晚区间 `0.910` 更集中；v-head
则保持分散（`0.980 -> 0.956`）。q-head 晚层集中在 step 500 已经存在（`0.784`），final-minus-
step500 increment 也为 `0.782`，不是训练末段突然发生的 head collapse。更关键的是，Q3 final
q-head participation 与 common-update share 的 Pearson 为 `+0.395`，Q2 attention-head pooled
participation 对应为 `-0.476`；相似的深度集中趋势伴随相反的 common/residual transport 结构。
因此 attention-head update anisotropy 不是足以区分机制的 biomarker，必须由 D3 edge/head 因果
结果与 D4 MLP 结果共同解释。

D5 的全量 qrels-blind token-position 审计还收紧了“距离压缩”的含义。冻结 content-neutral
eligibility 的 7,254 个请求、每模型 146,112 个候选中，实际保留 history span 的 token 数中位数为
`333`（均值 `296.68`）；history 到 native readout 的中心相对距离中位数仅为 Q2 `248.95`、
Q3 `240.45`。因此注册的 `d' = d - H` 不是温和位置微调：相对相位在 Q2/Q3 平均分别有
`65.75%/68.56%` 的 history-key/readout-query pair 跨过零点，压缩后的 signed center distance
均值为 `-65.33/-73.83`。这只是 layer-local RoPE phase coordinate 的符号变化；token index、
causal mask 和注意力先后关系都没有改变，不能称为 hidden preference vector reversal 或因果顺序
反转。

过零也不等于绝对距离普遍变大。按 `|d-H| < |d|` 逐 edge 计算，Q2/Q3 仍有
`79.20%/76.89%` 的 pair 在绝对相位距离上更近，`20.67%/22.98%` 反而更远；平均绝对距离从
自然值 `231.35/222.85` 降为 `95.52/98.47`。两个固定 query fold 的过零比例和绝对变远比例差异
都小于约 `0.8` 个百分点。按 Qwen3 `head_dim=128, rope_theta=1e6` 对各频带作各向同性参考，
一个实际 H 的相位旋转期望 cosine 均值仅 `0.544`、相对 L2 约 `0.954`，说明 D5 是强相位重排，
不是小扰动。Q3 因 prompt readout 比 Q2 固定近 `8.5` token，过零与绝对变远略多，但两模型 H
完全相同。

所以正式 D5 仍按冻结的 compression-minus-equal-expansion family 运行；结果只能回答“对称的
`±H` 强相位重排是否产生模型/层特异的排名效应”。即使 compression 胜过 expansion，也不能
外推为自然地缩短 position IDs 就会改善推荐，更不能单独证明原失败由 RoPE 距离造成。只有 active
NDCG 越过预注册等价带、compression 与等幅 expansion 分离，并与 D3/D4/D6 证据一致时，才允许
把 position explanation 标为 supported。

冻结 final RMSNorm 与 tied yes-minus-no unembedding 后，对同一 482-request、20,357-candidate
qrels-blind 锚点作全 29 状态 logit lens，进一步把 activation geometry 接到了模型原生读出方向。
Q2 state 28 的 full-null score delta RMS 可分为 common `0.77125` 与 candidate-relative residual
`0.24397`，common energy fraction 为 `0.90904`；fold 0/1 分别为 `0.90793/0.91026`。Q3 的
首 answer-token 对照为 common `0.10125`、residual `0.14657`、common fraction `0.32307`。
Q3 数字不是完整 two-token teacher-forced likelihood，只用于模型对照；中间层把 final norm/readout
移到非原生深度，也只能作描述性轨迹，不能据此选择层。

这项读出审计不支持“候选相对信号在最后被彻底弄没”。Q2 state 28 的 full/null
candidate-relative score RMS 为 `0.32492/0.20903`（比值 `1.554`），residual 在 native direction
上的能量相对各向同性参考仍为 `17.91x`；它仍然存在且能进入最终读出。更精确的表述是：历史影响
主要被组织成不改变 request 内候选顺序的 common 路径，而剩余 candidate-relative contribution
具有足够幅度却缺少正确的 signed usefulness。其 raw candidate-relative RMS 从 state 18 的
`1.31159` 衰减到 state 28 的 `0.32492`，但同期 null 也从 `0.76645` 衰减到 `0.20903`；这是晚层
压缩/重校准，不是消失。结合已封口的有害 ranking/margin 端点，问题更像“方向选择性和组合错误”
而非“readout 看不见任何历史”。

逐层 lens 的一个非选择性线索是 Q2 common energy fraction 从 state 25 的 `0.76516` 在 state 26
升到 `0.91195`，之后保持约 `0.91`；Q3 同区间反而从 `0.49255` 降到 `0.40767`。这使 block 25
附近的 attention/MLP composition 值得由既定 D2/D3/D4 family 检验，但不能用这个 post-hoc 跃迁
新增 block、head 或 group。两次确定性运行在删除仅含 output-dir 的 command 字段后规范化 SHA256
均为 `25da5c58aff18092b3ac87de73391db3b8edef3796ae0850ccae9d1d117a509a`。

embedding/readout 底层也没有出现字面反转。对 Q2 base/final 的全部 151,936 个 tied
embedding/readout rows 作 BF16 精确差分，lowercase yes-minus-no direction 的 base/final cosine 为
`0.9999516`，方向更新仅为 base direction norm 的 `0.993%`；common yes/no direction cosine 为
`0.9999998`。yes/no 两行的 update RMS 处于全词表约 `97.1%/97.3%` 分位，说明它们作为直接
监督的输出行确实比多数词表行更新更多，但两行差分没有翻向。Q3 final adapter 的 112 个参数对象
全部是 28 层 q/v LoRA A/B，embedding/lm-head 参数对象为 0，故 Q3 的底层词表/readout 保持 base。

固定 512 candidate rows（482 requests）的角色 token 汇总也不支持 history embedding 单独塌缩：
Q2 history/candidate/query occurrence-weighted row update RMS 分别为
`1.879e-4/1.907e-4/2.042e-4`，base-final row cosine 分别为
`0.9999788/0.9999783/0.9999756`。全词表 top-10% rows 承担 `99.03%` update energy、normalized
participation ratio 仅 `0.0800`，主要表明实际暴露 token 与仅受微小 weight decay 的大量词表行分开，
但角色内 history 并未比 candidate/query 呈现独有的范数或方向异常。由于 Q2 input/output rows tied，
这仍是描述性几何，不能把 input embedding 与 output readout 的因果作用分离；结合 native readout
和全层结果，只能把“底层词向量或 yes/no 行先反转”降级，而不能替代 D2--D4 组件干预。

训练目标本身给出了一个更基础的解释边界。对冻结训练选择中的 288 个 request groups（recurrence、
strict transfer、other 各 96），以及逐组确定性的 label-shuffle control，共 576 个 score-space
Hessian 检查，实际 RankNet、ListNet 和二者各半组合在每一个 group 都恰好只有一个零特征值；
该零空间就是全一向量。把同一 request 内所有 candidate score 同时加 `137` 后，所有目标的 loss
最大变化不超过 `2.1e-14`，gradient sum、`H @ 1` 和 common-direction Rayleigh quotient 也都只在
浮点误差范围；与此相对，candidate-relative 子空间保持完整的 `n-1` 个正曲率方向。两次独立运行的
JSON SHA256 均为 `0d9e5e1cdf9c99119aa1b8c4847c425ad356fbd44a9b21cce8bd22df726d6535`。

因此，Q2 中增长很快的 candidate-common history response 确实没有受到排序损失的直接约束；但
common shift 本身既不改变 request 内排序，也不改变 margin，不能被称为性能下降的直接原因，简单
减去均值也不会修复端点。它更像容量占用、残差组合或校准漂移的标记。真正改变排名的路径仍是
candidate-relative contribution 的 signed direction/usefulness；D2--D4 需要判断晚层究竟在哪个
attention/MLP/residual composition 节点压缩或重组了这部分。这个结论也排除了把“共同分量很大”
直接包装成方法动机的做法。

## 3. 已注册实验如何区分机制

- D2 blocks 13--27 fold-0 只选择均值最负的相邻步 `j-1 -> j`，fold-1 独立确认；这定位注册范围内
  最强的相邻下降候选，不等于最早开始损失的层或全局唯一损失层，也不按结果追加层。
- D2 selected block 的七节点按 block input、input RMSNorm、attention output、
  post-attention residual、post-attention RMSNorm、MLP output、block output 排列。全部相邻
  sufficiency变化用于描述block内attenuation profile，但不以“首次负转折”单独决定primary组件。
- donor-direction/recipient-RMS、recipient-direction/donor-RMS 和随机方向 controls 分开检验
  direction 与 scale。只有方向证据通过，才允许升级为 vector-direction explanation。
- D3 history-to-readout mask、value-edge zero 与 neutral K/V 检查 attention transport 是否使
  history signal 消失；D4 检查 SwiGLU/MLP 覆写；D5 检查 RoPE 距离而非内容丢失；D6 检查
  native readout 是否丢弃仍存在的中间状态。

D3 per-head observation 的首个 Q3 b13 v1 正式 run 在固定样本第 135 行触发机械门并停止：native
score identity 仍为 `1.49e-8`，但 manual attention reconstruction 为冻结低精度界的 `1.0667x`。
逐路径复现确认原因是观测器把 BF16 Q/K 点积在 softmax 前再次量化为 BF16，而冻结 native SDPA
对 BF16 输入采用 FP32 accumulation；这不是 attention 科学效应。没有放宽 `4*eps` 门，修复为
QK、softmax 和 probability-weighted V 全程 FP32，并以直接 selected-SDPA 作独立交叉核对。覆盖
原失败点的 136-row smoke 最大比值为 `0.1388x`；新实现 digest
`40bffa52cca427d6756a3e972e1ecc7388e85729cca2cd7d12ded00e829de20d` 下，Q3/Q2 b13 v2 的
512/512 rows 均完整，最大比值分别为 `0.1435x/0.1486x`，score identity 分别为
`2.98e-8/0`。失败 v1 原样保留为 mechanical non-result；v2 从第 0 行重跑，未读取 qrels/test。
最终 head evaluator 另行强制六个 Q2/Q3 × b13/b20/b27 bundles 使用同一非空 implementation
digest，防止修复前后观测被静默混合。

D2 的选择链也在 qrels 打开前增加了同级实现血缘门。每个模型、每个 fold 的 15 个 post-block
bundles 必须共享一个非空 implementation digest，且 metadata 必须与各自的 immutable run contract
一致；fold-1 的 digest 必须与冻结 fold-0 selection 完全一致。selection、confirmation 和随后只暴露
selected block 的最小 branch contract 都携带该 digest，七节点 scorer 的 metadata 又单独绑定自己的
implementation digest。最终 selected-branch evaluator 会重新散列 branch contract、fold-0 selection
和 fold-1 confirmation，并在读取 fold-1 qrels 前核对 method、checkpoint、selected block 与两级
实现血缘。这保证“在哪一层、哪个组件发生作用”的证据不能跨 scorer 版本拼接。完整测试现为
569 passed、7 subtests passed；冻结 plan/manifest SHA256 未改变。

对剩余单 bundle 评估路径的横向审计还发现并补上两处尚未触发的完整性缺口：D6 Q1 trajectory
现在要求最终 `index.json` 与 metadata 中的 SHA256 一致、index 与 metadata 共享同一 run-contract
SHA256，并把非空 scorer implementation digest 与 run contract 绑定；D6 Q3 native readout 现在会在
打开 qrels 前重新散列 `scores.jsonl`，并作同样的 implementation/run-contract 绑定。两项修改只在
evaluator 侧，未改变正在运行的 scorer、冻结 bundle 或科学 family。

同一规则现已提升到 deep-dive closeout 总审计，而不再只依赖各 evaluator 的局部实现。对每个
`status=completed && result_eligible=true` 的正式 run，closeout 会重算 canonical run-contract
SHA256，核对 implementation identity 与 contract 的 digest，并重新散列已声明的 `scores.jsonl`、
`rows.jsonl`、`observations.jsonl`、`groups.jsonl`、`replays.jsonl` 或 `index.json`。现有 34 个正式
completed runs 全部通过该新门；19 项交付仍为 5 completed、14 pending、0 failures。多 bundle
evaluator 还会在此基础上再次检查跨 block/模型实现一致性，形成 run、bundle、evaluator、closeout
四层独立血缘链。

最终报告也已增加 fail-closed 合同，防止长队列结束后只摘取“看起来有意思”的组件。报告必须逐项
覆盖 `serialization/tokenization`、`token embedding/position`、attention Q/K/V、attention output、
MLP、residual/norm、layerwise representation、history routing、candidate-conditioned interaction、
readout/calibration、loss/gradient、optimizer/scheduler 共 12 个组件；每项只能标为 supported、
weakened、unresolved、untested 或已绑定的 mechanical failure，并引用 19 项注册交付物之一。H0--H5
必须全部有结论边界，五个预注册架构机会必须恰好排序一次，且必须明确保留“不打开 source test、
不把诊断控制升级为方法、异质性不外推”的边界。合同会先调用 closeout：在 19/19 未完成时直接拒绝
产生科学报告，完成后也只接受 closeout 已核验的交付物与逐字节绑定的失败记录。
报告合同还冻结了 component/H0--H5/opportunity 到允许引用 deliverable 的语义映射；即使 19 项均已
admit，也不能用 optimizer replay 填充 tokenization 结论、用 RoPE 结果替代 loss-gradient 证据，
或以其他语义不相关的完成项凑齐矩阵。

D4 b13 的首轮 MLP group queue 也在结果读取前发现一个独立机械门问题。Q2 v1 已完整写出
512 行，但 finalize 报 permutation recomposition ratio `1.43884x`；最大 FP32 重组误差实际上只有
`1.19209e-6`。根因不是 SwiGLU 科学效应，而是控制代码把原生 BF16 product/weight 转成 FP32 做
更精确的代数重组后，又错误地以 FP32 epsilon 作为冻结 `4*eps(dtype)` 的 dtype。冻结计划明确该
界针对原生 BF16/FP16 tensor；按 BF16 epsilon，同一最大误差在 unit-scale 最保守上界只占
`3.8147e-5x`。Q3 v1 偶然以 `0.79160x` 落在错误的 FP32 界内，也不能与修复后实现混用。

修复没有改变 `4*eps(dtype)*max(1,max_abs(reference))` 公式：仍以 FP32 重组降低测量误差，但 bound
明确绑定 native SwiGLU product dtype，并在每行记录 `recomposition_dtype=float32` 与
`bound_reference_dtype=bfloat16`；runtime 还会把 finalize 异常正式写为 mechanical failure。新实现
digest 为 `0fbf6d77eddebdde602ec6a8af250cc05ebe61a94fd9ac99e8ded242368b2895`。Q2/Q3 单行
v2 smoke 的 permutation ratios 分别为 `1.30384e-5/2.87112e-6`，same-group score identity 都为
0。Q2 v1 partial 与 Q3 v1 completed bundle 原样保留但均不进入综合；Q2/Q3 b13 以 v2 从第 0 行
全量重跑，最终 evaluator watcher 只接受 b13 v2 和同一新 digest 的 b20/b27 bundles。

Q3 b13 v2 随后已完成 512/512 行并通过 finalize：same-group identity、原生 BF16 参考界下的
FP32 permutation recomposition、有限覆盖与 run-contract/implementation 绑定全部通过，`rows.jsonl`
SHA256 为 `6f5e6e4e4ae917900cfce7d5ce07f3313b389b52e5c52ae5dba83200fe3d3606`。closeout 因此将
完整性复核的正式 completed runs 增至 34，仍为 5/19 交付、14 pending、0 failures，并继续只承认
那一条逐字节绑定的 Q2 v1 mechanical failure record。GPU 队列已按固定顺序切换到 Q2 b13 v2；
在六个 b13/b20/b27 bundles 全部完成前不读取或综合任何 MLP group 科学效应。

Q2 b13 v2 也已完成 512/512 行并通过相同 finalize 与血缘门；实际 `rows.jsonl` SHA256
`a5e7ea62935ab378bd76d5a837042424b5a7506c11cef43551d0e2b9e31dae59` 与 metadata 声明一致，
implementation digest 与 Q3 完全相同。closeout 的正式 completed-run 完整性计数增至 35，交付与
失败状态不变；GPU 队列已继续到 Q3 b13 GQA group。b13 MLP 的跨模型成对完成只解除机械阻塞，
不解除六 bundle family 的结果封口。

Q3 post-block fold-0 block 20 已完成 4,082/4,082 请求，完整有限 score coverage、实际
`scores.jsonl` SHA256 `f17332e1e1ab97c94af9a8e12c05d13c09e2571820bf61735f24ce2985db70d2`、
implementation/run-contract、qrels-blind 与 source-test-closed 门全部通过。closeout 正式完整性计数
增至 36；even-block lane 已按冻结序列切换到 block 22。全 15-block fold-0 family 完成前仍不读取
层效应或生成选择结果。

Q3 fold-0 block 19 随后也完成 4,082/4,082 请求并通过同一完整性门；`scores.jsonl` SHA256 为
`9b7ae2ab3a1561bbaca13e4e09a07fd68ee96ff183e5f0a2f069f69359ef15d3`，implementation digest 与
block 20 一致。closeout 计数增至 37，odd-block lane 已切换到 block 21；当前 Q3 blocks 21/22
继续并行，选择器仍保持关闭。

Q2 fold-0 block 22 已完成 4,082/4,082 请求并通过相同完整性门；实际 `scores.jsonl` SHA256
`309689ecaa08d7384c356a1e0996f0a54987ce60bf7874d5e3fa37da72c655a2` 与 metadata 一致，
post-block implementation digest 仍为统一的
`c6bf5664de2e69318159acbf0e18cc296220542dde88ae4f949e4a41d3e081c4`。closeout 计数增至 38，
Q2 lane 已按冻结顺序切入 block 23；fold-0 选择仍未开启。

Q3 b13 GQA causal-localization 已完成 512/512 行；8 个 group、每 group 2 个 query heads、五个固定
condition 的有限覆盖和 identity 均通过，实际 `groups.jsonl` SHA256
`2e2a46f1d17d2597178e9a7a9ef7c2d0ce1b21d8ce6e8564946c54484c8a65f6` 与 metadata 一致，
implementation digest 为 `49147c5447cfb1e6bd9d6b355be97fa07d8fa71f6f26a8291035b0d7acf6da3d`。
closeout 正式完整性计数增至 39；GPU 队列已切换到 Q2 b13 GQA。六个 Q2/Q3 × b13/b20/b27
bundle 全部完成前不读取 group effect，也不 outcome-select group。

Q2 fold-0 block 23 已完成 4,082/4,082 请求并通过完整性门；实际 `scores.jsonl` SHA256
`e8b9c3d5cfaf31e8a3cdcc832fea132c018e23af47268686049d935dc7ec17d9` 与 metadata 一致，
implementation digest 保持统一。closeout 正式完整性计数增至 40，Q2 lane 已按冻结顺序切入
block 24，fold-0 仍剩 blocks 24--27。

Q2 b13 GQA 随后也完成 512/512 行；8-group coverage、identity（maximum delta精确为 0）、
implementation/run-contract 和数据边界全部通过，实际 `groups.jsonl` SHA256
`5b6668413284cec342d03d242223b8d19092a52e3b34615fb0ef483f4a9ddbf8` 与 metadata 一致，且
implementation digest 与 Q3 b13 完全相同。closeout 正式完整性计数增至 41；GPU 队列已切入
Q3 b13 RoPE，GQA 仍等待 b20/b27 四个固定 breadth bundles 后再整体评估。

Q2 fold-0 block 24、Q3 fold-0 blocks 21/22 随后均完成 4,082/4,082 请求；三者的完整有限
coverage、identity、统一 post-block implementation digest、qrels-blind 与 source-test-closed 门均
通过。实际 `scores.jsonl` SHA256 依次为
`2162dbecb4294cde7e162509b32a72642c132553e8d43453f3139065c4350f77`、
`cb65dec3e32c1b8ba2977eb62ddcbf8fb2d118aedfa75495404440e365383281` 和
`3c93e83c28a820917c0e512169494a4f39ffb0a4d18c7da42c72bdb637de5119`。closeout 正式完整性计数
增至 44，仍为 5/19 交付、14 pending、0 audit failures；Q2 已切入 block 25，Q3 两条 lane 已继续
blocks 23/24。全 fold-0 family 完成前仍不读取层效应或启动选择。

首个 Q3 b13 RoPE formal v1 在第 23 个 request 前触发机械门并停止。zero-phase score identity 为
精确 `0`，但 common `+17` phase 的最大 native-score delta 为 `0.0625013`；旧实现错误地把这一
score 差交给仅允许用于 RoPE norm/代数审计的 `4*eps(dtype)` 界，且仍达到 `1.999995x`。这不是
RoPE 科学效应，未读取 partial 科学值、qrels 或 source test。v1 metadata/progress/partial scores 已
逐字节绑定在 SHA256 为 `86c7916256c2e9b520c43fe5121e99142029449dafcedf06c4e4f495dfb0210f`
的 mechanical failure record 中并排除出综合。

修复未放宽 `1e-5` score identity，也未改变六个 compression/expansion 科学条件：common `+17`
现在只在全部 Q/K rows 上以 FP32 执行成对旋转几何审计，随后把未改动的 native Q/K 委托给冻结
backend；RoPE norm 仍按 native dtype 的冻结 `4*eps` 界审计。新 implementation digest 为
`fce73e1241cc67533bd4547a7e061f33c6df199a192e0c6ae2e17433f43db469`，相关 targeted tests 为
27/27 通过。replacement 必须使用新 run ID 从第 0 行重跑，并先覆盖原失败 request 边界。

Q3/Q2 的 32-request 覆盖性 smoke 已在同一新 digest 下成对完成，均跨过原 Q3 v1 失败边界，
461/461 candidate rows 完整。两模型的 zero-phase 与 common-offset score identity delta 均为精确
`0`，RoPE norm algebra 最大 bound ratio 分别仅为 `6.9371e-6/5.6821e-6`；实际 score SHA256
分别为 `4950625167a5a28f55b5dbce4913d1e8092663cc0bfc4bb2e1091eaf28a432ad` 和
`d7d34bf921f1fea0713be7f3a96b03cdf92cd15d21477d44251deb527303fc11`。全套测试更新为
570 passed、7 subtests passed；`git diff --check`、两项 frozen plan/manifest SHA 与 closeout audit 均
通过。GPU2 已从第 0 行启动 Q3 b13 v2，完成后按固定队列启动 Q2 b13 v2；新 evaluator watcher
只接受 b13 v2 与相同 implementation digest 的 b20/b27 bundles。

对 19 项 closeout 的当前 producer graph 又作了一次 qrels-blind 静态/进程审计：14 个 pending
deliverable 均有已启动的唯一 synthesis/evaluator watcher 或其上游固定队列，路径与 closeout 合同
逐项一致。attention-head、MLP-group 与 RoPE 三条修复后的 family 均明确使用 b13 v2 加尚未运行的
b20/b27 v1 路径；后者启动时读取同一修复后 source digest，不会混用失败实现。D2 synthesis 在模型
因 gate 停止时仍按预注册 planned family 写入 missing cell、`p=1`，不会缩小 family 或等待不存在的
selected-branch score。当前没有孤立 pending deliverable、重复 evaluator 或错误 v1/v2 依赖。

最终报告合同进一步补上逐模型 `primary_loss_attribution` 硬门：Q2/Q3 必须各且仅各有一行，
`primary_component` 只能在 attention output、MLP、attention+MLP mixed、residual composition、
residual/norm interaction 或 unresolved 中选择。任何非 unresolved 结论都必须同时引用 D2 post-block
与 D2 selected-branch 两项因果交付、复现 fold-1 转折并标为 registered-confirmatory；attention、MLP、
mixed、post-block-only composition 与 residual/norm 各有互斥 flag 真值表。head/GQA/MLP-group 的
描述性定位被结构化固定为不能单独建立 primary cause，gate-stopped 也不能伪称 fold-1 已复现。
Markdown 会单列这一逐模型表，而不是把答案藏在自由文本 narrative 中。新增合同/渲染测试后全套为
575 passed、7 subtests passed，运行中的 scorer digest 未受影响。

closeout 级验证又将这些 flags 与真实 D2 bytes 自动交叉核对，而不只检查人工逻辑：
`fold1_transition_reproduced` 与 `postblock_registered_support` 必须逐模型等于 post-block synthesis 的
`localization.resolved`；attention 与 MLP flags 必须分别由 `attention_o_projection`、
`mlp_down_projection` 在 primary target-margin 上同时通过 same、cross stress、wrong-history
specificity、norm、direction、random-direction 六类注册门。residual composition 至少要有
post-attention/block-output residual 节点的完整注册支持，residual/norm 至少要有一个 RMSNorm 节点
完整通过；否则报告生成直接失败。这样无法把格式正确但与 metrics 不符的人工 flags 写入最终报告。
新增 bytes-to-decision 测试后全套为 576 passed、7 subtests passed。

component evidence matrix 的 `supported` 状态也增加了组件特异因果门，避免它与 primary attribution
表互相矛盾。例如 MLP 只有 D4 group 描述性定位时不能标为 supported，必须引用 D2 full-branch
因果交付；attention output 需要 D2 branch 或 D3 aggregate edge，position 需要 D5 RoPE，history
routing 需要 branch/aggregate-edge/context 中至少一项因果证据。冻结计划明确 optimizer/LoRA replay
没有注册 SESOI、只能 descriptive/unresolved，因此 `optimizer_scheduler=supported` 被合同直接禁止。
新增反向测试后全套为 578 passed、7 subtests passed。

architecture opportunity ranking 现在要求恰好一个 `primary` 且必须为 rank 1，并对 primary 增加
机会特异的确认性证据组：router 需要 D2 branch/D3 aggregate edge/D5 context 中至少一项因果证据；
H2 factor bottleneck 需要 representation 或 Q0/Q1 跨模型边界证据；H3 signed residual 需要 D2
post-block/branch 或 native readout；H4 gradient budget 只能由注册的 Q2 objective family 支持。
联合 H2+H3 路径必须同时引用 H2 侧和 H3 侧交付，不能只凭一侧描述性结果成为统一 primary。
新增单/多 primary、描述性 router 与联合路径缺边反向测试后，全套为 581 passed、7 subtests passed。

H0--H4 的 `supported` 状态现在也必须满足冻结的“双源”要求，而不是只列一个相关文件：H1 需要
branch/aggregate-edge 因果证据加独立 head/group/context 源；H2 需要表示证据加 Q0/Q1 跨模型边界；
H3 需要 D2 post-block/branch 加独立 native readout；H4 需要注册 Q2 objective family 加 context/
LoRA/optimizer 独立诊断；H0 需要表示加 trajectory/readout。H5 在本阶段没有独立第二 seed，因此
不能标为 supported，只能根据现有跨模型/人口证据保持 weakened、rejected 或 unresolved 的相应
边界。新增 H3 缺独立源与 H5 越界反向测试后，全套为 583 passed、7 subtests passed。

进一步按 H5 的原始反证条件收紧：没有独立第二 seed 时也不能把 H5 标为 rejected；跨模型、fold 和
请求簇一致性只能将其 weakened，不能排除 seed instability。因此本阶段 H5 合法终态只剩 weakened
或 unresolved。对应 supported/rejected 双向越界测试均通过，测试总数不变。

最终 architecture opportunity 不再只允许填写一句自由文本。五个候选方向都必须形成结构化 design
card，逐项给出 innovation claim、training signal、training data requirements、exact-null recovery
invariant、required modules、至少三项 critical ablations，并分别说明相对 CoPPS、BATA、HMPPS、
MemRerank 的 load-bearing difference。每张 card 还必须显式标记为
`design_opportunity_only_not_implemented`，防止机制诊断越界成为未经实验的方法声明。合同与 Markdown
渲染的正反向测试均已补齐；全套当前为 585 passed、7 subtests passed，运行中的 scorer digest 未受
影响。

冻结计划 W2c/W2d 的 selected-branch A/B 并行在原执行脚本中尚未真正物化；正式 branch scorer
启动前已补为 qrels-blind 的确定性 request shard。fold-1 冻结顺序按 `ordinal mod 2` 分到两个独立
run directory，shard 本身 `result_eligible=false`；合并器逐项复核实现 digest、run contract、冻结
输入哈希、完整有限覆盖、互斥性和全集覆盖后，才发布 evaluator 兼容的唯一正式 bundle。两个 shard
仍完整覆盖七节点、58 conditions，科学 family 和 frozen plan/manifest 均未改变。已有旧队列仅暂停
父 shell，当前 Python scorer 在完整落盘后由 watcher 无损交接到新队列；Q2 的第二 shard 另受 RoPE
b13 recovery 完成门约束，避免 GPU2 重叠写。端到端微型 merge、runtime、evaluator、closeout 与全套
测试通过；全套更新为 587 passed、7 subtests passed。

第一诊断阶段已有的五张 architecture opportunity card 又被迁成结果无关的固定 catalog，内容哈希为
`91f49f1b82ba04877cd9c4bd3192d57324a4113e5ba0a8bf2c31a524b764d29e`。创新声明、训练信号、
train-only 数据要求、exact-null/no-intervention recovery invariant、所需模块、关键消融以及相对
CoPPS/BATA/HMPPS/MemRerank 的 load-bearing difference 均在剩余 D2--D7 结果前固定。最终 closeout
只能改变 rank、status、evidence 与 falsification gate；若改写设计内容，合同直接失败。JSON 与
Markdown 最终报告都会写入 catalog hash 和 `design_opportunity_only_not_implemented` 边界。新增后验
改写反向测试及报告哈希测试后，全套先达到 588 passed、7 subtests passed。随后又补上跨表逻辑门：
rank-1 primary opportunity 的关联 H1--H4 必须是 supported/weakened，不能把 unresolved/rejected
hypothesis 重新包装成主方向；任一关联 hypothesis 被 rejected，其单项或联合 opportunity 必须同步
rejected。逐模型 resolved primary loss attribution 也必须在 12 组件矩阵中具有对应的
causal-supported 行，mixed attention+MLP 要求两行同时成立。新增反向测试后全套为 590 passed、
7 subtests passed；closeout 仍为 5/19 complete、14 pending、0 audit failures。

最终报告还增加自动 execution census，从 closeout 的全部 run declarations 计算 completed/running/
failure 状态、result-eligible 正式 run、逐 `analysis_stage`/`method_id` 的总量与正式量、19 项
deliverable、dev ledger 与 failure record 数量，避免人工填写“总共跑了多少”，也避免用大量 smoke
掩盖某个 Transformer 组件没有正式覆盖。相应地 closeout 现在把任何 `result_eligible=true` 的 initializing/running/
wall-time-exhausted run 保持为 pending；正式 mechanical terminal 必须有同 run-id 且原始
metadata/progress/partial bytes 哈希完整的 failure record。这个新门发现此前遗漏的正式
`d3_q3_attention_heads_b13_v1` 记录：它是已在本文件登记的 BF16 Q/K 观测路径重数量化问题，现已
绑定 135-row partial 并指向完整 Q2/Q3 b13 v2 replacements，未读取科学 effect。closeout 当前为
5/19 deliverables complete、14 missing deliverables + 7 in-flight formal runs、3 bound mechanical
failures、0 audit failures；全套更新为 592 passed、7 subtests passed。

双分片 selected-branch scorer/merge 的结果前实现锁为
`3c98effc5e96cef7e9310ade19f515002e3b60bc717e9b4213a8b94a17c5a727`，测试会逐次重算并
拒绝源码漂移。首次真实无损队列交接已由 Q2 b25 验证：旧实现完整发布 4082 requests、81,889
score rows 后旧父 shell 才退出；新版队列跳过 b13--b25，仅创建 b26 的独立目录并从 0 行启动，
post-block implementation/run-contract digest 均保持
`c6bf5664de2e69318159acbf0e18cc296220542dde88ae4f949e4a41d3e081c4`。没有重算、双写或条件
变化。新增实现锁测试后全套为 593 passed、7 subtests passed，两项 frozen hash 仍精确一致。

为了让“全面覆盖 Transformer”可直接审计，最终报告现在还自动输出 12 组件 probe coverage
topology：逐项列出全部注册 deliverables、其中具备 component-specific causal-support 资格的
deliverables，以及只允许 descriptive 的边界。它与 outcome-dependent component evidence matrix
分离，因此不能用大量 smoke 或一个无关正式 run 填补 attention/MLP/residual/readout 等组件的缺口；
`optimizer_scheduler` 在本阶段没有注册 SESOI，coverage 表会明确显示其 causal 列为空且禁止标为
supported。当前 execution census 可重建 139 个 run declarations、54 个 result-eligible 正式 run、
45 个已完成完整性检查，并按 analysis stage 与 Q0--Q3 方法分别计数；最终数值将在 closeout 时从
实际 bytes 重新生成，不使用这里的瞬时快照。

为防止上述 12 组件总表被误读为四个模型都具有同等直接证据，coverage topology 进一步固定为
`Q0/Q1/Q2/Q3 × 12 components` 的逐模型注册表。每个单元只允许列出为该模型预先注册且同时属于该
组件 evidence allowlist 的 deliverable；空单元明确渲染为 `not directly registered`。因此 Q2/Q3
的 attention-QKV、attention-output、MLP 或 residual 证据不能填入 Q0/Q1，而 Q0/Q1 的 trajectory/
branch/readout 也不会被误写成 Q2/Q3 的组件干预。19 项 deliverable 的 model assignment 与 closeout
清单要求精确同集，新增逐模型交叉测试后报告与合同定向测试为 33 passed；全套为 594 passed、
7 subtests passed，两项 frozen hash 未变。

逐模型表随后把每个单元进一步分成 `causal-support-capable`、`descriptive-only` 与
`not-directly-registered`。causal 资格必须同时满足组件专属 causal allowlist 和该 deliverable 的
预注册 model assignment；例如 Q0/Q1 的 layer trajectory 只能描述，不能借用 Q2/Q3 的 post-block
干预升级成因果支持，optimizer/scheduler 四模型在本阶段都没有 causal 资格。

一次只读终态审计还发现 3 个旧正式尝试的原 metadata 保留在 `running`：Q3 native-gate v1 的 b13/
b27 分别只完成 175/174 个请求，随后由不同 implementation/run-contract digest 的完整 8000-request
v2 replacement 取代；Q2 MLP b13 v1 已有 dtype-bound mechanical failure record 和完整 v2。
closeout 现在只有在 failure record 的 schema、run-id/path、metadata/progress/partial SHA256 全部通过
时，才允许该记录把原始 `running` bytes 解释为有效 mechanical terminal；否则仍 fail-closed pending。
两项 gate v1 新记录均声明未读科学 effect、未读 qrels、未开 source test，并绑定各自完整 v2 run。
审计由 7 个伪/真实混合 in-flight formal run 收敛为 4 个真实在跑 run，仍为 5/19 deliverables、
14 pending、0 failures；mechanical records 从 3 增为 5。

最终 primary-loss 归因的 outcome-independent 规则也补齐了 residual/norm 的双向字节门。此前
attention/MLP 标志会与 D2 synthesis 精确相等，而 residual/norm 只检查“报告为真时有节点支持”，
仍可能在两个下游节点都通过时事后挑标签。现在固定顺序为：先判 attention 与 MLP；只有二者都
不足且 fold-1 transition 复现时，norm-node 支持归为 `residual_norm_interaction`，只有 residual-node
支持且 norm-node 不支持才归为 `residual_composition`，其余保持 unresolved。六个注册 contrast 的
全部 BH/sign/evidence-role 条件仍必须逐项通过。新增重叠 residual+norm 反向测试后，全套为
597 passed、7 subtests passed。

12 组件 outcome matrix 进一步要求每行显式声明非空 `model_scope`。scope 中每个模型必须至少被
该行引用的注册 deliverable 直接覆盖；若 status 为 supported，则每个 scoped model 还必须分别有
component-specific causal deliverable，不能只由另一个模型的因果结果代替。primary-loss attribution
与组件表的交叉门也按同一模型匹配，因此 Q3-only 的 attention 支持不能作为 Q2 attention 主因的
背书。model/deliverable topology 被移到独立结果无关模块供 builder 与 contract 共用，避免两套映射
漂移；新增 unknown-model、部分 scope 无证据、跨模型主因借用三类反向测试后，全套为 600 passed、
7 subtests passed。

同一终态规则又清理了两个仅影响 census 的非正式陈旧 smoke：Q3 MLP one-row CPU smoke 与 Q2 RoPE
two-request smoke 均已写完目标 partial、无活进程，但未原子发布 completed；二者分别由新 digest 的
GPU identity smoke 与 32-request coverage smoke 替代。新增记录绑定原 metadata/progress/partial，
不改变科学实验数。当前 closeout 仍为 0 failures、4 个真实 in-flight formal runs；有效 census 的
`running` 从 6 降至恰好这 4 个，bound mechanical records 为 7。

rank-1 architecture opportunity 的 evidence gate 也从“一个大集合中命中任意一项”收紧为独立模态
组合：H1 需要 attention/selected-branch 因果侧加 head/group/context 侧；H2 需要 D1 representation
加 Q0/Q1 breadth；H3 需要 D2 causal transition/branch 加 native readout；H4 需要 Q2 objective 加
context/Q3 path/optimizer 中的独立侧。联合 H2+H3 必须同时通过 representation、breadth、D2 causal、
native readout 四个组，不能凭一侧故事升为 primary。对应的描述-only router 与不完整 combined card
反向测试保持通过；全套仍为 600 passed、7 subtests passed。

机会卡现在也必须声明非空 `model_scope`，且每个 scoped model 至少被引用的注册 deliverable 直接
覆盖。H1/H3/H4 的预注册最大范围固定为 Q2/Q3，H2 与联合 H2+H3 才允许扩到 Q0--Q3；最终结果只能
在这个上界内收缩，不能扩张。catalog 的设计内容与 SHA256 不变，报告另行列明允许随结果填写的仅是
rank、status、model_scope、rationale、falsification gate 与 evidence；innovation/training signal/
data/null invariant/modules/ablations/prior-work differences 仍逐字段锁定。新增无直接证据 scope 与越界
scope 反向测试后，report/closeout 定向测试为 53 passed。

全套回归随后为 602 passed、7 subtests passed；closeout 仍为 0 failures、5/19 deliverables、
4 个真实 in-flight formal runs，frozen plan/manifest SHA256 未变。

Q3 b23/b24 的无损 queue handoff 曾同时保留一组 parent-1 detached watcher 与一组同命令 session
watcher；虽然后续命令的 PID/argv 检查与 `&&` 可防止双写，仍存在无意义竞态。已在逐 PID/PPID/argv
核验后只终止两个重复 session watcher，保留 detached watcher；活跃 scorer、旧 stopped parent、
metadata/progress 与输出目录均未改动。每条 Q3 lane 现在只有一个未来可写 owner。

新增 `deep_dive_progress` 只读审计器与 CLI，固定枚举 Q2/Q3 × blocks 13--27 × folds 0/1 的 60 个
mandatory scientific bundles，以及两个 selected-branch conditional bundles。它只打开 metadata、
branch contract 与 branch evaluator 的顶层 status，不打开 scores、qrels 或科学 effect；会校验 run-id
与 result eligibility，并分别报告 completed/in-flight/missing、gate-stopped、mandatory remaining 与
maximum 62-bundle fraction。真实工作区复算为 23 fixed completed、3 fixed in-flight、34 missing、
37 mandatory remaining、最多 2 conditional remaining、23/62=`0.3709677`，0 audit errors。

最终 report builder 同时嵌入这份 census，并新增独立终态门：60 个 mandatory bundle 必须全部完成，
每个 conditional branch 必须完成或由冻结 contract gate-stop；否则即使19项表面 deliverable存在，
JSON/Markdown也拒绝生成。新增 identity-drift、gate-stop/completed、nonterminal-report 四类测试后，
progress/report/contract/closeout 定向测试为 57 passed。

全套回归为 606 passed、7 subtests passed；真实 progress CLI 再次得到同一 23/62 census 与 0 errors，
四卡利用率 98--100%，两条唯一 Q3 detached handoff watcher 均存活，frozen hash 未变。

D2 后续 breadth 队列进一步按四张物理 GPU 重新分配唯一 ownership，避免 Q2/Q3 selected branch 完成后
GPU0/GPU2 空闲而 breadth 只在两条 Q3 lane 上串行。lane0/1 只保留 Q3 b20/b27 component breadth、
context 与 native readout；lane2 接管 Q2 b20 attention/MLP/RoPE、Q0 breadth 与 Q2 optimizer；lane3
接管 Q2 b27 components、Q1 breadth 与 Q3 optimizer。Q2 selected-branch shard1 queue 在无论 eligible
完成还是 gate-stop 后都转入 lane3，新增的 Q2 postselected watcher则等待当前 Q2 D2 owner 与 branch
contract/必要 evaluation 闭合后转入 lane2。所有 run-id、config、evaluator、gate 与冻结实验集合不变，
只改变未来唯一执行 owner；新增静态互斥、shell syntax 与 owner-command identity测试。

重算 D2 qrels-blind 科学账本仍为 23/62=`0.3709677`：60 个固定 post-block bundle中23 completed、
3 in-flight、34 missing，因此 attention/MLP/residual 主归因还差37个 mandatory bundle；Q2/Q3 若各自
fold-0选择与fold-1确认门通过，再各有最多一个 selected seven-node branch bundle，即最大还差39个。
协议计划与 manifest SHA256 分别仍为 `07440f4a...a584` 与 `76445ae3...a758`。定向队列/进度/closeout
测试20 passed；全套回归为609 passed、7 subtests passed。

对长队列末端做未定义变量审计时发现 lane3 的 Q3 optimizer replay曾引用不存在的 `q3_config`；
在 `set -u` 下会于所有前序 breadth结束后机械退出，但尚未执行、未产生或污染任何 run。现已改为
脚本顶部冻结的 Q3 `config`，并增加禁止 `$q3_config` 回归断言。两个已启动 Q3 lane 的 `/proc/.../fd/255`
与当前脚本 SHA256 均为 `f6b1a578...e5f7f`，确认修复已进入活队列后续读取路径；当前四个 scorer未重启。
全套回归保持609 passed、7 subtests passed。

selected-branch handoff又补了两类结果无关防护。Q2 shard1与Q3 lane1不再用 `b*_smoke` 通配符；
它们从冻结 contract读取 `selected_block`，只等待该精确 block 的 smoke metadata，避免其他层残留
completed smoke误放行。选中层范围按注册选择域硬限制为14--27，contract eligibility必须严格为
布尔值；无效值会机械退出，不能被当作 gate-stop。Q2 shard1 的 smoke等待也从无限轮询改为识别
terminal failure。相同 eligibility/block门已加入仍在运行的Q2主队列与两条Q3主队列，并通过各自
`/proc/PID/fd/255`确认活脚本已更新；未改 scorer、run-id、科学条件或选择规则。

Q2 b26 fold0随后以4082/4082 requests原子完成，`scores.jsonl` SHA256
`b0eb754e...bc97c`与metadata完全一致，4082行、qrels未读、source test未开、result eligible；
closeout完整性计数从45增至46。Q2唯一owner在约21秒内自动切换到b27 fold0。D2机械进度因此为
24/62=`0.3870968`，固定60项中24 completed、3 in-flight、33 missing、36 mandatory remaining，
再加最多2个conditional selected branch，即最大还差38个科学bundle。
新增精确smoke、strict contract与通用shell变量审计测试后，全套回归为610 passed、7 subtests passed。

qrels/score-blind D2 progress审计现在对每个 in-flight bundle额外报告机械 request进度：model、block、
fold、completed/total requests、fraction与更新时间；科学完成率仍只按原子completed bundle计算，
不会把partial折算为证据。审计会对 completed>registered total与损坏progress JSON fail closed。
真实输出可见Q2 b27与Q3 b23/b24三条独立进度而不打开任何科学effect。定向44 passed；全套为
611 passed、7 subtests passed。

最终 component matrix 的结果无关 taxonomy从12个合并项拆为14个独立项：
`token_embedding_position`拆成`token_embedding`与`positional_encoding_rope`，`residual_norm`
拆成`residual_composition`与`normalization`。token embedding当前只有表示/readout-row/optimizer的
描述性注册证据，禁止借用RoPE因果结果升级为supported；position必须由D5 RoPE因果证据支持。
residual composition必须由D2 selected branch支持，normalization则由独立RMSNorm/native-readout
路径支持；primary residual与residual/norm标签分别交叉门控到对应组件，不再共享一个模糊行。
实验、manifest、机会catalog与scorer均未改变。新增四组件逐模型coverage和反向claim测试后，定向
56 passed，全套614 passed、7 subtests passed。

继续按冻结计划的`Q/K/V/O`与`MLP gate/up/down`边界拆分后，component matrix最终预注册为18个
独立行：Q/K routing、V transport、attention output，MLP feature formation与MLP output分别列出；
native readout与score-calibration/nullspace分开，optimizer改为准确的effective update，并把LoRA
parameterization单列。D3 head observation不能替V-edge因果证据，D4 exploratory group不能替
MLP output/feature的确认性因果证据；Q0/Q1 aggregate branch与native readout则只给其实际覆盖的
attention output、MLP output、residual与normalization逐模型因果能力。LoRA gauge/merge identity和
一步geometry保持descriptive，不能升为parameterization因果支持。primary MLP归因现在必须与
`mlp_output`支持一致，不再从feature-group描述性结果借证据。定向59 passed，全套617 passed、
7 subtests passed；实验与冻结hash均未改变。

组件矩阵的模型边界进一步从deliverable级收紧为`component × deliverable × model`三级拓扑。原因是
D7 optimizer aggregate虽同时登记Q2/Q3，Q2 replay覆盖embedding、Q/K/V/O、norm与MLP全参数族，
Q3 replay只覆盖q/v LoRA；旧的粗映射会让Q3借到Q2的embedding/MLP/O update证据，或让Q2借到
Q3 LoRA parameterization证据。现在token embedding、attention output、MLP feature/output的D7
coverage只属于Q2，LoRA replay/path只属于Q3；Q/K与V因Q3确有q/v LoRA仍保留对应描述性coverage。
最终builder也输出同一三级表，contract的scope、evidence relevance和per-model causal门全部使用它。
新增五个跨模型借证据反向单元后，定向64 passed，全套622 passed、7 subtests passed。

D2 progress的24/62计数资格又加入scorer自身的data/identity边界：每个fixed bundle必须逐项匹配
method、block、fold、`registered_mechanism_diagnostic` evidence mode、`qrels_read=false`、
`source_test_opened=false`与result eligibility；审计器自己未读qrels不再被误当成scorer也合规。
conditional gate-stop/completed还必须通过最小contract的method、14--27 selected block、七节点顺序、
fold-1 population、evidence role与qrels isolation；漂移contract不计resolved unit。真实27个已存在
postblock metadata全部通过，当前24/62仍为0 errors。

最终architecture opportunity schema不再无条件强迫“恰好一个primary”。只要任一候选同时满足全部
独立证据组且linked hypotheses均为supported/weakened，仍必须选唯一rank-1 primary；若所有候选
都因完整负结果或unresolved假设不具资格，允许零primary，但rank-1只能deprioritized/rejected，
不能用secondary模糊处理。全拒绝的合法negative closeout也可生成报告，避免为了schema包装方法。

为直接回应“被弄没而非反转”，final builder现在从admitted D2 selected synthesis自动、完整抽取
`2 models × 6 adjacent node pairs × 2 endpoints = 24`行attenuation transition profile。所有行固定
展示mean、CI、BH q、missing与evidence role；显著负变化只标为`significant_attenuation`，显著正变化
标为amplification，明确声明既非hidden-state literal sign reversal，也不能单独作为primary component
归因。adjacent family原注册没有期望符号，因此builder强制`expected_sign=null`、
`registered_support=false`，不允许事后把方向描述升级为确认性支持。全套回归为636 passed、
7 subtests passed。

最终component matrix又加入结果级、逐模型的fail-closed交叉门。过去`status=supported`虽必须引用
component-specific causal deliverable，却仍可能只凭“该实验存在”而无视其注册结果；现在每个supported
模型都必须由所引用实验的实际family结果通过固定路由。D2 branch逐节点复用same/stress/specificity/
direction-scale六门；D2 post-block要求fold-1定位resolved；D3 Q/K routing只能用logits-mask，V transport
只能用value-edge，均要求BH `q<0.05`、all/fold0/fold1同非零方向且all CI排除0；D5 contextual按
content-neutral与attention-null各自路由；D6 native readout要求同一node/scope的same-minus-null与
same-minus-cross都通过；D7 loss-gradient只接受预注册conflict SESOI与FDR同时通过。Q0/Q1 final-readout
明确是`confirmatory_family_membership=false`，因此不再被列为native readout或normalization的
causal-support-capable证据；Q2/Q3 native readout也不能被重命名为RMSNorm因果，normalization确认只由
D2 selected norm节点承担。

RoPE的冻结停止规则需要两个独立量：compression-minus-expansion进入36-cell FDR family，同时active
compression-minus-baseline的NDCG区间必须完全越出`±0.005`。原evaluator只保存前者的推断区间、后者
仅保存均值，无法机械验收完整规则；现给后者补上all/fold0/fold1 cluster-bootstrap CI并明确删除其p值、
标为support gate而非新family member，不改变36-cell family、scorer、condition或冻结manifest。
新增context、Q/K-vs-V、RoPE双门、readout specificity、objective SESOI以及描述性越级等反向测试后，
定向86 passed；全套为643 passed、7 subtests passed。冻结计划与manifest SHA256仍分别为
`07440f4a...a584`与`76445ae3...a758`。

Q3 b24 fold0随后以4082/4082 requests原子完成，scores SHA256
`87a762c8...b762b2b`与metadata一致，qrels未读、source test未开、result eligible；lane1自动切换到
Q3 b26 fold0。D2机械账本因此更新为25/62=`0.4032258`，固定60项中25 completed、3 in-flight、
32 missing、35 mandatory remaining，再加最多2个conditional selected branch，即最大还差37个
科学bundle；四卡继续满载，未读取未闭合family的科学effect。

native-readout结果路由随后进一步要求same-minus-null与same-minus-cross必须在同一node/scope且同一个
endpoint共同通过，禁止用margin和NDCG各支撑半条证据链。RoPE support-gate定义也写入evaluator顶层
schema与closeout contract：active contrast、NDCG endpoint、`[-0.005,+0.005]`区间、fold方向门、
compression-minus-expansion FDR门和“active contrast不是新family member”都必须逐项一致，缺失即
fail closed。全套回归为645 passed、7 subtests passed。

Q3 b23 fold0也以4082/4082 requests原子完成，scores SHA256
`f413492b...e980d69`与metadata一致，qrels未读、source test未开、result eligible；lane0自动切换到
Q3 b25 fold0。D2账本现为26/62=`0.4193548`，34 mandatory与最多2 conditional仍未完成；closeout
已逐项核验48个formal completed run，5/19 deliverables闭合、0 failures，未提前读取D2 family effect。

result-level路由继续去除了两个结构性借证据：D3 logits/value/neutral edge只定位Q/K routing或V
transport，不能把上游edge效应升级成`attention_output/o_proj`瓶颈；Q0/Q1 block-output patch没有
selected-branch的stress/specificity/direction-scale门，不能单独升级成residual composition。
attention output仍可由D2 selected o-proj节点或Q0/Q1直接o-proj node支持，residual composition确认则
只保留D2 composition-safe selected branch。score-calibration/nullspace不再错误依赖readout显著性，
改为只接受qrels-blind的精确`score_ij=common_i+relative_ij`重组、零和relative以及严格数值误差门。
新增静态闭包测试保证每个causal-support pair都有且只有一个结果级路由，未来扩展证据拓扑却忘记
实际结果门会直接失败。

H1/H4 supported也与组件矩阵联动：H1必须在同一模型同时具备Q/K routing与history-routing支持，
H4必须具备通过SESOI/FDR的loss-gradient支持，不能只引用描述性LoRA/optimizer geometry。机会矩阵
进一步限制H4证据scope为Q2、H2+H3 signed path为Q2/Q3；H1/H3/H4及组合方向的primary对每个声明
模型逐证据组核验coverage，禁止用Q2 readout替Q3补齐或用Q0/Q1 breadth扩大signed-residual scope。
全部新增反向测试后全套为654 passed、7 subtests passed；冻结计划/manifest与运行中scorer均未改。

最终报告又把结果级门与决策门本身公开成机器可读catalog：25个`component × deliverable`精确支持
条件逐项列出，H0--H5的独立证据组/组件要求以及五个机会方向的全局、逐模型证据组和允许scope也
进入JSON/Markdown。这样最终文本不能只给一个结论而隐藏其判定规则，也不能让同一证据在不同模型
之间暗中借用。catalog只冻结解释合同，不读取当前未闭合family的effect，也不改变实验或family。

19项closeout交付物进一步加入SHA绑定的result-structure census。审计沿实际JSON结构逐项计数：
确认性family rows、D2 family container与flattened rows、Q2/Q3 model-block、每个block内部8个GQA
group/16个MLP group、LoRA states/comparisons、optimizer每个control/surface下的三objective或三
coordinate mode；通配路径实际遍历内部单元，禁止用`6 blocks × 8 groups`的乘法假设掩盖单个缺失
group。Q0/Q1 trajectory只要求注册描述性结构非空并记录实数，不擅自制造确认性family。任一source
路径越界、SHA漂移、container类型错误、精确注册单元缺失或19项coverage不全都拒绝最终报告；
census明确不按effect筛选或汇总，也不把重叠container和flattened rows相加。定向closeout/report
测试110 passed，全套为659 passed、7 subtests passed。

运行期间Q2 b27 fold0已经原子完成，D2机械账本更新为27/62=`0.4354839`，33 mandatory与最多2个
conditional尚未完成。Q2队列按硬顺序先物化fold-0选择记录，再自动进入b13 fold1；这里不读取或
报告所选层与effect。Q3 b25/b26 fold0继续分卡，第四卡仍运行注册RoPE任务，四张卡均由独立目录的
注册队列占用。冻结计划与manifest SHA256仍分别为`07440f4a...a584`和`76445ae3...a758`，source
test保持关闭。

只对当前已闭合的5项真实交付物试运行新structure census也全部通过：D1 cells=96、Q3 native gate
blocks=2、Q2 native readout rows=12、Q2 objective rows=12、Q3 LoRA states=3且comparisons=168；
没有输出effect值。同期只读closeout核验49个formal completed run、5/19 deliverables、0 failures，
7条既有mechanical failure record继续原样绑定，不能被删除或解释成科学结果。

Q2 fold0/fold1硬顺序也作了独立机械复核：selection状态completed，文件mtime严格早于首个fold1
metadata；fold1 command与run contract同时绑定selection路径及SHA256
`d80ab0f3...0d29e`，scorer声明`qrels_read=false`、`source_test_opened=false`。该检查没有读取
selected block或任何effect，确认后续组件归因使用的是先冻结选择、再物化确认fold的路径。

为避免后续只用D2的bundle百分比代表整套模型探索，新增长度独立的
`status_deep_dive_overview.py`。它联合冻结evidence topology、D2机械进度与closeout，但不读取或
汇总科学effect；分别输出D1--D7交付闭合、18组件的注册输出/因果可支持输出、四模型各自coverage、
formal run状态与机械failure记录。完成一个描述性LoRA或optimizer输出不会被写成因果支持，
`all_causal_capable_outputs_closed`也只表示所需文件齐全，明确固定
`scientific_support_inferred_from_completion=false`。

真实账本当前为D1 `1/1`、D2 `1/3`、D3 `0/3`、D4 `0/1`、D5 `0/2`、D6 `1/6`、D7 `2/3`；
18组件中10个已有至少一个注册输出闭合、8个尚无闭合输出，14个具因果支持资格的组件中仅1个所需
输出全部闭合。formal result-eligible声明为58项：49 completed且完整性通过、4 running、5条旧
mechanical-failure formal路径；closeout仍绑定7条总mechanical failure record且0 audit failures。
overview/closeout定向测试109 passed，全套更新为662 passed、7 subtests passed。

广度账本继续下钻到H0--H5与五个架构机会的独立证据组readiness。每组只记录注册source中是否已有
闭合文件，并固定`scientific_gate_passed=null`、`hypothesis_support_inferred_from_readiness=false`、
`opportunity_priority_inferred_from_readiness=false`；H5因本阶段没有独立第二seed，readiness固定为
not applicable。对带逐模型合同的机会，group source还必须覆盖该模型，不能只在全局union中存在。
真实结构账本因此显示H0/H4的全局独立source组已各有闭合文件，但这不等于假设supported；尤其H4
机会虽全局两组看似ready，第二组当前闭合来源只属于Q3，不能替Q2 objective补证，所以允许的Q2
model readiness仍为false。H1/H2/H3及相应组合机会仍有结构缺口。新增反向测试后定向79 passed，
全套仍为662 passed、7 subtests passed。

组件账本再增加逐模型coverage debt，避免把Q2/Q3深挖外推到所有四个ranker。真实冻结拓扑中，Q2
注册17/18组件、14项具因果支持能力，Q3注册18/18、13项具因果支持能力；Q0/Q1各注册10/18，但
当前结果合同下只有attention output与MLP output两项具直接因果支持能力。Q0/Q1未直接注册Q/K
routing、V transport、candidate interaction、history routing、loss/optimizer等八项，其trajectory、
readout、norm/residual与RoPE多数只作为描述性跨架构边界。全局`token_embedding`、
`mlp_feature_formation`、`optimizer_effective_update`、`lora_parameterization`四项也只有描述性证据，
不能因参数/几何已观察就写成因果机制。冻结计划不允许当前按结果追加Q0/Q1内部实验，因此这些项
必须进入最终remaining uncertainty，而不能声称四模型同深度覆盖。新增反向测试后定向67 passed，
全套仍为662 passed、7 subtests passed。

同一coverage debt现已从运行状态工具接入最终closeout JSON/Markdown，不再依赖作者在
`cross_model_boundary`自由文本中自愿披露。最终报告固定增加逐模型表，列出registered component
count、causal-support-capable count、registered-but-descriptive-only与not-directly-registered完整
清单，并在表前逐字声明“causal-capable不等于科学gate已通过”。公共摘要由冻结拓扑直接生成，不读
outcome，Q0/Q1的八项缺口与四项全局描述性边界不能在写作时删除。针对性report/overview测试
80 passed，全套更新为663 passed、7 subtests passed。

为避免19项结果闭合后自由手写漏掉负结果、模型边界或证据门，新增独立的
`transformer_deep_dive_decision_worksheet`。它逐项物化7段narrative、18组件、Q2/Q3主损失归因、
H0--H5、5个冻结架构机会与19项closeout路径，同时公开每行允许的deliverable、逐deliverable模型
coverage、因果支持资格、独立证据组和机会逐模型证据组。所有科学decision字段固定为`null`，状态
固定为`todo_not_final_report_input`，`final_report_input=false`；把它直接交给最终decision validator
必须因空narrative失败。H5因本阶段无独立第二seed，在工作表中直接移除`supported/rejected`两个
允许状态；四个只有描述性证据的组件也固定标记为本阶段不可升级supported。生成器先核验4项冻结
source SHA，任一漂移就拒绝物化，且不读score、qrels或source test。真实仓库已生成结构审计副本到
`tmp/deep_dive_decision_worksheet_v1.json`：19 deliverables、18 components、2 attribution models、
6 hypotheses、5 opportunities全部齐全，所有decision仍为空。定向70 passed，全套更新为
669 passed、7 subtests passed；冻结计划/manifest SHA仍分别为`07440f4a...a584`与
`76445ae3...a758`。

机械失败的最终可见性审计又发现一处合同缺口：此前closeout会把所有有效failure record放入JSON
admission，但Markdown只显示总数，组件矩阵也可以一条不引用仍声称“全部保留”。现已要求全部
admitted failure path的并集必须至少绑定到一个组件行；未分配、引用未接纳路径或把
`mechanical_failure`状态写成无绑定记录都会拒绝最终报告。历史失败可与其后成功替代实验共同绑定，
不会被误写成科学阴性。Markdown新增逐条ledger，固定显示run ID、状态、record path和SHA256，
同时组件表继续显示其解释归属。当前7条既有记录因final closeout尚未开始人工归因，不提前分配；
空白worksheet只冻结“全部必须分配”的规则并保持decision为null。定向84 passed，全套更新为
671 passed、7 subtests passed。

结果级支持门也从静态规则目录扩展为最终自动outcome census。closeout完成后，报告使用与
`supported`组件反向校验完全相同的`component_result_support`函数，对25条预注册
component×deliverable route展开45个逐模型结果，逐项输出`registered_support`或
`registered_support_not_established`并绑定source path/SHA。这里不会复制原始effect值，且固定声明
“support not established不自动等于相反因果效应”；因此missing、gate-stopped、方向/FDR/CI未过
不会被包装成反向机制，但也不能只藏在原始metrics中。所有25 route必须全覆盖，未知route、未接纳
source、SHA漂移、analysis type/status不符都会拒绝最终报告。这样18组件矩阵既公开门的定义，也
公开每个模型实际过门与否，避免只展示支持项。定向80 passed，全套更新为673 passed、
7 subtests passed。

最核心的attention/MLP/residual主归因也改为显式机械推导，不再只在validator内部核对若干人工
boolean。新增纯函数从D2 post-block localization与selected七节点的六重确认门，为Q2/Q3逐模型输出
fold-1 transition、attention、MLP、residual-node、normalization-node支持，以及composition与
residual/norm criterion；随后按冻结顺序唯一派生
`mixed_attention_mlp → attention_output → mlp → residual_norm_interaction → residual_composition`
（任何resolved标签前必须fold-1复现，否则unresolved）。norm在residual与norm同时过门时保持既有
确定性优先级，禁止事后选更好讲的标签。最终报告会先展示这张machine-derived表，再展示作者的
归因解释；人工`primary_component`与派生标签不一致即失败，descriptive head/group固定不能参与
主因选择。两个D2 source都绑定closeout SHA，表只输出门结果，不复制原始effect。定向81 passed，
全套更新为674 passed、7 subtests passed。

主归因worksheet进一步公开每个标签的精确criterion与确定性优先级，并冻结
`residual_node_support_alone_is_not_composition=true`。新增反向测试同时让attention o-proj与
block-output residual节点通过：机器必须保留`residual_node_registered_support=true`供路径解释，
但composition criterion为false，主因仍派生为`attention_output`；只有fold-1复现、attention/MLP与
normalization均不足、且residual节点独立过六重门时才允许`residual_composition`。这阻止“下游
residual也响应”被误写成“损失源于residual composition”。定向88 passed，全套更新为675 passed、
7 subtests passed。

主归因的`evidence_strength`也移除人工自由度，直接由D2 localization状态与派生标签确定：具体组件
唯一成立才是`registered_confirmatory`；fold0已选转折但固定fold1未复现是`exploratory_only`；
`fold0_no_negative_transition`或科学/机械门后未运行是`gate_stopped`；fold1虽复现但无唯一组件门，
或旧fixture缺少状态，保持`unresolved`。validator现在同时核对六个直接flag、派生component和
派生strength；把未复现写成gate-stopped、把gate-stopped写成普通unresolved、或把确认性转折但无
组件写成registered都失败。worksheet公开四种强度的固定映射。定向89 passed，全套更新为
676 passed、7 subtests passed。

六重selected-node门又增加行级唯一性。此前helper按`contrast_id`建字典，正常synthesis已保证唯一，
但若最终文件损坏为“重复一行、缺失另一行”，后写值理论上可能覆盖前值。现在每个
model×node×target-margin必须恰有唯一的same、cross、wrong-history、norm、direction、random六个
contrast；任何重复直接抛出evidence schema错误，不能先把attention/MLP误判为false后再回落成
residual composition。缺失仍按fail-closed不支持处理。该helper同时服务主归因与25-route结果门，
所以两处口径一致。定向90 passed，全套更新为677 passed、7 subtests passed。

其余result route的重复键风险也作了逐类审计：attention/context/readout通过`_family_q`要求匹配
family row恰为一条，重复时q值为None并fail-closed；D7 loss-gradient此前直接`any`扫描12-cell
family，理论上可能在一个重复cell存在时由另一个cell过门。现已先检查全部
`(state,surface,endpoint)`键唯一，重复直接报objective-family schema错误，再允许SESOI/FDR门。
该变化不改变任何effect、SESOI或多重检验，只阻止损坏family建立支持。定向84 passed，全套保持
677 passed、7 subtests passed（反向断言加入既有D7测试）。

D7完整性再从“键唯一”收紧到精确冻结笛卡尔积：family必须等于2个state
`{base_initialization,frozen_final_checkpoint}` × 3个surface
`{recurrence,strict_transfer,other_overlap}` × 2个endpoint
`{ranknet_listnet_cosine,observed_minus_label_shuffle_cosine}`的12个键。即使行数仍为12且互不重复，
用未注册surface替换一个cell、漏掉任一键或加入额外键都会报coverage差异；随后才评估冲突SESOI和
BH门。测试fixture也改用完整12-cell family，不再以单行近似正式schema。定向84 passed，全套保持
677 passed、7 subtests passed。

长任务运行中又做了一次四卡单写者与后继顺序审计。四个active run各恰有一个Python scorer：Q2
b13 fold1绑定physical GPU0，Q3 b25/b26 fold0分别绑定GPU1/GPU3，Q3 RoPE b13 v2绑定GPU2；每个
run另有一个只负责wall-time/resume的shell parent，不是第二writer。Q2 b14 fold1、Q3 b27 fold0与
Q2 RoPE b13 v2均尚未创建metadata且writer计数为0，说明后继没有绕过当前bundle提前启动；现有
queue将在当前run完成后按冻结序列接力。Q2/Q3 postblock当前进程预计仍能在13,500秒单段边界内
完成，RoPE已由resume loop正常跨段续跑，无需中断或改调度。

针对长队列的重复watcher风险，公共`run_deep_dive_resume_loop.sh`现对metadata规范绝对路径取
SHA256，并在锚定仓库根的`tmp/deep_dive_resume_locks/`用nonblocking `flock`持有单写者锁；同一
路径即使从不同cwd、使用含`..`的别名启动，第二个wrapper也固定以exit 7拒绝，原writer完成后锁
自动释放。锁在检查既有metadata
状态前取得，因此不会出现两个wrapper同时把missing判断为可启动的TOCTOU窗口；不同metadata路径
仍可并行，且锁目录不创建run目录或科学结果。该保护只作用于修改后启动的wrapper，不追溯重启
四个当前writer；当前writer已另行确认各自唯一，所以没有中断实验。规范路径别名、并发拒绝和完成
后幂等重入的runtime测试均通过，queue定向测试5 passed，全套更新为678 passed、7 subtests
passed；冻结计划/manifest SHA仍分别为`07440f4a...a584`与`76445ae3...a758`。

D2进度账本现明确分开两种不能混用的百分比：`maximum_completion_fraction`只在一个科学bundle
完整闭合后增加；新增`maximum_request_weighted_execution_fraction`则把每个active bundle的
`completed_requests/fold_request_count`折算成bundle-equivalent，仅用于运行进度。每个partial
progress在纳入折算前必须用`run_contract_sha256`与metadata精确互证；缺失或漂移时fraction固定为
null、账本失败，不能用错run的partial虚增进度。该工具仍不读score effect、qrels或source test。
真实接力中Q2 b13 fold1已完整闭合，b14由冻结队列在同一卡自动启动且使用新单写者锁；D2科学
闭合由27/62升到28/62=`0.4516129`，另外两条Q3 fold0与RoPE lane均未被打断。定向
progress/overview测试20 passed，全套更新为679 passed、7 subtests passed；冻结计划与manifest
哈希保持不变。

对“attention/MLP/residual主因”的fold scope作了单独合同审计。冻结计划的精确统计规则是：fold0
只选择post-block相邻转折，fold1独立确认该转折；七节点branch的effect、bootstrap、p值与BH全部只
使用fold1。因此若最终唯一组件门通过，允许的表述是`split-sample confirmatory localization`，不能
表述为attention/MLP/residual节点效应本身在fold0/fold1重复。该边界现由三级lineage强制：每模型
selected-branch evaluator输出固定fold-scope对象，双模型synthesis逐输入精确校验，最终report
derivation与人工decision再次要求`selected_branch_node_inference_fold=1`、
`node_effect_two_fold_replication_tested=false`和`split_sample_component_localization=true`；缺失、漂移
或人工改写均fail closed。Markdown同时显示该scope，worksheet仍保持全部decision为null。这个修正
不读取或改变任何effect、family、门槛或实验顺序；节点两折重复若仍有必要，只能在本轮冻结closeout
以后由新授权注册，不能按当前结果追加。相关定向测试100 passed，全套更新为684 passed、7 subtests
passed，冻结计划/manifest SHA保持不变。

“主组件”与“transfer ranking失败原因”也增加独立endpoint边界。冻结计划把target margin定义为
primary direction endpoint、strict-transfer NDCG@10定义为独立secondary utility family；此前机器
主归因只读取target-margin六门，若直接写成NDCG transfer failure cause会过度外推。现在每模型同时
计算同一节点在NDCG family的六门支持，并派生三种唯一scope：组件未解决为`unresolved`；margin组件
已解决但同节点NDCG未通过为`target_margin_only`；只有同一组件在两个endpoint均过门才是
`target_margin_primary_with_strict_transfer_ndcg_corroboration`。residual/norm类别要求两个endpoint
支持节点集合有交集，禁止用一个residual节点的margin与另一个residual节点的NDCG拼接成佐证。
worksheet、人工decision validator与Markdown都固定primary endpoint、NDCG corroboration flag及
`target_margin_component_is_not_automatically_ndcg_cause=true`；因此最终可以找到有害margin生成位置，
但只有secondary utility family也支持时才升级为完整ranking-transfer解释。该修改不读取当前未闭合
effect，也不改变注册统计；定向测试102 passed，全套更新为688 passed、7 subtests passed，冻结
计划/manifest SHA保持不变。

七节点干预的因果语义也由计划文字升级为强制合同。每个patch是
`null_context_sufficiency`：检验full-context节点状态放入null recipient是否足以重现有害响应；没有
反向necessity干预，也不是additive或Shapley component decomposition。因此机器派生与人工decision
都必须固定`necessity_tested=false`、`exclusive_component_origin_established=false`、
`additive_or_shapley_contribution_estimated=false`，primary标签的解释固定为
`registered_candidate_bottleneck_not_unique_origin`。最终Markdown章节也从“primary loss attribution”
改为“primary registered candidate bottleneck”，并显式展示该claim boundary。这样attention output
通过只能说明其状态对有害margin（及可选NDCG corroboration）具注册充分性，不能声称attention是唯一
起源或给出贡献比例；MLP、residual/norm同理。定向测试103 passed，全套更新为693 passed、
7 subtests passed；所有科学effect、family与冻结SHA保持不变。

NDCG corroboration进一步区分统计支持与实际utility尺度。冻结`±0.005`等价带已在计划中注册；一个
NDCG contrast即使BH显著，也可能CI完全落在等价带内，不能据此解释有实质大小的transfer failure。
机器现对同一margin主组件同时派生：(a)六门`registered_support`的statistical corroboration；
(b)更严格的utility-relevant corroboration，要求同组件全部六个NDCG contrast的95% CI上界均低于
`-0.005`。transfer scope因此固定为四档：`unresolved`、`target_margin_only`、
`target_margin_primary_with_statistical_ndcg_corroboration_only`、以及
`target_margin_primary_with_utility_relevant_ndcg_corroboration`。utility flag必须蕴含statistical
flag；residual/norm仍要求两个endpoint落在同一具体节点。该层只是预注册等价带上的保守claim gate，
不改写p值、family或科学结果。定向测试104 passed，全套更新为694 passed、7 subtests passed；
冻结计划/manifest SHA保持不变。

层选择语义也避免了“最早弄没信号”的过度表述。冻结selector是
`argmin_k(E_k-E_{k-1})`并tie取较小block，它找到的是blocks 13--27中fold0均值最负的相邻post-block
step，再在fold1确认该固定step；它不是change-point/first-onset估计，也没有证明全局唯一loss layer。
最终机器与人工decision现固定`selected_transition_interpretation=`
`largest_fold0_mean_negative_adjacent_postblock_step`、`earliest_loss_layer_established=false`和
`global_unique_loss_layer_established=false`。Markdown与本日志也相应把“首次衰减”改为“最强注册
相邻下降候选”；block内adjacent-node表继续只是sufficiency transition profile，不能单独决定primary
组件。定向测试107 passed，全套更新为697 passed、7 subtests passed；冻结SHA与科学结果不变。

## 4. 可能的防止方式（等待诊断后排序）

以下只是架构机会，不在本阶段实现，也不作为新的论文方法：

1. 若selected-block证据把attention列为注册候选瓶颈：保留独立的history-preference transport path，或对
   history-to-candidate edge 使用受控 gate，避免被 query-only attention 稀释；Q2 的总跨位置
   delta 已随深度增强，因此 gate 必须选择 factor-aligned content，不能以增加总 history attention
   mass 为目标或验收指标。
2. 若selected-block证据把MLP列为注册候选瓶颈：对中层history delta使用residual preservation/gated skip，或增加
   layerwise direction-preservation objective，而不是简单提高 LoRA rank。
3. 若信号仍在但 readout 利用错误：显式分解 request-common shift 与 candidate-relative
   preference 以便审计和训练归因，并只对后者施加 signed compatibility/abstention gate；不能把
   现有分数简单去均值当作改进，因为 common offset 本来就不改变排名。
4. 若只有特定 request surface 会被覆盖：使用 recurrence/strict-transfer-aware signed trust gate，
   不对所有请求统一放大历史。
5. 若 direction 保留而 scale 崩溃：优先做 norm/calibration control；若 direction 真正改变，才考虑
   subspace alignment 或 block-local adapter。
6. 若总 full-null 位移继续增长、但只有某些 preference factor 衰减：不优先增加 history embedding
   幅度或 LoRA rank；改为 factor-selective residual slots/gates，并要求每个 factor 对候选相对分数的
   signed contribution 可独立归零和 patch。Q2/Q3 的 brand/category 异质性要求这种控制按模型与
   factor 接受或失败，不能用一个全局 preservation loss 覆盖所有历史分量。
7. 已观察到 same > wrong-user 但 same < null 的请求：把 user/history specificity 与
   relative-to-null usefulness 拆成两个独立、可审计的 gate。usefulness gate 为零时必须精确恢复
   query-only/null ordering；wrong-user 必须在 specificity gate 失败，specific-but-harmful 必须能在
   usefulness gate abstain。两个 gate 都不能在推理时读取 qrels，只能由 train-visible
   query-history-candidate consistency 学习，并以冻结 same/wrong/null 对比分别验收。

上述 preservation 与 abstention 不矛盾：preservation 只保证一个中层、factorized、
candidate-relative residual **仍然可用且可独立 patch**，不要求它无条件进入最终分数。条件式组合
应保持 query-only/null backbone，并把保留下来的 residual 依次经过 specificity 与 signed
usefulness gate；任一 gate 为零都必须精确回到 null ordering。只有当 D2 独立 fold 确认一个有益
中层状态、D2 branch 定位表明后续确有丢失/覆盖、且 factor/same-wrong 控制支持特异性时，这个
组合才保留为 primary architecture opportunity；任一前提失败就淘汰，不能靠当前描述性分组把它
升级成方法。

最终机会排序必须以完整 D2--D7 证据矩阵为依据。当前不把上述任何控制包装成方法，不打开
source test，不增加结果依赖的层、head、group、seed 或数据集。
## 2026-07-19 transfer explanation ladder contract

为避免最终报告把不同强度的证据压缩成同一句“找到 transfer 失败原因”，报告契约新增五级、
严格有序且机器派生的 explanation ladder：

1. `unresolved_or_gate_stopped`；
2. `reproduced_layer_transition_without_unique_component`；
3. `target_margin_component_sufficiency`；
4. `target_margin_component_with_statistical_ndcg_corroboration`；
5. `target_margin_component_with_utility_relevant_ndcg_corroboration`。

等级 2 只表示 fold-0 选择的层转折在固定 fold-1 重现，但七节点尚未给出唯一组件；等级 3
开始要求 target-margin 上的注册组件充分性；等级 4 要求同一组件、同一具体节点通过独立 NDCG
family；等级 5 还要求所有同组件 NDCG contrast 的 95% CI 上界严格低于冻结的 `-0.005`
等价带。residual/norm 不能把不同节点的 margin 与 NDCG 证据拼接成跨端点佐证。

即使等级 5 也固定解释为 candidate-bottleneck sufficiency diagnosis，不建立 necessity、唯一
因果来源、可加/Shapley 贡献或 transfer 失败的完整原因。证据派生、最终人工决策、空白
worksheet 和 Markdown 必须给出完全一致的等级，任何越级或降级都 fail closed。

实现后定向报告测试 110 项通过；完整仓库测试为 `700 passed, 7 subtests passed`。重新生成的
`tmp/deep_dive_decision_worksheet_v1.json` 保持所有科学 decision 为空，且
`scientific_effect_values_read=false`、`qrels_read=false`、`final_report_input=false`。冻结 plan
SHA256 仍为 `07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584`，manifest
SHA256 仍为 `76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758`。

## 2026-07-19 cross-model attribution boundary

单模型 explanation ladder 之外，最终报告新增机器派生的 Q2/Q3 跨模型归因状态：

1. `no_registered_component_resolution`；
2. `single_model_registered_component_only`；
3. `model_heterogeneous_registered_components`；
4. `shared_registered_component_sufficiency_across_q2_q3`。

该状态直接从两条 primary component 决策派生，不能人工选择。即使两模型得到同一组件，固定
语义也只是两个冻结系统内都建立了注册的 component sufficiency；`generalization_beyond_q2_q3`
和 `universal_llm4rec_mechanism_claim` 始终未授权。若只定位一个模型，不得把该组件投射到另一个
未解决模型；若两模型组件不同，最终报告必须显式保留异质性。

证据 census、最终 decisions、worksheet 与 Markdown 均已接入该边界。定向测试 112 项通过，
完整仓库测试为 `702 passed, 7 subtests passed`。重新生成的 worksheet 中四个跨模型 scope 全部
列出、decision 仍为空，且未读取 qrels 或 scientific effects。冻结 plan/manifest 哈希不变。

## 2026-07-19 deterministic GPU3 short backfill

Q3 fold-0 偶数 lane 在 b26 完成后必须等待奇数 lane 完成 b25 与 b27，按已完成 bundle 的稳定
吞吐预计形成约 3.4 小时物理 GPU3 空窗。为提高四卡利用率而不改变科学设计，新增
`scripts/run_deep_dive_q3_b20_short_backfill.sh`：它只在 b26 metadata=`completed` 后启动已经冻结的
Q3-b20 attention-head observation、GQA-group intervention 与 MLP-group localization，并复用主队列
完全相同的 run ID、scorer 与 resume wrapper。主队列后续重入 completed run 会直接成功返回。

该 watcher 只读取 b26/b27 的机械 status、completed request 数和 run-contract target，不打开
scores、metrics、qrels 或 scientific effects。每个短任务启动前分别要求 b27 完成比例低于
`0.90/0.25/0.75`；attention-edge 与 RoPE 等长任务明确排除，确保 D2 fold-1 保持优先。三个任务
按 b13 实测合计约 2.2 小时，阈值留出额外时间余量。物理卡映射核验为 lane1/GPU3，watcher 已以
`CUDA_VISIBLE_DEVICES=3` 启动并等待 b26。新增 bash、未声明变量、固定 run、qrels-blind 与长任务
排除测试后，完整仓库为 `703 passed, 7 subtests passed`。

首次通过一次性 shell 的 `nohup` 启动被执行环境清理，未进入任何 b20 scorer、未产生 run 目录；
随后改用持久 exec session `42769` 启动，watcher PID `1964860` 正常停留在 b26 mechanical-status
等待。该运行时修正不改变脚本、run contract 或调度阈值。

物理 GPU2 还有第二个结果无关空窗：固定 Q3→Q2 b13 RoPE 队列结束后，Q2 fold-1/contract 预计
仍未闭合。新增 `scripts/run_deep_dive_q2_short_backfill_after_rope.sh`，只预跑 Q2 b20/b27 的
attention-head、GQA-group 和 MLP-group 短任务。每项前重新统计 15 个 Q2 fold-1 metadata：head/MLP
至少保留 2 个未闭合 bundle，attention-group 至少保留 3 个；selected-branch contract 一旦完成便
停止。长 attention-edge/RoPE 明确排除，run ID 与后续 lane2/3 主队列完全一致。

该 watcher 已在持久 session `55946`、物理 GPU2 上等待 Q2 RoPE，不与当前 scorer 竞争；PID 为
`1973293`。两个 backfill 的 bash、未声明变量、固定任务集合、机械门、qrels-blind 和长任务排除
测试均通过；完整仓库为 `704 passed, 7 subtests passed`。

运行时验证已通过：Q3 b26 fold0 以 `4082` requests、`81889` score rows、identity/result-eligibility
全部通过后封口，D2 闭合数从 29 升至 30；原 lane1 进入 `wait_all_postblocks`，没有启动 fold1。
GPU3 backfill 随后独占启动 Q3-b20 head，`235.98s` 后正常 completed，并在 b27 尚未创建时通过
`<0.25` 门启动 Q3-b20 attention-group。主队列与 backfill 没有并发 writer，证明 completed re-entry
与空窗接管路径按设计工作。

## 2026-07-19 layer scan design-value boundary

层扫描本身只选择一个无偏的组件解剖位置，并判断注册区间内的衰减更像局部转折还是分布式
过程；精确 block 编号不是架构证据。若最终只得到“block j变化”，不能据此提升任何
layer-specific adapter、跨模型推广该层号，或把它写成可迁移的设计发现。真正的设计含义必须来自
selected block内attention/MLP/residual/norm七节点的组件级干预，或来自完整注册曲线支持的分布式
衰减模式；固定`[13,20,27]` breadth仍独立运行，避免解释完全依赖fold0选择的`j`。

实现复核进一步确认了观测范围：Q2只patch每个候选的单一native Yes/No readout位置；Q3只patch
shared prompt、`P+Yes`和`P+No`三个native scoring states。层扫描没有直接patch或追踪所有history
token states，所以只能解释为历史影响在native candidate-scoring state中的累积充分性，不能称为
直接观察了history-token flow；后者由D3 attention edge/QK/V与D5固定长度控制负责。

该边界现由报告合同强制，而非只靠文字提醒。机器证据、人工decision和空白worksheet都必须固定：
`layer_scan_role=unbiased_localization_for_component_decomposition`、
`layer_scan_observed_state_scope=native_candidate_scoring_positions_only`、
`history_effect_interpretation=accumulated_state_sufficiency_not_token_path`、
`history_token_flow_directly_observed_by_layer_scan=false`、
`exact_layer_index_is_architecture_evidence=false`、
`cross_model_exact_layer_generalization_authorized=false`以及
`design_implication_requires_component_or_distributed_pattern_evidence=true`。任何人工改写或证据漂移
均fail closed；Markdown直接显示相同claim boundary。含post-block scorer语义在内的定向测试121项
通过，完整仓库测试为`711 passed, 7 subtests passed`。重新生成的worksheet保持全部科学decision为null，且
`scientific_effect_values_read=false`、`qrels_read=false`、`final_report_input=false`。冻结plan与
manifest SHA256仍分别为`07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584`和
`76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758`。

## 2026-07-19 per-component probe claim boundaries

为使“覆盖18个组件”不被误读成“18个组件都被同等强度地因果分解”，最终报告的component
coverage新增逐组件、机器固定的claim boundary，而不再只给出统一的causal/descriptive标签。18条
边界与component IDs一一对应，缺失或多余都会在导入时失败；同一边界同时进入空白worksheet和
最终Markdown。

本轮重点审计了MLP与RoPE。D4在固定`[13,20,27]`仅替换down-projection输入处的16个冻结
SwiGLU-product groups，因此只能描述product子空间和group-local score变化；它没有分别干预
`gate_proj`、`up_proj`或SiLU/product操作，`mlp_feature_formation`继续禁止标为supported。MLP主
候选瓶颈必须来自selected branch的全`mlp_down_projection` increment充分性，而D4不能挑选best
group/neuron。

D5 active条件只在一个固定block改变post-RoPE readout Q、history K或paired Q/K phase，token IDs、
mask、candidate slate、自然position IDs及其他层均不改变；它回答layer-local phase-distance是否有
因果影响，不能声称自然position ID错误、比较替代位置编码，或把某个固定block推广为通用层。
`common_offset_plus_17`因BF16重量化会破坏严格score identity，正式实现只在FP32审计共同旋转的
范数/几何，并把native Q/K原样交给backend作score no-op；报告边界显式写明这一点，不能把它当成
active科学扰动。

其余边界也分别限制Q/K与V到注册readout edges、attention/MLP output到null-context sufficiency、
residual到非additive且有incoming-state guard、norm到selected-position状态而非operator necessity、
layer scan到candidate-scoring state而非history-token flow，以及embedding/optimizer/LoRA等
descriptive-only组件。报告/overview定向测试127项、RoPE机械测试17项通过；重新生成的18组件
worksheet全部boundary非空、全部decision仍为null，且未读取scientific effects或qrels。

## 2026-07-19 full layer-scan shape reporting

复核最终report builder时发现一个与“精确层号不是设计证据”直接冲突的遗漏：D2 synthesis虽完整
保存Q2/Q3、两个endpoint的15个all-layer cells与14个adjacent-layer cells，最终Markdown此前只
渲染selected block内的七节点相邻变化，没有展示完整层曲线，因而无法让读者区分单点转折与
分布式衰减。

新增`layerwise_attenuation_profile`，从closeout绑定SHA的`d2_postblock`唯一读取全部60个
all-layer rows与56个adjacent-layer rows，不做best-layer筛选。每个Q2/Q3×target-margin/NDCG组合
使用预先固定的BH-significant adjacent-step taxonomy：`gate_stopped_or_missing`、
`no_registered_significant_adjacent_change`、`localized_single_attenuation_step`、
`distributed_multi_step_attenuation`、`mixed_attenuation_and_amplification`或
`amplification_only_no_attenuation`。至少两个显著负相邻步才设置
`distributed_attenuation_pattern_established=true`；混合模式保留正负步计数，不伪装成单调衰减。

最终Markdown新增shape summary、15层full-state sufficiency表和14步adjacent-change表。所有行固定
`exact_layer_index_is_architecture_evidence=false`且
`used_as_primary_component_attribution=false`；分布式模式最多提供跨层设计候选，单点层号仍必须回到
七节点组件证据。合成fixture覆盖60/56精确行数、Q2混合分布式模式、无变化和Q3 gate-stop，report
builder定向测试16项通过。

完整仓库回归为`716 passed, 7 subtests passed`，冻结plan/manifest哈希保持不变。此报告增强只定义
结果完整呈现与解释边界，没有读取未闭合family的scientific effects，也没有改变layer selection、
family、阈值或正在运行的scorer。

## 2026-07-19 attention edge causal-level separation

D3实现与报告路线再次按功能层级审计。`history_logits_mask`只改注册readout query到history keys的
softmax logits并重新归一化，因而只可支持Q/K routing；`history_value_edge_zero`在原概率下减去
history value contribution、不重新归一化，只可支持V transport；aggregate attention output仍必须
由selected-block `attention_o_projection`充分性干预支持，不能借用任一上游edge结果。合同已有
V不能借Q/K的门，本次补齐Q/K不能借V的对称测试。

formal edge scorer会保存全人口score identity，但其manual selected-row/native-backend误差摘要没有
逐bundle持久化；不能据score identity单独升级edge组件。固定512-row D3 head observation使用同一
Q/K/V selected-row重组并在每个model/block执行dtype-aware low-precision ratio门。因此新增机械
依赖：任何以`d3_attention_edges`把`attention_query_key_routing`、`attention_value_transport`或
`history_routing`标为supported的组件行，必须同时引用闭合的`d3_attention_heads`重组证据；缺失时
fail closed并保持unresolved。该依赖显示在最终component probe coverage中，不修改正在运行的
scorer、family或effect。attention/report定向测试131项通过，完整仓库为
`714 passed, 7 subtests passed`；冻结plan/manifest SHA不变。

## 2026-07-19 selected-branch composition audit and incoming-state guard

针对层扫描是否真正服务于后续模块分解，逐路径复核了selected block七节点实现。普通节点在显式
native scoring positions替换对应absolute state；`post_attention_residual`不能用普通pre-hook，
而是同时捕获recipient block input `r_N`、在attention output写入`u_F-r_N`、强制MLP norm输入为
`u_F`，并将block output重组为`u_F+MLP(u_F)`。新增直接observer测试，不再只用最终logit变化作
间接证据：实际post-attention state逐元素等于目标`u_F`，实际block output逐元素等于目标加本次
MLP increment；full→full与null→null identity仍为逐元素精确。selected-branch、runtime、evaluator
与instrumentation定向测试35项通过，未修改已冻结的scorer或implementation digest。

报告判定同时补上incoming-state混淆保护。此前若attention/MLP都不满足、而
`post_attention_residual`或`block_output_residual`满足六个充分性gate，可能被机械标为
`residual_composition`；但若`block_input_residual`本身已经满足相同gate，该现象只能证明有害状态
从上游进入selected block，不能归因给当前block的residual或norm。现在primary derivation显式读取
block-input support：incoming state满足时，residual/norm criterion均禁止通过，并记录
`incoming_state_confounds_residual_or_norm_attribution=true`，最终保持`unresolved`；attention或MLP
increment若自身独立满足gate，仍可按null-context sufficiency解释为候选瓶颈，但不升级为唯一来源。

证据census和Markdown新增incoming-state列，空白worksheet固定该防过度归因规则。报告定向测试
123项通过，完整仓库测试为`715 passed, 7 subtests passed`。重新生成的worksheet仍为
`todo_not_final_report_input`，全部decision为null；冻结plan/manifest SHA256仍分别为
`07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584`和
`76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758`。

对等价helper继续作malformed-input red team：现在拒绝重复/缺失fold、非精确三行fold结构、倒置CI、
Q2/Q3 readout deliverable与模型错配，以及D7精确12-row family之外的额外行。因而只有结构完整、
route绑定正确且CI上下界有效的结果才能触发weakened；完整仓库回归更新为
`730 passed, 7 subtests passed`。

最终报告进一步新增逐模型`component_practical_equivalence_gate_outcomes`：四条注册route会在19个
deliverable全部closeout后逐一读取SHA绑定的metrics，只输出Boolean gate outcome和source SHA，不复制
raw effect。完整route共有5个model-specific outcome；`not_established`显式不作为non-equivalence或
相反效应证据。这样component decision的机器校验、Markdown可见表和底层metrics三者使用同一helper，
避免人工摘要与合同判定漂移。

四卡调度最终把Q3 lane-1等待窗口优先用于D2关键路径，而不是再跑一个广度MLP：b20 MLP-group完成且
主Q3 b27 fold-0机械进度仍低于0.65时，GPU3只对固定Q2 b27 fold-1作一次2,100秒bounded attempt；
它使用与主resume queue相同的canonical metadata lock，最多留下可恢复的`wall_time_exhausted`状态，
不循环第二次、不读取qrels/effect，主Q2队列以后续跑同一atomic bundle。随后Q3 b27仍低于0.90才跑
约四分钟的固定head observation。实测约100分钟的attention-group、长edge、RoPE和b27 MLP均排除，
保证Q3 fold-1优先。队列与报告新增测试后完整仓库为`732 passed, 7 subtests passed`。

## 2026-07-19 paired normalization-boundary guard

继续审计native readout与七节点归因后，发现仅凭`input_rmsnorm_output`或
`post_attention_rmsnorm_output`状态补丁满足充分性gate，仍不足以把候选瓶颈命名为
normalization：若配对的pre-norm状态已经满足同一gate，post-norm支持只是上游有效状态被携带通过
该边界，而不是效果在RMSNorm边界新出现。

primary derivation现在固定两对边界：`block_input_residual -> input_rmsnorm_output`与
`post_attention_residual -> post_attention_rmsnorm_output`。只有post-norm节点满足六个fold-1
target-margin gate、配对pre-norm节点不满足、attention/MLP均不足且incoming block state不满足时，
才允许`residual_norm_interaction`。若post-attention residual及其post-norm状态都满足，则记录
`normalization_state_support_without_boundary_isolation=true`，不把RMSNorm置于优先位置；在其余
residual条件成立时归为residual composition，否则保持unresolved。NDCG corroboration也必须在同一
配对边界上隔离，不能从任意norm节点拼接。

这一规则仍是fail-closed的边界局部化充分性，不把“配对pre-norm未过注册门”解释为统计等价或
RMSNorm算子必要性。报告claim boundary明确保留operator necessity与all-token normalization未测试；
Q3 native-readout边界也明确teacher-forced term substitution不覆盖autoregressive generated-token
feedback。空白worksheet固定配对保护，Markdown显示isolated与unisolated norm支持。报告、worksheet
与native-readout定向测试136项通过，完整仓库为`717 passed, 7 subtests passed`；未读取未闭合family
科学效果、qrels或source test，且未修改正在运行的scorer。

层扫描的后续用途也改为机器显示的确定性映射，而不依赖报告撰写者临时讲故事：单个显著衰减步只
触发split-sample selected transition的七节点分解；两个以上衰减步触发跨层传播形态解释且禁止精确
层号设计；正负混合同时保留局部分解与全层形态；无显著相邻变化转向固定breadth/readout检查；仅
放大不允许推断attenuation bottleneck；gate-stop不允许任何层结论。所有shape均固定
`layer_scan_alone_authorizes_design=false`。对应report定向测试125项通过。

随后检查18组件matrix的result-level route，发现primary attribution虽然已有上述保护，component
matrix仍可由任意residual/norm节点单独升级为supported，形成旁路。现已统一两套规则：
`residual_composition`只有在attention/MLP和incoming block state均不足、有residual节点支持且没有
isolated norm boundary时才能supported；`normalization`只有在attention/MLP和incoming state不足、
post-norm支持且配对pre-norm不支持时才能supported。因而component matrix、primary label及由其约束
的机会排序不能互相矛盾，也不能用一个普通被携带的state绕过组件分解。合成测试覆盖incoming carry、
attention解释、isolated norm、pre/post norm同时支持及norm+attention五类冲突，report定向测试126项
通过，完整仓库为`718 passed, 7 subtests passed`。

为防止最终机会排序重新把localizer误当设计结论，closeout boundary assertions新增两条全局硬门：
`exact_layer_index_used_as_architecture_design_parameter=false`与
`layer_scan_alone_used_to_rank_architecture_opportunity=false`。因此即使完整层曲线或单个transition显著，
绝对层号也不能进入架构参数，层扫描也不能单独把任何冻结opportunity升级为primary；仍需组件或
分布式形态对应的独立证据组。合同、builder与worksheet定向测试128项通过。
完整仓库回归为`720 passed, 7 subtests passed`。

同一边界扩展到用户指出的数据集/模型规模不稳定性：新增
`layer_shape_generalized_beyond_frozen_models_or_dataset=false`，禁止把本次Q2/Q3形态外推为跨数据集或
跨模型规模规律。另加静态审计，冻结architecture opportunity catalog不得包含`layer/block + 数字`
形式的绝对层设计参数；报告侧定向测试130项通过。

七节点解释继续收紧到实验真正回答的范围。selected-node patches能证明某个full-context状态在null
recipient中足以复现harm，但adjacent-node family没有预注册方向性support gate，不能由“前节点未过、
后节点过门”升级成已确认的擦除算子。primary intervention scope新增
`within_block_adjacent_change_role=descriptive_only_without_registered_directional_gate`与
`component_erasure_boundary_established=false`；final Markdown直接显示这两项。任何把它改成
confirmatory erasure、necessity、exclusive origin或Shapley贡献的decision都会fail closed。报告定向
测试132项、完整仓库`724 passed, 7 subtests passed`。

最终人工decision的自由文本边界也被封闭：18个component row的`claim_boundary`必须逐字等于各自
机器注册的`COMPONENT_PROBE_CLAIM_BOUNDARIES`，Q2/Q3 primary attribution则必须逐字等于固定
null-context sufficiency boundary。finding、rationale和optimization implication仍可按结果填写，
但不能用自由文本把partial causal scope扩成唯一原因。新增正反例验证两类boundary drift均失败，
完整仓库为`725 passed, 7 subtests passed`。

负证据解释同样完成结构化保护。component gate的
`registered_support_not_established`现在明确既不是opposite-effect，也不能自动作为component
weakened或hypothesis rejected证据；全局断言禁止用`p>0.05`或missing support完成削弱/拒绝。
component与H0--H5 decision新增`negative_evidence_basis`，允许值只包括注册practical equivalence、
显著反方向、独立counterexample、mixed registered evidence或not-applicable；weakened禁止
not-applicable，rejected进一步禁止仅用mixed evidence。最终Markdown逐行显示basis，空白worksheet
同时列出允许值。这样target margin无SESOI时不能把跨零区间写成“没有机制”。定向测试135项，完整
仓库`727 passed, 7 subtests passed`。

`registered_practical_equivalence`随后由结构标签升级为result-level机器门，避免人工把“不显著”填成
等价。当前只允许四条预注册路线：D5 RoPE、Q2/Q3 native readout与D7 Q2 objective。RoPE必须完整
覆盖blocks 13/20/27和readout-Q/history-K/paired-QK九个cell，且每个cell的NDCG
compression-minus-expansion与compression-minus-baseline全人口CI都完全落入`±0.005`；Q2/Q3
readout必须分别完整覆盖2/4个scope，并要求same-minus-null和same-minus-cross的NDCG全人口CI均
落入`±0.005`；D7必须有精确12个family key，六个RankNet/ListNet cosine row均通过`±0.1` SESOI
等价门。所有路线同时要求all/fold0/fold1结构完整且有限，但按冻结协议只有all-population CI判等价。

最终报告与空白worksheet新增完整practical-equivalence gate catalog，明确
`registered_support_not_established`或`p>0.05`不构成等价。worksheet已重新生成，四条路线可审计，
全部scientific decision仍为null，且未读取effect、qrels或source test。报告定向测试137项通过，完整
仓库为`729 passed, 7 subtests passed`；冻结plan/manifest SHA256继续保持
`07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584`和
`76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758`。

native-readout等价route又与真实闭合Q2 metrics作了structure-only兼容审计：两个scope都精确含
same-minus-null/full/cross三种comparison，每种NDCG均有all/fold0/fold1三行。helper现在要求这三个
comparison结构精确同集；same-minus-full只作完整、有限、CI顺序有效检查，不参与`±0.005`等价判定，
same-minus-null与same-minus-cross继续执行全人口CI SESOI门。该审计只读取key、row count与fold label，
未读取mean或CI值。

selected-block七节点在formal scoring前完成直接observer审计。六个普通patch节点逐一验证目标向量
精确出现在注册module input/output边界，并验证同一边界所有未选token逐元素保持baseline；特殊
post-attention residual除`state=desired`与`block_output=desired+MLP(desired)`外，也新增norm-input和
block-output未选token逐元素不变检查。因而七节点不再只依赖identity logits或最终分数作间接验证。
完整仓库回归为`738 passed, 7 subtests passed`；正式scorer与implementation digest未修改。

D3--D5深层intervention也补齐非identity locality observer。MLP-group在down-proj input验证只替换
selected token的注册SwiGLU columns，同token其余columns及其他token保持baseline；attention edge在
o-proj input验证logits-mask与value-zero只改变注册readout query rows；GQA-group进一步按16×2真实
head reshape验证只改变目标KV group对应的两个query heads；RoPE直接截取backend收到的post-RoPE Q/K，
分别证明readout-Q只改注册query行、history-K只改history span、paired-QK只改二者，所有互补位置
逐元素不变。对应compression模式的Q/K norm ratio仍通过冻结dtype-aware机械门。

这些测试不把head/group localization升级为主要机制，只证明正式干预边界没有越界。全仓库回归更新为
`746 passed, 7 subtests passed`，未改变任何已运行scorer source、family、阈值或选择记录。

Q3 native-readout随后补齐了term substitution与真实final-RMSNorm output state substitution之间的
逐作用域等价审计。shared-prompt、Yes-context、No-context与joint四个注册scope分别直接替换donor
最终norm状态，经冻结lm-head重新计算四个原生log-prob项；所得terms与score逐值等于公开compose
路径。该测试把D6严格限制为最终读出分解，不能借term代数把结论外推到更早Transformer block。

D7代码审计同时发现Q3 LoRA path row中原名`post_step501_delta_w_norm`的量实际是更新后完整有效
权重`2(B+deltaB)(A+deltaA)`的norm，而非本步delta-W。正式D7尚未生成，故在结果产生前将其拆成
`step500_effective_weight_norm`、`post_step501_effective_weight_norm`与
`step501_effective_delta_norm`；最后一项与joint function-delta norm直接手算一致。optimizer/readout
相关定向测试共15项与13项分别通过。结果盲decision worksheet重新生成，全部component、primary、
hypothesis与opportunity decision仍为null，effect/qrels/source-test读取标志均为false。

Q0/Q1 breadth的cache/readout边界也补充真实模型审计。Q1使用28-block tiny Qwen、两个不同长度候选
response，验证prefix cache与包含padding的全部teacher-forced continuation token：final-RMSNorm input/
output两个边界以及block 13的input-residual、attention-o、MLP-down、output-residual四个内部节点均能
逐值identity replay，donor continuation覆盖与实际forward call一一对应。该检查只证明Q1干预在其原生
多token listwise路径上机械等价，不把Q0/Q1 breadth结果提升为Q2/Q3主要transfer机制证据。

修正后的LoRA已有权重、本步后权重与step-501有效delta norm均进入D7描述性聚合，不新增统计门；
本轮全仓库回归为`752 passed, 7 subtests passed`。冻结plan/manifest哈希保持不变，diff whitespace
检查通过。

attention edge语义又增加手算边界：在等logit、values=`[1,4,10]`且中间token为history时，
`history_logits_mask`精确得到排除后重归一化均值`5.5`，`history_value_edge_zero`精确得到保持原
`1/3`概率但移除history-V贡献的`11/3`。这直接固定Q/K-routing与V-transport两条route的不同算子
含义，不能互借结果。新增测试后全仓库为`754 passed, 7 subtests passed`，冻结哈希不变。

GPU3限时critical-path backfill在`2100.476s`按注册上限退出，Q2 fold-1 block-27保留1807个请求并
标记`wall_time_exhausted/resumable`，未循环重启或覆盖；同一有界脚本随后按预案交接到Q3 block-27
attention-head observation，交接后四卡重新达到高利用率。这里只记录机械状态，不读取两者科学效果。

D5 contextual attention-null又增加真实Qwen位置审计：full mask与只将内部history key span置零的mask
两次forward送入model-level RoPE的position IDs逐元素一致，均为默认共享`arange(S)`；因此内部零段
不会通过attention-mask cumsum压缩后续query/candidate位置。该测试只验证机械隔离，不读取D5效果。

Q3 block-27 attention-head observation闭合为512/512后，原有有界lane无后继任务。新增GPU3
qrels-blind utilization queue，仅消费manifest已注册的Q2固定blocks 20/27，顺序为attention-head、
GQA-group、MLP-group；全部经canonical resume lock，与post-RoPE backfill可安全去重，不读取metrics、
qrels或selected block。Q2 block-20 head随后闭合512/512并自动交接block-27。queue/context定向测试
12项通过，全仓库回归为`756 passed, 7 subtests passed`，冻结plan/manifest哈希不变。

审查D6输出完整性时另记一项报告待办：Q2/Q3 scorer原始bundle已保存yes/no common offset、原生term
分量与final-norm几何，但当前主evaluator只汇总排序contrast与request common/relative score分解。
这些诊断不改变任何注册因果或等价门；在Q3 native-readout家族闭合前不读取数值，最终综合时应作为
qrels-blind描述性附表汇总，不能事后追加显著性筛选。

D3 attention-head六个固定model×block bundle随后全部闭合，统一evaluator integrity通过且全程
`qrels_read=false`。新增固定浓度综合同时保留全部8个GQA group与16个query head，不据top结果追加
实验。native-readout→history的top-head mass share跨六cell为约`0.124--0.308`，Simpson有效head数
约`4.51--10.75/16`；top-group share约`0.244--0.321`，有效group数约`3.93--6.07/8`，不支持
单一head/group垄断解释。

具体top identity也不稳定：Q2三个anchor的mass top group为`3/7/3`，Q3为`2/5/6`；query-head
mass top分别为Q2 `4/14/13`、Q3 `4/11/13`。最晚固定anchor两个模型的top-3 head与group集合
均完全重合，但更早anchor的top-3 Jaccard只有`0.2--0.5`，且该观察只限同一base规模、同一数据集
下Q2/Q3，不能泛化为精确层或跨规模规律。

attention mass与真实o-proj contribution norm在中/晚anchor的相关约`0.965--0.981`，但Q3早期
query-head/GQA仅约`0.517/0.667`；六cell中group-level mass与contribution top均一致，而head-level
只有三cell一致。因此raw attention weight在部分早期路径不是可靠贡献proxy。当前证据更符合
分布式、阶段依赖的路由形态；它仍是描述性localization，必须等待GQA-group和full-edge因果家族，
不能据此选择head、block或架构。

attention pattern综合的三个手算/coverage边界测试及全仓库回归通过，当前为
`759 passed, 7 subtests passed`；GPU3已进入Q2 block-20 GQA-group正式包，四卡采样时均为
`96--100%`利用率，冻结plan/manifest哈希继续不变。

## 2026-07-19 Q/K stage geometry and full/null position-confound audit

D3 attention-head完整六cell闭合后，新增一个不读qrels的固定网格Q/K几何综合；它覆盖Q2/Q3、
blocks `[13,20,27]`、Q/K、pre-norm/post-norm/post-RoPE、全部注册语义位置和全部head。初版将Q3
Yes/No native path平均，复核发现这可能隐藏path异质性，因此保留v1不覆盖，并在v2同时输出6个
model×block汇总cell和9个逐path cell。正式v2为
`runs/20260719_kuaisearch_mech_d3_qk_geometry_v2/metrics.json`，SHA256为
`c3d01eceffd9d06da00ff30de5a595e4866c00ac579e9a4c9af5a0e940478cf3`，明确
`descriptive_only=true`、`qrels_read=false`。

RoPE前后每个full/null Q或K向量的norm保持到最大相对L2误差`3.996e-4`，符合BF16旋转的机械预期；
但逐path的66个tensor×position比较中，post-RoPE相对full/null delta全部增加、cosine全部下降。
该一致性本身不能解释为偏好内容被RoPE放大或损坏，因为full与null同一语义位置可能使用不同绝对
position ID。按位置拆开后，query-end的平均relative-delta变化仅约`8.77e-5`、cosine变化约
`-7.02e-7`；history-summary平均分别约`+0.237/-0.170`，两个native readout约
`+0.552/-0.214`与`+0.552/-0.221`。因此原始post-RoPE full/null几何明显混合content与位置相位。

为直接核验这一解释，另用冻结512 candidate rows、同一Q2/Q3 tokenizer和正式prompt构造重建
full/null实际padded-sequence indices；没有加载模型权重、没有读取模型分数或qrels。正式输出为
`runs/20260719_kuaisearch_mech_d3_position_shift_audit_v1/metrics.json`，SHA256为
`97bec9c4c793130ecaa19e797c249d20b42355f4d0dd151ffa7f80280609194a`。Q2 prompt与Q3 Yes/No三条
path的结果完全相同：512/512 query-end绝对位置不变；512/512 history-summary和native readout
均后移；readout位移与full-minus-null序列长度差逐行完全相等。平均序列/readout后移
`299.455` tokens，中位`332`，范围`31--447`；history-summary恒少1 token，平均`298.455`。

这建立的是一个先前未显式量化的**测量混杂**：默认Qwen forward没有显式`position_ids`，使用
`arange(padded_sequence_length)`，所以删除历史不仅改变内容，也将history之后的K/Q readout旋转到
不同相位。它不建立RoPE是transfer失败原因，也不支持改位置编码；D5固定长度content-neutral、
attention-null和active compression/expansion仍是必要因果证据。Q/K与位置分析合计10个定向测试
通过，完整仓库回归更新为`763 passed, 7 subtests passed`；冻结plan/manifest哈希仍分别为
`07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584`与
`76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758`，diff whitespace检查通过。

同期机械进度只作结果盲登记：D2固定60 bundles中34 completed、2 in-flight，request-weighted执行
约`58.74%`；Q2 fold-1 b17为`3072/3918`，Q2 b27保留`1807/3918`可续跑partial。Q3 fold-0 b27
已完整闭合；Q3 RoPE b13 v2与Q2 b20 GQA-group继续运行。四卡均保持高利用率，未读取未闭合family
的科学效应。

Q3 fold-0全层闭合并自动冻结selection后，两条fold-1 lane按注册顺序启动。此时审计发现物理GPU3
仍有一个先前空窗启动的Q2 b20 GQA-group backfill，与新Q3 fold-1 scorer短暂并发；输出目录彼此
独立，未发生双writer或科学污染，但会拖慢关键路径。先停止低优先级backfill，保留其`196/512`
partial与SHA/progress供canonical resume，GPU3恢复单scorer。opportunistic GPU3脚本新增fail-fast门：
Q3 selection存在后直接yield，由后续注册main breadth lanes接管Q2 partial，避免人工误重启再次争卡。
queue topology 9项测试通过。

D6还补上一个只准备、不提前读Q3结果的native-readout诊断附表。Q2 scorer保存的full/null
`common_offset=(Yes+No)/2`被Yes-minus-No分数精确抵消；Q3保存的四个log-prob term可精确拆成
shared-prompt contrast、continuation contrast及各自common mode。新分析器只有在Q2/Q3两个bundle
均`completed`、全8000 requests/160753 candidate rows完整、scores SHA匹配且qrels/source-test均false
时才运行；它同时汇总final-RMSNorm input/output norm geometry，但不把末端代数描述提升为更早层
机制或ranking utility。两个手算/fail-closed测试与既有readout测试合计10项通过；watcher已等待Q3
正式bundle，当前未读取其数值。全仓库回归更新为`765 passed, 7 subtests passed`，冻结哈希与
diff whitespace检查继续通过。

该时点四卡均为单writer且利用率`96--99%`：Q2 fold-1 b17为`3679/3918`，Q3两条新fold-1 bundle
分别为`234/3918`与`208/3918`，Q3 RoPE继续独占另一卡。D2固定bundle仍为34 completed，计入所有
partial后的request-weighted执行约`59.19%`；这是机械进度，不包含任何未闭合效应。

## 2026-07-19 MLP feature-formation exploratory extension

逐组件coverage复核确认D4现有正式任务只在`down_proj`输入抓取/patch最终
`SiLU(gate_proj(x))*up_proj(x)` product；它不能区分gate preactivation、SiLU后的gate、up projection
或乘法交互。因此新增一个不改冻结确认family的固定网格描述性扩展，仍使用同一qrels/score-blind
512 candidate rows、Q2/Q3和blocks `[13,20,27]`，明确
`confirmatory_family_member=false`且禁止layer/group/neuron选择。

底层observer在同一forward的注册语义位置同时抓取`gate_pre`、`gate_activated`、`up`和实际
`product`。对每个冻结维度组，将未量化因子的product变化精确分解为：
`delta_gate*null_up + null_gate*delta_up + delta_gate*delta_up`；三项保留norm及与product delta的夹角。
这只是包含interaction的代数恒等式，不解释为可加因果贡献。实际BF16 product与FP32因子重算的
量化残差单独报告，并按原生dtype的`4*eps`门验证；不再错误要求`1e-5`字面相等。

扩展现包含可恢复512-row runtime、no-op原生score identity、product recomposition门、六bundle统一
evaluator和完整性合同。evaluator要求六cell同一implementation digest、每cell 512/512、Q2 prompt与
Q3 Yes/No全部path、Q2三个与Q3四个语义位置、每位置16个完整group，任一缺失即失败。真实28-block
tiny Qwen hook、手算delta decomposition、BF16边界、synthetic six-cell evaluator和queue topology
共15项定向测试通过。

为不占用当前D2--D7关键路径，六个formal bundle分成四条末班车lane：各自等待物理GPU当前注册主队列
的最后任务完成，再在原卡先跑1-row smoke并执行固定blocks；第5个watcher等待六包闭合后统一汇总。
四lane、MLP evaluator与D6 readout附表共6个watcher均已存活且未触发，上游metadata此时均不存在，
GPU仍只有原四个scorer。全仓库回归更新为`771 passed, 7 subtests passed`，冻结plan/manifest哈希和
diff whitespace检查继续通过。

运行期间Q2 fold-1 b17完整闭合并自动进入b18，D2 fixed completed从34升到35，resolution为
`35/62=56.45%`，request-weighted执行约`59.76%`；Q2 b18为`500/3918`，Q3两条fold-1为
`543/3918`和`503/3918`。这些仍仅是结果盲机械状态。

Q/K几何综合随后升级为v3，把pre-norm→post-norm与post-norm→post-RoPE的逐path一致性计数直接写入
机器输出，避免只保留临时查询。正式文件为
`runs/20260719_kuaisearch_mech_d3_qk_geometry_v3/metrics.json`，SHA256为
`4f19c4eedd221da342b4e340512e719c634b3eca84518d1ba07e23aaabe1a6d6`。完整66个
path×tensor×position比较显示Q/K RMSNorm不是统一的“保留/抹除”算子：K的relative full/null delta
在31/33 cell下降，另2个仅为Q3早期Yes/No query-end的`+6.12e-5`微小变化；Q在早/中两个固定anchor
22/22上升，在晚anchor 11/11下降。对应cosine方向计数与之大体镜像。

该模式跨Q2 prompt与Q3 Yes/No path、所有注册语义位置成立到上述边界，但仍只是每cell的512-row均值
几何，不是逐请求显著性、score mediation或精确层推广。它说明后续若attention routing被因果支持，
Q/K normalization应按K-vs-Q及stage分别审视，不能提出一个通用“加/去norm”修复。RoPE阶段的66/66
delta增加、66/66 cosine下降也已机器固定，但与独立position audit共同解释为content+phase混杂，
必须等待D5因果门。

v3与新增MLP/readout扩展后完整仓库仍为`771 passed, 7 subtests passed`；冻结哈希、v3与position
audit输出SHA及diff whitespace均通过。D2 completed保持35，request-weighted执行在快照时达到约
`60.00%`；Q2 b18为`803/3918`，Q3两条fold-1为`666/3918`和`643/3918`，四卡继续原主线。

## 2026-07-19 component-state necessity extension

层扫描用途收紧后，重新审计selected-branch的干预方向。冻结D2只把full-context节点状态写入
null recipient，回答null-context中的状态充分性；报告合同也固定`necessity_tested=false`。这留下
一个与架构设计直接相关的非对称：某节点能复现harm，不等于在full inference中移除它就能消除harm。
若没有反向移除，层扫描后的七节点分解仍可能只找到一个可替代的state carrier，而不是必须处理的
中介。

因此在未读取未闭合D2 family效应、source test保持关闭时，独立冻结
`transformer_component_necessity_extension_plan.md`与manifest；SHA256分别为
`124beb7a19a8d22a5d5ddb9e87d67c868c038b2250027c6822382c2985c7fb40`和
`540f626d83e280f6a43ee781c7e8d59f9e5458cf60c45a5a1d3d042ec929f900`。它不修改原deep-dive
plan/manifest、family或implementation digest，只在原D2 fold-0选择、fold-1确认且parent selected
branch完整后运行。

扩展固定四个功能节点：incoming `block_input_residual`、attention increment、MLP increment与完整
`block_output_residual` ceiling。每个节点只增加full→full identity与same-request
null→full removal；Q2改native candidate readout position，Q3同时改shared prompt、Yes context与
No context。RMSNorm输出和post-attention residual没有被伪装成operator bypass：最高claim仍只是
“component state是full-context harm的必要中介”，不授权operator necessity、唯一来源、精确层设计
或跨模型规模推广。

统计固定为Q2/Q3×4节点×target-margin/NDCG共16 units，按endpoint分成两个8-unit BH family；
gate-stop或mechanical missing保持`p=1`。支持还必须与原D2对应节点的sufficiency和history-specific
negative control同时成立；单独removal通过不能升级架构。scorer qrels-blind，evaluator只有在完整
score/identity/parent SHA审计后读取fold-1 qrels。

独立scoring primitive、可续跑runtime、共享evaluator、CLI、队列与fail-closed tests已实现。
scorer强制所有removal使用full recipient；runtime强制parent selected bundle完整且contract确认为
registered transition；evaluator保留gate-stop family size并把positive-removal与NDCG practical
equivalence分开。组件、parent/scalar/report合同与queue定向回归共135项通过；新增模块与queue合计
22项定向测试通过，bash语法、CLI、diff whitespace及四个冻结hash均通过。

两条GPU lane与一个CPU evaluator queue已启动但只在parent selected branch和各自MLP formation
末班车释放后触发；当前四卡仍只有原四个scorer，未产生竞争。机械快照为D2固定35/60 completed、
四个in-flight、request-weighted约`60.18%`；完整仓库回归为`783 passed, 7 subtests passed`，
未读取任何未闭合family科学效应。

### Position-preserving V2 correction before execution

在两条necessity lane仍只等待contract、尚无任何score目录或扩展结果时，进一步把D3位置审计应用到
反向移除设计：V1的`null_to_full_removal` donor来自删除历史后的短序列，虽然recipient是full，donor
state本身仍混合history content与不同自然position/phase，不能作为主要设计证据。因此停止三个V1
waiter；检查确认没有V1 bundle启动、没有qrels/effect读取后，冻结V2并保留V1字节作为superseded
先验记录。

V2 plan与manifest SHA256分别为
`866b04d3b072fb37e1374db754efe80056fc39d067ce9d58de47b9b2cf95c614`和
`6b784682239e5ce8e6f1c37fed1c648658912ed03801750fe99d256b0777c0e3`。V2复用deep-dive manifest已经
冻结的7,254-request content-neutral eligibility及Q2/Q3 row SHA，在完整full prompt中只把注册history
span token IDs替换为`151643`；shape、mask、padding、readout indices、span外tokens与默认自然position
全部保持。每个节点新增`neutral_to_full_removal`作为主要position-preserving necessity条件，原
`null_to_full_removal`降为position-confounded sensitivity。

family相应固定为每endpoint `2 donor modes × 2 models × 4 nodes=16`，两endpoint共32 units；neutral
只在strict-transfer与冻结eligible交集推断，null使用全部strict-transfer。只有neutral removal通过并
与parent sufficiency/history-specific gate一致时才可影响设计排序，null单独通过不授权支持。

scorer新增逐batch full/control-builder identity与neutral span内外逐元素审计，不合格request的neutral
score精确复制full identity并由evaluator排除；runtime绑定content-control manifest/row SHA与eligible
count；evaluator保存两个donor mode并分别标识primary与sensitivity。V2 scorer/runtime/evaluator定向
测试12项、相关组件/队列/报告合同回归157项通过；完整仓库为`785 passed, 7 subtests passed`。
V2两条GPU等待lane与统一evaluator queue已重新挂载，仍不抢占当前四卡。

另在不加载模型权重、不读取qrels/score的条件下，对Q2/Q3各16个稳定哈希fold-1 eligible requests
直接重建正式full builder与content-control builder；Q2的16条prompt path、Q3的32条Yes/No path均
通过ids/mask/native-position逐元素identity，neutral后全部span外token、mask、shape与position保持，
span内恰为`151643`。输入row SHA分别为`c38002...3cd12f`与`9c9e14...983093`，与冻结manifest一致。

### Pre-outcome component design-gate synthesis

在component-necessity family与parent selected-branch family均未闭合、未读取其效应时，把V2计划中
“removal还必须与parent sufficiency和history specificity一致”的文字规则实现为fail-closed机器综合。
它不增加统计family、不重选block，也不读取qrels或score bundle；输入仅允许两个完整共享evaluator
输出，并逐字节反查necessity bundle metadata、selected-branch evaluator input与两者共同的parent
selected-branch metadata/scores SHA。任一模型gate-stop保留为不支持，不能缩小到另一模型。

每个Q2/Q3×四功能节点×endpoint的注册state-mediator门固定为三项全真：等长neutral→full反向移除
主要门、原same-request full→null充分性门、same-minus-wrong-history用户特异性门。另把原D2更严格
的same-minus-cross stress及norm/direction/random三类结构负控冻结为**设计优先级附加门**：最低三门
通过只称history-specific state mediator，全部六个parent gates再加neutral necessity通过才进入设计
排序。null donor只保留位置混杂sensitivity；NDCG门与target-margin主门分开。综合按功能节点输出
incoming-state、attention-output、MLP-output与block-output ceiling的model-scoped解释；只有Q2/Q3
同一功能节点共同通过全部门，才标记为可改变component-path设计排序。输出合同显式禁止operator
necessity、唯一来源、直接history-token flow、精确层设计参数及跨规模/数据集推广。

新增`component_design_synthesis.py`、CLI、CPU等待队列及四类定向测试：全部门共同通过的正例、parent
bytes不一致硬失败、null sensitivity单独通过仍不授权、最低state门通过但结构负控不全时不得进入
设计优先级。与necessity/selected/queue相关回归共24项
通过，bash语法通过。实现时D2仍为35/60完整包，四个主scorer占用四卡，未出现新闭合family或科学
效应读取。随后完整仓库回归为`789 passed, 7 subtests passed`；四个冻结plan/manifest哈希与既有值
逐字一致，diff whitespace检查通过。CPU综合watcher已以独立进程等待两个完整上游，不占GPU。

最终deep-dive报告合同也同步收紧但不修改19个冻结deliverable或其统计：D2 primary attribution继续
原样报告“null-context sufficiency localization”，新增硬边界断言禁止以selected-branch sufficiency
单独排序架构；attention/MLP/residual组件边界明确要求另行通过position-preserving reverse-removal
overlay。综合、报告合同、报告构建与decision worksheet相关回归144项通过。

上述结构负控分层与报告边界收紧后的完整仓库回归为`791 passed, 7 subtests passed`；冻结哈希及
diff whitespace再次通过。期间Q2 fold-1 block-18闭合并自动进入block-19，D2固定包更新为36/60，
计入partial后的request-weighted执行约62.72%；Q3两个fold-1包约53%，Q3 RoPE replacement继续运行。
这些只更新机械进度，未读取未闭合family效应。

## 2026-07-19 exhaustive supplemental-evidence registry

为避免最终报告只呈现19个formal deliverable而遗漏已经完成的内部几何/分解，或反过来只挑显眼的
附加分析，建立一个穷尽式supplement registry。它登记21项：17个已经存在的retrospective inventory
全部冻结path、analysis type与SHA，另4项在输出前冻结为pending——MLP gate/up/product formation、
Q2/Q3 native-readout diagnostics、position-preserving component necessity V2与最终functional design
gate synthesis。Q/K v1/v2只作为v3的superseded lineage保留，不重复算作证据。

注册表明确区分retrospective descriptive inventory与pre-output extension；既有描述性输出不能因被纳入
报告而升级confirmatory claim。21项中只有`component_functional_design_gate_synthesis`有资格改变组件
设计排序，且仍禁止绝对层号、operator necessity、单模型全局推广或把诊断patch当方法。独立manifest
绑定registry SHA `1c53204d...bede251`、四个parent plan/manifest及全面报告计划。

新增outcome-independent registry auditor与status CLI；当前审计为17 completed、4 pending、0 failures，
不输出效应、不打开qrels/source test。它会检查完成项冻结字节、未来项analysis schema、qrels边界、模型/
组件scope与唯一design authority。hash drift、design authority漂移、claim boundary漂移及真实清单测试
4项通过；与component/report/queue相关定向回归共137项通过。

同时冻结全面报告合同，要求最终报告覆盖执行总表、18组件×4模型、五层失败链、功能S/N/G因果链、
Q0--Q3边界、H0--H5、全部负结果/冲突、优化机会与最小证伪实验、claim边界及SHA附录。main 19项或
supplement 21项任一pending时只能报告进度，不能宣称最终原因或最终架构。

进一步增加统一readiness输出，把三种容易混淆的“百分比”分开：formal deliverable closure、supplement
closure与D2 request-weighted compute。当前机器快照为formal `6/19`、supplement `17/21`、合并artifact
`23/40=57.5%`（明确不是计算/功效/确定性）、D2 fixed `36/60`、D2 resolution `36/62=58.06%`，
request-weighted约`63.18%`。18个组件中16个已有至少一个完整artifact；尚无完整artifact的两个正是
`mlp_feature_formation`与`mlp_output`，其formal和supplement任务均已注册并在队列中，而不是被遗漏。
readiness与registry的6项定向测试通过；完整仓库回归为`797 passed, 7 subtests passed`，七个当前冻结
plan/manifest/registry哈希、diff whitespace与四卡单writer状态均通过。

### Fail-closed comprehensive synthesis builder

用户进一步指出绝对层号很可能随数据集或模型大小漂移，单纯定位层变化对设计没有稳定意义。该边界
已经写入正式报告合同；现在又实现为最终生成器的硬门，而不只依赖人工表述。新增
`comprehensive_report_builder.py`与CLI，在19项formal closeout、D2终态、21项supplement完整、
component design-gate闭合和source-test未打开全部成立前拒绝生成最终JSON/Markdown。

最终interpretation worksheet必须穷尽18个组件、Q0--Q3、H0--H5、incoming state→attention→MLP→
block output→final norm→native score功能链，以及input/representation/routing/readout/training五层。
每条finding必须引用已admit的formal或supplement ID；全部负结果与冲突、剩余不确定性、最小证伪门和
do-not-infer均为必填。设计机会中若出现绝对`layer/block/head/neuron`编号即硬失败。

`design_qualified`进一步要求实际证据等级为G、显式引用唯一有设计排序权限的component synthesis，且
functional node确实在Q2/Q3跨模型共同通过反向neutral removal、正向same-request sufficiency、
wrong-user specificity、cross stress和方向/尺度/随机负控。只有描述证据、单向patch或单模型结果不能
标G；诊断patch仍不得晋升为方法。生成器本身不打开qrels或score bundle，只记录已审计输入SHA。

新增7项测试覆盖完整轴、绝对层号拒绝、跨模型G门、描述证据越级拒绝、组件/H矩阵缺失硬失败、13节
Markdown结构与输入不可变性，均通过。此时四卡仍为96--99%利用率；D2保持36/60完整包，4个in-flight
分别为Q2 b19 fold-1约44%、Q2 b27 partial约46%、Q3 b13/b14 fold-1约67%；Q3 RoPE replacement约
`7716/8000`。这些仍只是机械状态，未读取未闭合family效应。

### Cross-component descriptive synthesis of the frozen 17

为检验“中间信号被某处弄没”是否与已闭合描述证据一致，新增一个retrospective cross-link生成器；它
穷尽读取registry中17个已冻结输出并绑定每个输入SHA，不读取四个pending结果、不打开qrels/score、
不新增确认family。输出为
`runs/20260719_kuaisearch_mech_cross_component_descriptive_v1/metrics.json`，SHA256
`6691a8770b14d5b888d35d2a7d6a738b47904295d441695cdb2ada49f36811c7`。它只允许D级描述，显式禁止改变
component support/design ranking或声称causal erasure/reversal。

固定early-vs-late region与两个既有query folds给出以下跨组件模式：

- Q2 history channel participation从`0.1755`降到`0.0888`，但history pairwise cosine从`0.4965`
  升到`0.8082`；Q3 participation从`0.1865`到`0.2039`，pairwise cosine从`0.5091`到`0.7364`。
  因而Q2后段history effect更低维/集中，而不是表示幅度简单归零；Q3没有同样的channel压缩。
- history delta与candidate-relative delta的cosine，Q2从`0.0233`升至`0.5567`，Q3仅从`0.0119`
  升至`0.0889`。Q2/Q3 late fold差分别只有`0.00262`和`0.000565`，说明这是当前两fold中的稳定
  模型差异，但仍不是显著性或跨模型族结论。
- late frozen logit lens中，Q2/Q3的candidate-relative full/null score RMS ratio分别为`1.633`和
  `2.428`，所以native方向响应没有消失；但common-history score cosine分别仅`0.0623`与`-0.3576`，
  same-sign fraction为`0.473`与`0.320`。history→candidate几何对齐与native score符号对齐之间的
  描述差为`0.494`与`0.446`。这更符合“候选传递后未稳定校准到正确score方向”，不支持把向量
  存在性直接当正确transfer。
- Q2 late RMSNorm把total delta gain缩到`0.5489`，但common与candidate-relative gain为`0.5484`和
  `0.5692`，residual/common gain ratio仅偏离1约`0.0139`，pre/post方向cosine仍为`0.8721`。Q3
  total gain为`0.9639`，ratio偏离1约`0.0567`。因此Q2确有晚段尺度压缩描述，但不是对candidate-
  relative/history部分的选择性抹除，也不跨Q2/Q3成为共同RMSNorm原因。
- history delta对query-only causal floor在全部region/model中正交fraction均大于`0.999`且RMS大约为
  query floor的`52--133`倍，反对“history只是在重复query方向”的简单解释。attention descriptive
  summary没有固定magic head/group，Q/K normalization又呈tensor/stage异质；RoPE几何仍被自然位置
  位移混杂，均须等待各自因果family，不能据此提出head/layer/RoPE修复。
- Q2全参数最终相对更新Frobenius仅`0.00245`，但update energy并未只落在单一组件：Q/K/V/O合计
  `0.4109`，MLP gate/up/down合计`0.5884`，norm约`0.00067`。跨层update RMS CV仅`0.0735`，
  max/min为`1.33`；history/candidate token-role相对更新为`0.00639/0.00646`、比值`0.988`，不支持
  简单history embedding starvation。词表top 10% rows承载`0.9903`更新能量，yes/no方向相对更新约
  `0.00993`，其Yes/No row update RMS约在词表`97.1/97.3`百分位，因此Q2也不是readout方向完全没动。
  Q3则冻结embedding/readout，仅训练28层q/v LoRA；late q head participation为`0.750`，比Q2同几何
  低`0.160`，而v为`0.956`、与Q2仅差`-0.002`。Q3 q/v-only约束可能是模型特定瓶颈，但Q2同样没有
  确认strict transfer，所以“只解冻readout”不是跨模型共同原因。RankNet与ListNet逐请求score-gradient
  cosine均值在全部surface/control很高，parameter-family share的observed最大均值差仅`0.00303`、没有
  cell达到预设`0.05`重分配阈值，不支持把两objective的普遍family冲突当现成主因；精确gradient→
  optimizer→delta归因仍等待注册replay。
- 把candidate common mode与candidate-relative hidden geometry并列后，Q2 late hidden state在当前观测
  接口有`2.32%`能量位于relative部分，common/relative比为`42.09`；Q3对应为`15.20%`和`5.58`。
  这不是隐藏层“排序有效能量”的比例：下游非线性仍可能把candidate-common位移转成candidate-relative
  位移，因此hidden common不属于严格rank-null方向，也不能据此称为浪费。冻结native lens的最终标量
  score中relative能量为Q2 `18.37%`、Q3 `51.04%`，说明readout并非完全忽略relative成分，但符号/
  用户特异性仍未建立。对全部6 cell×3 objective施加`+137` common scalar-score shift时，
  最大loss变化仅`2.02e-14`、最大gradient sum为`3.61e-16`，数值确认common score是RankNet/ListNet的
  精确rank-null方向。只有这个最终scalar-score公共平移可以严格称为loss null；该组合使“追踪并改善
  candidate-relative transport”成为更有针对性的待证伪候选，但尚不授权hidden-state centering设计，
  也不能声称centering必然提高NDCG。

随后把early/late两点扩成完整28-block的candidate-common/relative轨迹，并逐block绑定residual
decomposition、block update与非原生frozen logit lens；56个model×block行全部保留，绝对block号仅作
lineage，不选best layer。flow与residual两份独立汇总的common fraction最大差仅`1.11e-16`。固定四阶段
出现以下更关键的区分：

- Q2 hidden candidate-relative fraction依次为`0.0586/0.0758/0.0822/0.0232`，Q3为
  `0.0612/0.0878/0.1274/0.1520`。Q2后段fraction下降是真实现象，但不能等同绝对signal erasure。
- Q2四阶段mean candidate-relative energy change均为正：约`3.22e-05/4.36e-04/4.53e-03/3.79e-02`；
  Q3同样全为正：约`4.11e-05/1.38e-04/5.71e-03/2.66e-02`。两模型late stage中energy下降的
  request fraction都为`0.0`，且output/input RMS ratio分别为Q2 `1.185`、Q3 `1.138`。所以当前描述
  证据反对“某个晚层把candidate-relative幅度抹掉”；Q2更像common能量增长更快造成组成比例下降。
- input-update cosine在早中段为负（Q2约`-0.243/-0.197/-0.041`，Q3约
  `-0.266/-0.315/-0.129`），但energy change为正且output/input RMS大于1。这说明block update含有
  anti-aligned重写/抵消分量，不等于block output反转，更不能把行为margin的负方向翻译成隐藏向量翻转。
- frozen-lens relative score fraction在Q2四阶段为`0.081/0.230/0.280/0.184`，Q3为
  `0.134/0.223/0.394/0.510`。它说明固定readout对relative成分的响应随模型不同，但中间logit lens并非
  native深度、也没有用户特异正确符号，因此不能证明hidden→score转换或正确transfer。

进一步把每个state与history→candidate transport、candidate-relative brand/category real-minus-random
probe projection和frozen score-history alignment联结：

- Q2 history→candidate cosine四阶段为`0.023/0.046/0.195/0.557`，但frozen score-history cosine为
  `0.179/-0.106/-0.275/0.062`。因此中两个阶段出现“candidate state与history方向正对齐、score方向却
  负对齐”，late两者差仍为`0.494`。这比“信号消失”更像state形成后readout方向未同步，但只能作为
  D级readout-misalignment候选。
- Q3 history→candidate cosine仅`0.012/0.017/0.041/0.089`，frozen score-history cosine则为
  `0.217/0.096/0.043/-0.358`；late同号率约`0.320`且两种对齐差为`0.446`。它描述上更像candidate
  transport偏弱并叠加late score符号失配，尚需native readout与双向component gate确认。
- candidate-relative brand real-minus-random projection原始值在两模型约`1e-7--1e-5`量级；category约
  `1e-5--4e-4`。按probe rank/hidden-dimension isotropic baseline归一化后，brand excess仍仅约
  `-0.0004--0.0186`倍，category约`-0.058--0.151`倍；candidate-readout category excess则在
  `-0.251--0.215`倍之间跨阶段反复变号。因而该结论不是单纯由高维原始比例造成，但这些仍是train-only
  线性probe row-space、无确认性uncertainty且只覆盖固定样本，不能把小值或符号变化解释为偏好语义
  消失；它只说明当前brand/category代理没有给出跨模型、跨阶段稳定的正excess或语义保留证据。
- adjacent probe row-space canonical continuity在两模型的history-summary/candidate-readout、brand/category
  全部8个比较中都随depth上升，但random-label row-space也全部同步上升；late-minus-early real与random
  增量分别约`0.212--0.540`与`0.180--0.543`，全网格real-minus-random continuity最大绝对差只有
  `0.0571`。因此“深层probe子空间更稳定”主要反映generic geometric stabilization，不能当作偏好语义
  被保存或被native readout使用的证据。

因此“仍有candidate-relative能量”不等于“仍有正确偏好语义”，“history进入candidate表示”也不等于
“native score以正确用户特异符号使用”。输出边界显式把这两种越级推断设为false。

这把后续组件分解的问题收紧为：不是简单寻找“哪层把幅度弄没”，而是检验attention/MLP/residual是否
重写了candidate-relative语义或用户特异方向、common增长是否掩盖relative几何，以及native readout是否
把仍然存在的relative state映射到正确排序符号。完整轨迹仍不得修改已注册selected block或node。

综合只把候选机制问题从“哪一层变了”推进为三个需要因果分解的功能断点：history-specific state是否
形成、它是否沿candidate-relative path传递、native readout是否以正确用户特异符号使用。新增6项
cross-link测试加12项comprehensive-builder测试共18项通过；四个pending补充仍按冻结队列等待，
上述描述值不会改变其节点、层、阈值或统计门。

最终worksheet进一步强制在八类功能失败模式中做显式判定：state进入candidate path前缺失、局部state
attenuation、分布式attenuation、candidate transport失败、state存在但readout失配、objective/update
失配、多瓶颈或unresolved。D/M/S/U级证据都不能授权causal erasure/loss-of-use；授权必须为G。若称
loss-of-use，还必须让相应Q2/Q3 scope的finding同时绑定各自native-readout formal证据，防止只凭logit
lens或单向patch下结论。unresolved必须保持U且两个causal flag均为false；任何精确层号进入design继续
硬失败。

加入两套综合后完整仓库回归为`816 passed, 7 subtests passed in 24.04s`；七个冻结资产SHA与原值一致，
`git diff --check`通过。

随后Q3 block-13 RoPE replacement机械闭合：8,000/8,000 requests、160,753 score rows，identity
最大误差`0.0`，qrels/source-test读取均为false，score SHA
`2d145eee9c52bb41ae5828f55bf40250ddb17c29c8c45e0c15d741f7037174b1`与文件字节一致。未读取单模型
效应；固定队列已自动在同一卡启动Q2 replacement，等待Q2/Q3完整D5 family后统一评估。

### Layer scan is a locator, not a design result

进一步把层扫描的科学角色写入最终报告生成器。它只回答“history-conditioned state、candidate-relative
geometry或native-score relation在哪个功能接口前后发生变化”，授权用途仅是界定后续双向因果分解的
pre/post接口。它不能单独证明attention、MLP、residual、normalization或readout的因果责任，不能把
绝对block号变成结构超参数，也不能从一个模型的转折层外推数据集或规模规律。

跨Q2/Q3的设计对齐单位固定为功能节点及其有符号因果行为：incoming residual、attention o-proj、MLP
down-proj和完整block output。只有同一功能节点在两模型同时通过position-preserving reverse removal、
same-request sufficiency、wrong-user/cross-request specificity以及direction/scale/random结构负控，才有
设计排序资格。报告Markdown现在显式写明这一区分，并新增测试阻止层扫描被当作设计证据；因此层号即使
随模型大小或数据变化，也只影响实验lineage，不改变设计命题的表达单位。

### Fail-closed comprehensive decision worksheet

最终report builder原先要求外部提供human decisions JSON，但没有可执行的穷尽模板。新增
`scripts/init_transformer_comprehensive_decisions.py`与
`build_comprehensive_decision_template()`，固定生成18组件、H0--H5、incoming→attention→MLP→block
output→final norm→native score、五个系统层和Q0--Q3边界的全部槽位。模板默认
`worksheet_status=incomplete`，validator新增硬门：只有显式`final`才进入证据引用、组件支持和设计机会
检查；因此字段占位符或空finding不能被误生成成最终报告。模板同时记录绝对层/head/neuron禁止用于设计、
source test必须关闭、metric只从admitted evaluator复制等指令。CLI smoke生成18组件/6假设/6功能节点，
完整仓库回归为`819 passed, 7 subtests passed in 24.42s`。

### From localization to design: explicit four-stage evidence bridge

为避免“层扫描做完后仍不知道如何影响设计”，comprehensive builder进一步固定四级桥接，而不是让最终
作者按结果自由解释：

1. `state_localization`只用完整轨迹、绝对/相对state measure和固定depth region界定pre/post接口，
   不选best index，`design_authority=false`；
2. `component_disambiguation`用同一parent的incoming residual、attention output、MLP output与完整
   block state干预区分carrier，仍不声称operator necessity，`design_authority=false`；
3. `bidirectional_causal_mediation`要求same-request sufficiency、wrong-user specificity、cross/random/
   direction/scale结构负控与position-preserving neutral reverse removal共同闭合，只形成model-local
   设计候选，`design_authority=false`；
4. `cross_model_functional_replication`要求Q2/Q3在同一功能节点呈现相同有符号因果行为，才允许该节点
   改变冻结scope内的architecture opportunity排序，`design_authority=true`，绝对层号仍只作lineage。

最终Markdown会逐行显示问题、所需证据、授权后果和design authority，因而层扫描与组件分解之间不再是
隐含推理。该增强没有新增实验family、没有读取未闭合effect，也没有改变冻结层、节点、阈值或统计门；
comprehensive/readiness/design定向回归为`20 passed`。

### Candidate-common anisotropy and the generic-history competing explanation

此前已经区分hidden candidate-common与最终scalar-score common shift，但还可能把大量hidden common
能量误读成request-specific preference。新增固定四阶段anisotropy cross-link，不选层、不读qrels：

- Q2 late candidate-common energy fraction为`0.97725`，其中common delta跨请求global-mean energy fraction
  为`0.78177`，mean pairwise cosine为`0.79019`；Q3对应为`0.84636/0.74528/0.74888`；
- 两模型四个固定阶段的common global-mean fraction全部大于`0.5`，common pairwise cosine也全部为正，
  表明candidate-common history delta在请求间含有显著共享方向；
- Q2 late common-history channel energy cosine从early `0.3224`升到`0.7682`，history delta自身的
  global-mean fraction从`0.4916`升到`0.8026`；Q3对应history global-mean从`0.5042`升到`0.7257`。
- Q2 late history channel participation ratio仅`0.08885`且top 1% channels承载`0.25786`能量；Q3对应
  为`0.20391/0.16655`。这提供了Q2晚段channel concentration候选，而不是幅度消失；它将由已冻结
  MLP feature-formation extension检查，不能据此选择neuron或追加group。

因此generic history/prompt response成为需要保留的竞争解释：大量公共位移不能自动等同个性化偏好。
但高跨请求同向性也不能证明非个性化或无用，hidden common仍可能经后续非线性转成relative score；只有
最终scalar-score公共平移是已数值验证的exact rank-null。结果只提高wrong-user specificity、cross donor
和position-preserving neutral removal的解释必要性，不改变任何注册family或设计排序。新增手算测试后
cross-component定向回归为`9 passed`，再生成artifact SHA256为
`6691a8770b14d5b888d35d2a7d6a738b47904295d441695cdb2ada49f36811c7`。
包含report bridge与anisotropy cross-link后的完整仓库回归为
`820 passed, 7 subtests passed in 24.09s`；`git diff --check`与七项冻结资产hash继续通过。

### Exhaustive component questions and attention mass/contribution separation

最终全面报告的18组件矩阵此前有status、evidence level、model scope和remaining uncertainty，但组件要
回答的机制问题仍依赖自由文本。新增exact-coverage `COMPONENT_FUNCTIONAL_QUESTIONS`：从serialization/
token embedding/RoPE、Q/K routing/value transport/o-proj、SwiGLU formation/down-proj、residual/
RMSNorm、layer/history/candidate interaction、native readout/nullspace，一直到loss/optimizer/LoRA，18项
均有单独问题；集合与`COMPONENT_IDS`不一致时import即失败。JSON每个component row和Markdown矩阵都
显式携带该问题，因此不能再用“层表示有变化”代替attention、MLP或训练路径的具体回答。相关定向回归
为`21 passed`。

同时从冻结attention-pattern supplement提取12个固定model×block×query-head/GQA-group单元，区分history
attention mass与history o-proj contribution norm：

- 全12单元Pearson均值`0.90174`、范围`0.51712--0.98092`，说明多数固定单元中mass与贡献幅度相关；
- 唯一低于`0.8`的两个单元均在Q3早段：query-head `0.51712`、GQA-group `0.66685`；attention mass
  因而不是跨模型/阶段通用的contribution proxy；
- contribution使用norm，既没有有符号偏好方向，也没有用户特异正确性；mass或norm都不能证明value-edge
  causality，更不能按top head/group选择设计。

该审计把“attention看到了history”与“history value因果改变正确排序”分开，后者仍等待注册edge/value与
component双向干预。新增手算测试后cross-component定向回归为`10 passed`，artifact SHA256更新为
`6691a8770b14d5b888d35d2a7d6a738b47904295d441695cdb2ada49f36811c7`。
包含两项新增覆盖测试后的完整仓库回归为`822 passed, 7 subtests passed in 24.63s`。

### D2 fold-1 mechanical continuation and causal-bridge audit

Q2 block-20 fold-1 post-block bundle已机械闭合：3,918/3,918 requests、3,918 score rows，
`complete_finite_score_coverage=true`、`identity_passed=true`、qrels/source-test读取均为false；metadata中
score SHA `2bd1ab941a810f4fe621c34c966a3e3b9813187b2d7ae8e6c4f0edf35b2e4deb`与文件字节一致。
队列随后自动启动block-21；此时D2固定bundle为40/60，按request加权执行进度约`0.6841`。这里只记录
机械完成度，没有读取block-20科学效应，也没有据此改变selected layer、family或后续队列。

同时按冻结V2合同逐项核查“定位→分解”实现：parent D2对同一selected transition固定七个state接口，
本轮reverse extension只在同一parent bytes上对incoming residual、attention o-proj、MLP down-proj和
block output做position-preserving neutral-to-full removal；Q3覆盖shared prompt、teacher-forced Yes/No
三条native scoring state。最终design gate要求neutral necessity、same-request sufficiency、wrong-user
specificity、cross-request以及direction/scale/random负控同时通过，并且Q2/Q3复现同一功能节点。
因此这组实验能回答“哪一个组件state是必要且充分的中介”，但仍不能回答operator本身是否必要、是否是
唯一来源或能否跨数据集/模型规模泛化；这些边界继续显式保留。

### Artifact breadth is not causal breadth

readiness此前只报告18组件中多少已有任意formal或supplement artifact，容易把描述性几何覆盖误读为机制
闭合。现改为逐组件并列报告：任意artifact、注册的causal-role artifact、已闭合causal-role artifact，且
明确completion只表示evaluator可用，不表示注册符号、CI、specificity、跨模型门通过。当前机械状态为：

- 16/18组件已有至少一个完成artifact；
- 14/18组件在冻结计划中至少有一个注册causal-role evaluator；
- 只有3/18组件已有完成causal-role artifact：Q2 native-readout覆盖的`native_readout`与
  `score_calibration_nullspace`，以及Q2 objective覆盖的`loss_gradient`；这里仍不从完成度推断支持；
- 其中0/18已同时具备Q2与Q3的完成causal-role artifact；三个已完成组件目前model scope均只有Q2，
  所以不能把“有一个因果evaluator闭合”误写成跨模型机制复现；
- `token_embedding`、`mlp_feature_formation`、`optimizer_effective_update`与`lora_parameterization`在当前
  停止阶段没有独立causal-role artifact，属于必须进入最终报告的causal debt，不能用描述性update/geometry
  结果冒充排除或归因。

全面报告的执行总表现在会同时显示任意artifact、causal-role artifact与Q2/Q3双模型causal-role覆盖。
第一轮相关readiness/report/contract回归为`134 passed`，随后双模型scope增强定向回归为`18 passed`；
scope增强后的完整仓库回归为`823 passed, 7 subtests passed in 26.57s`，`git diff --check`通过。
未新增实验family、未读取未闭合效应、未改变任何冻结统计门。

### Mechanical non-results remain first-class evidence records

为防止后续V2修复掩盖早期机械失败，readiness与最终全面报告执行总表现显式携带closeout已审计的
mechanical non-result ledger。当前保留7个绑定记录，覆盖Q3 native-position gate、Q3 attention
reconstruction、Q2/Q3 MLP permutation smoke与Q2/Q3 RoPE precision gate等失败；每个record保留输入与
失败artifact hash。它们的统一解释固定为“可复现机械non-result”，既不是科学null，也不是反对相应机制
的证据。V2成功bundle只能替代正式执行路径，不能删除这些历史记录。

新增readiness/report/closeout定向回归为`51 passed`，当前closeout audit仍为pending但`failures=[]`，
绑定机械non-result数量为7；完整仓库回归为`823 passed, 7 subtests passed in 26.40s`。

### Component claims cannot borrow unrelated causal findings

进一步red-team最终comprehensive worksheet时发现：旧validator要求`status=supported`必须使用S/N/G，
但没有强制该finding的因果deliverable属于对应组件。理论上可能用`d2_selected_branches`的attention/MLP
结果把token embedding或SwiGLU formation写成supported，或者把Q2/Q3 finding的scope扩到Q0/Q1。

现加入三层硬绑定：

1. S级组件必须至少引用一个S finding，且其formal deliverable与
   `COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component]`相交；causal集合为空的embedding、MLP
   formation、optimizer与LoRA不能借用其他组件证据；
2. N级组件必须属于position-preserving reverse-removal注册组件，并引用necessity supplement；G级组件
   必须属于实际通过的cross-model functional node对应组件，并引用design-gate supplement；
3. supported组件声明的model scope必须是上述matched causal findings模型scope的子集，禁止把Q2/Q3或
   单Q2结果扩到未覆盖模型。

新增S/N/G正反例与scope外推测试后，comprehensive/readiness/formal-contract定向回归为`138 passed`；
完整仓库回归为`827 passed, 7 subtests passed in 28.02s`。

### H0--H5 decisions cannot borrow another hypothesis's evidence

同一red-team发现hypothesis matrix也只校验决定性状态使用S/N/G，没有重验该finding是否属于对应H0--H5
及其全部注册evidence groups。现直接复用formal contract的冻结映射：

- supported/rejected必须有同evidence level且与该hypothesis允许deliverable相交的finding；
- supported必须覆盖该H的每个注册evidence group，不能只满足其中一组；
- H1还要求`attention_query_key_routing`与`history_routing`组件均supported，H4要求`loss_gradient`
  supported，防止只用描述性head/update结果升级机制；
- H5因本停止阶段没有注册独立第二seed，任何supported/rejected决定均fail closed。

H4借selected-branch、H1缺context group、H1缺必要组件、H5伪决定以及完整H1正例均已加入测试；相关
comprehensive/readiness/formal-contract回归为`141 passed`；完整仓库回归为
`830 passed, 7 subtests passed in 26.30s`。

### Optimization opportunities require component-matched evidence

继续审计“最可能优化方向”时发现默认测试夹具本身暴露了另一个风险：一个标为native-readout的candidate
引用D1 representation与activation anisotropy，旧validator只检查evidence存在，不检查它是否属于该
functional component。现把completed supplement的component/model metadata显式传入validator，并固定：

- 每个opportunity至少有一个formal deliverable属于`COMPONENT_ALLOWED_DELIVERABLES[component]`，或一个
  supplement的注册components包含该component；
- opportunity model scope必须是这些component-matched evidence model scope的子集；
- actual level S必须引用该组件注册causal deliverable，N必须引用该组件的reverse necessity，G继续要求
  实际通过的cross-model functional node；描述性geometry只能维持D/U候选；
- 测试默认候选改成与D1/anisotropy真正匹配的`layerwise_representation`，不再保留一个错误但能通过的
  readout示例。

新增component mismatch、scope expansion、descriptive-to-S与descriptive-to-N四个反例后，相关回归为
`143 passed`；完整仓库回归为`832 passed, 7 subtests passed in 26.48s`。

### Causal debt is named, not hidden behind aggregate coverage

readiness与最终Markdown执行总表现直接列出没有注册独立causal-role artifact的组件，而不只显示14/18：
`token_embedding`、`mlp_feature_formation`、`optimizer_effective_update`与`lora_parameterization`。这四项
仍有描述性geometry/update证据，但当前停止阶段不能支持组件因果结论；最终报告必须把它们作为明确的
remaining causal debt，而不是从“16/18已有artifact”推断接近全面因果闭合。新增定向回归为`27 passed`。
完整仓库回归仍为`832 passed, 7 subtests passed in 25.85s`。

### Functional causal-chain rows are now causal claims, not free text

`incoming_state→attention→MLP→block_output→final_norm→native_score`此前只固定node顺序，status为任意
非空字符串，理论上可用D级finding写成“supported”。现固定status集合为supported/weakened/unresolved/
mechanical_failure，并加入node→component映射：incoming对应residual/history routing，attention对应
Q/K、V与o-proj，MLP对应formation/output，block output对应residual，final norm对应normalization，
native score对应readout/nullspace。

supported chain node必须使用S/N/G、引用同等级finding，且至少一个映射组件已经supported；unresolved只
能用U，mechanical failure只能用M。非法status、D级伪support、unresolved+D、缺mapped component和完整
attention正例均有测试，相关回归为`145 passed`。
完整仓库回归为`834 passed, 7 subtests passed in 26.70s`。

### Behavioral sufficiency attenuation is not semantic signal erasure

针对“层扫描只定位变化是否能指导设计”的质疑，重新审计最终comprehensive failure-mode合同后确认一个
越级解释风险：旧validator虽然要求`causal_erasure_claim_authorized=true`至少达到G，但G只证明同一
功能节点通过same-request sufficiency、wrong-user specificity、reverse neutral removal与跨模型结构门，
仍只建立history-specific state mediation，不能证明某个operator销毁了语义信息。

现将边界机器化为：

- 当前注册实验不存在直接的semantic signal-erasure test；post-block层扫描测量的是native ranking
  behavioral sufficiency随深度的变化，不直接观测history-token flow；
- 即使bidirectional component mediation达到G，也不得把它改写为causal erasure；localized/distributed
  attenuation仍可作为行为轨迹形状描述，但`causal_erasure_claim_authorized`必须为false；
- 非`unresolved` failure mode必须引用同evidence level finding，且声明的model scope不得超过这些
  level-matched findings；
- `state_present_but_readout_misaligned`若要授权因果表述，除Q2/Q3对应native-readout formal deliverable外，
  还必须有model-scoped supported `native_readout`组件与supported `native_score`因果链；
- 最终comprehensive S级component/hypothesis不能升级或反转formal report的已审计outcome；formal report
  admission也会重验两个primary模型的`component_erasure_boundary_established=false`、层扫描未直接观察
  history-token flow及绝对层号不是架构证据。

新增G级伪erasure、缺native-score chain、formal outcome升级、model-scope扩大及formal erasure flag漂移
等正反例。相关定向回归为`164 passed`；完整仓库回归为`838 passed, 7 subtests passed in 27.20s`，
`git diff --check`通过，七个冻结计划/manifest/registry哈希均未改变。审计期间未读取未闭合D2科学效应，
未改变实验family、selected layer、统计门或四卡队列。

### Incoming state is not selected-block residual causality

继续审计“层定位如何转化为可迁移设计”时发现，综合报告的功能节点映射曾把
`block_input_residual`同时映射到`history_routing`与`residual_composition`。这会允许一种错误升级：如果
有害状态在进入selected block前已经同时通过S/N/structural gates，反而把当前block内部的residual
composition写成G级设计目标。该解释与formal attribution中“incoming-state sufficiency阻断当前block
归因”的规则矛盾。

现固定：

- `block_input_residual`只授权上游history-state path/history routing候选；
- `block_output_residual`严格保留为完整state ceiling；即使其S/N/structural gates通过，也不能映射到
  `residual_composition`或获得design priority，因为absolute-state patch没有隔离residual addition或
  nonlinear interaction；
- attention o-proj与MLP down-proj仍分别只映射到attention output与MLP output；
- comprehensive admission不再只信design supplement中的shared-node列表，而是从Q2/Q3每个
  `(model,functional node)`行重新推导state-supported与design-prioritized交集，验证primary endpoint固定为
  target margin、两个模型parent bytes均已绑定、design gate蕴含state gate，并重验全部汇总布尔值。

同时强化component design synthesis输入审计：任何`registered_support=true`的parent selected-branch行
必须是非missing、注册负符号、mean<0、BH q<0.05且来自confirmatory fold-1；任何
position-preserving necessity=true必须是neutral donor、mean>0、CI下界>0、BH q<0.05。伪造support布尔值
但不满足效应门的fixture现均fail closed。

新增incoming-state错误映射、伪cross-model汇总、design无state、伪selected support与伪necessity support
反例，并增加block-output-only state ceiling不得成为residual设计目标的正反例；完整仓库回归为
`843 passed, 7 subtests passed in 26.58s`，`git diff --check`及七个冻结哈希通过。
这次只修改综合/审计代码与测试，没有读取未闭合科学效应或改变实验family。

随后Q2 block-21 fold-1机械闭合，固定D2 bundle从40/60增至41/60；3,918 score rows完整，
`complete_finite_score_coverage=true`、`identity_passed=true`、最大identity delta为0，scores SHA
`9ca339f5294e545f6861a63267efb64cb19ca6d9011a4168442172b4e7e8b2df`与文件字节一致，qrels/source test
均未由scorer读取。队列已自动切到block-22。本次只读metadata、行数与hash，没有读取block-21科学效应。

### Functional-chain support cannot reverse causality across a block boundary

继续审计最终`incoming_state→attention→MLP→block_output→final_norm→native_score`因果链时发现，旧映射允许
`residual_composition`组件支撑`incoming_state`。这会把当前block内部的composition证据反向填成“进入
block前状态已存在”，与七节点时间顺序及incoming-state confound规则矛盾。

现将`incoming_state`只映射到`layerwise_representation`与`history_routing`，明确排除
`residual_composition`；新增反例验证即使residual component为S，incoming chain仍不能supported。并为六个
chain node自动附加不可手改的claim boundary：incoming不证明上游唯一来源，attention不证明唯一head/origin，
MLP output不把描述性SwiGLU group变成formation operator，block-output state ceiling不证明residual
addition，final-norm只允许pre/post state-boundary定位而不证明RMSNorm operator necessity，native-score只
覆盖冻结原生readout且不自动证明utility。

18组件矩阵也会直接携带formal contract的逐组件probe boundary，最终Markdown不再只显示自由文本summary；
因此normalization/residual/readout等组件即使被标为S，也必须和对应状态级、模型级边界并列展示。新增映射、
边界覆盖及render回归后，完整仓库为`844 passed, 7 subtests passed in 26.09s`；`git diff --check`与七个
冻结哈希通过。未读取运行中D2 effect，也未改变实验计划或统计门。

### Functional localization is model-scoped triage, not a transferable layer index

针对“知道哪一层改变是否对设计有意义”的进一步质疑，综合报告现在把层扫描严格降级为功能转折定位：
绝对block编号只保留为lineage metadata；它只能冻结后续分解的`j-1 → j`局部边界，不能单独产生
attention、MLP、residual、normalization或readout设计结论。真正可迁移的报告单位是
`incoming_state → attention → MLP → block_output → final_norm → native_score`功能链，以及相同证据等级、
相同模型scope的组件支持。

为防止“Q2某层/组件成立”被写成“Q2/Q3共同功能瓶颈”，supported chain node现在必须同时满足：

- S/N/G因果等级，并引用同等级finding；
- 至少一个映射组件在同一等级supported；
- chain声明的model scope非空，且是同等级finding与同等级组件model scope交集的子集。

因此，Q2-only组件证据即使与Q2/Q3 finding同时存在，也不能支撑Q2+Q3 chain声明；新增反例固定这一边界。
同时，N级组件model scope不再从“necessity supplement已完成”推断，而是从逐模型、target-margin、
position-preserving neutral-removal gate实际通过行派生；N级supplement本身也不得决定H0--H5。相关
comprehensive/component/readiness/formal-contract定向回归为`162 passed`。本次仍未读取未闭合D2科学效应，
没有改变冻结family、层选择、四卡队列或source-test边界。

### Negative conclusions are evidence-matched rather than free-text downgrades

继续red-team最终综合合同发现，`supported`已经有组件、等级和模型scope硬门，但`weakened`此前仍可能只靠
worksheet自由文本写入。这会允许用不相关的层曲线或描述性artifact把attention、MLP、功能链节点或H0--H5
写成“削弱”，虽然措辞比supported保守，仍会实质改变优化排序。

现补齐负向结论门禁：

- weakened组件只能使用D/S级、同等级且属于该组件注册deliverable的finding，声明model scope必须落在这些
  finding覆盖模型内；最终build还要求其status、scope和deliverable与已审计formal component outcome一致；
- weakened功能链必须达到S/N/G，引用同等级finding，并由同节点映射、同等级、同model scope的weakened组件
  支撑，不能只靠层扫描轨迹自由填写；
- weakened H0--H5只能使用D/S级且hypothesis-matched的正式deliverable，最终status必须与formal
  hypothesis matrix完全一致；necessity/design supplement不能单独制造负向假设结论。

新增formal mismatch、跨模型scope、借用不相关组件、缺负向组件链和hypothesis deliverable错配正反例；
comprehensive定向回归为`40 passed`，component/readiness/formal-contract相关回归为`126 passed`。
本次仍未读取运行中效应、未改变任何实验family或冻结哈希。

### A design-qualified component gate is not an H0--H5 refutation

继续审计G级综合时发现一个非对称漏洞：necessity早已禁止单独决定H0--H5，但G级design-gate finding此前
可在不引用该hypothesis允许的formal deliverable时充当`rejected`依据。G表示某个功能节点同时通过
sufficiency、reverse removal、history specificity与结构负控；这是设计方向资格，不是对任意竞争假设的
独立反证。

现固定：

- H0--H5的`rejected`只能使用S级正式结果，并必须与已审计formal hypothesis status完全一致；
- G级finding若要支持（而非拒绝）某个hypothesis，仍必须同时引用该hypothesis允许的formal deliverable和
  design-gate supplement；design supplement不再替代hypothesis-matched evidence；
- N级necessity、G级设计资格和S级hypothesis反证保持三个不同语义层，不能相互借证据升级。

新增G级伪refutation、G级缺formal evidence伪support和S级rejection与formal status不一致反例；
comprehensive定向回归为`42 passed`。未读取任何运行中科学效应，四卡队列与冻结协议未改。

### Queue handoff remains completion-gated and effect-blind

在Q3 fold-1接近尾段时重新审计四卡交接顺序：两条Q3 lane都必须先等待全部15个fold-0 bundle的最终
`metadata.status=completed`，lane 0才运行共享evaluator并原子冻结selection；随后两个lane才允许启动各自
fold-1。selected-branch confirmation与contract必须等待全部15个fold-1 bundle完成，contract只暴露
selected block、复现布尔值与evidence role，不携带fold效应值。组件reverse-necessity队列还要求
`fold1_negative_transition_reproduced=true`和confirmatory role；探索性selected-branch scoring不能解锁N/G
组件结论。

运行中四个partial bundle的请求行数、progress request count与partial SHA持续一致；closeout只把最终
metadata作为完成标记。queue topology、progress、overview、closeout与supplement registry定向回归为
`52 passed`。未提前读取fold-1效应或qrels，source test保持关闭。

### G means exactly Q2 plus Q3, not a selectable model subset

继续审计G级scope时发现，finding旧门只要求scope“包含”Q2/Q3，因此理论上可附加未由design gate覆盖的
Q0/Q1；而component、chain与opportunity又可能把已跨模型的G结果缩写成Q2-only。两种写法都破坏G的固定
语义：G不是可自由裁剪的强证据标签，而是同一功能节点在两个primary transfer模型上共同通过全部门。

现固定`PRIMARY_DESIGN_MODELS={Q2,Q3}`，并要求：

- 所有G finding的model scope必须精确等于Q2/Q3，既不能扩到Q0/Q1，也不能缩成单模型；
- G component、functional-chain node、failure-mode diagnosis与design-qualified opportunity同样必须精确覆盖
  Q2/Q3；
- 单模型S/N结果继续保留各自scope，但不得换标签为G。

新增G finding扩张/缩减、G component缩减、G chain缩减与G opportunity缩减反例；comprehensive定向回归
为`44 passed`，相关component/readiness/formal-contract回归为`126 passed`。实验family、四卡队列和冻结
哈希均未改变，运行中效应仍未读取。

### Necessary mediator of harm is not a beneficial transfer component

继续核对reverse-removal方向语义时发现，“necessary mediator”即使统计门正确，也容易在最终方法讨论中被误读
成“该组件对transfer有益，应加强它”。V2实际注册的行为是有害full-history target-margin response：把
position-preserving neutral state写入full recipient后若margin正向改善，证明的是移除该状态减少了伤害。

现把方向边界同时写入component-design机器输出、comprehensive admission与最终Markdown：

- `registered_behavior=harmful_full_history_target_margin_response`；
- positive neutral removal只解释为harm reduction；
- component beneficial for transfer与strengthen/preserve component两个授权标志固定为false；
- N/G只允许说该接口是已注册有害行为的必要/充分中介，后续设计可能需要抑制、重路由或重新校准，不能从门本身
  推出“增强该模块”或utility提升。

comprehensive会拒绝篡改这些方向标志的design supplement。component/comprehensive定向回归为`51 passed`，
readiness/formal/necessity相关回归为`131 passed`。未读取运行中效应，也未依据可能的结果改变方向定义。

### Design opportunities must state an intervention polarity

在“有害中介不等于有益组件”的边界上再向设计落地一步：仅报告component与priority仍可能让方法阶段自行选择
相反动作。comprehensive opportunity现新增机器校验的`intervention_polarity`：

- `suppress_harmful_state`、`reroute_history_state`、`recalibrate_candidate_readout`；
- `preserve_or_strengthen_beneficial_state`与`diagnostic_only`只可用于未达到N/G的其他候选语境；
- 任何N/G harmful-mediator opportunity若选择strengthen/preserve或diagnostic-only都会fail closed；
- 最终Markdown优化机会表显式展示polarity，使“定位到哪个组件”和“准备对它做什么”不能混为一谈。

新增N级strengthen与diagnostic-only反例，G级测试固定使用harm suppression；comprehensive定向回归为
`44 passed`，component/readiness/formal相关回归为`126 passed`。这只收紧报告到设计的解释合同，没有新增
实验family或方法实现。

### Opportunity ranking is ordered evidence, not an unordered shortlist

最终计划要求“最可能优化方向排序”，但旧worksheet只验证每行priority，不验证整表顺序，理论上可能把
deprioritized方向放在G级design-qualified之前。现固定跨档顺序：

1. `design_qualified`；
2. `candidate_to_test`；
3. `deprioritized`；
4. `not_recommended`。

同一档内仍由完整冲突证据、风险和最小证伪实验决定次序，不用绝对层号或单个效应大小机械排序。validator
要求列表按上述顺序单调，并为最终JSON/Markdown生成连续`rank`；新增降权项排在候选项前的反例与合法排序
正例。comprehensive定向回归为`45 passed`，相关回归为`126 passed`。未消费运行中效应进行排序。

### Q3 blocks 15 and 16 fold-1 mechanically closed

Q3 block-15与block-16 fold-1现均完整闭合，D2固定bundle从41/60增至43/60。两包各覆盖3,918个fold-1
请求、78,864个score rows；`complete_finite_score_coverage=true`、`identity_passed=true`、最大identity
delta均为0，full/null baseline最大差分别为`2.9802322387695312e-8`与
`1.4901161193847656e-8`。最终score SHA分别为：

- block 15：`2a6795ea01cbe71c272f8a03fd8f93da50b47bfeb5202857a964613dccdc7794`；
- block 16：`f4607ddad59c2c218bf6a22b1d4be066f19e95fcbc3f73de3d382674a61d3564`。

两者文件字节SHA、3,918行请求覆盖、progress final marker、fold-0 selection SHA及共同records/request/
candidate manifest均一致；scorer的`qrels_read=false`、`source_test_opened=false`。两条Q3 lane已自动切换
到block-17与block-18 fold-1。该审计没有读取任何patch或margin效应。

### Mechanism rank is not established utility rank

优化机会按设计证据排序后，仍需防止读者把高rank解释为“已证明NDCG收益更大”。当前阶段没有实现或评估新
transfer架构；G只表示功能节点通过跨模型双向机制门，target-margin方向修复也不自动等于ranking改善。

现要求每个opportunity显式携带`utility_gain_established=false`，任何人工改成true都会fail closed；最终
Markdown排序表同时显示该字段，顶层claim invariant也固定整体utility尚未建立。新增伪utility反例后，
comprehensive定向回归为`46 passed`，相关回归为`126 passed`。这使“最可能优化方向”准确表示机制设计
优先级，而不是未经实验的收益排行榜。

### The final component table is now a literal 18 by 4 matrix

最终报告计划要求“18组件×4模型矩阵”，但原comprehensive worksheet只有18条聚合组件行，依靠单个
`model_scope`字段表示覆盖模型。这种表示无法逐格区分Q2/Q3深测、Q0/Q1描述性覆盖与完全未测，也可能让
一个模型的结论在表格中被误读为组件级共同结论。

现新增强制`component_model_matrix`，精确覆盖18个注册组件与Q0--Q3四个模型，共72个单元。每格独立携带
status、evidence level、summary、supporting findings与remaining uncertainty；supported、weakened和
mechanical-failure格必须有同等级、同组件、同模型的证据。聚合组件若声明某模型为supported/weakened/
mechanical failure，对应模型格必须使用相同status与level；模型边界中的`uncovered_components`必须与
72格中`untested`集合逐项一致。最终Markdown现在实际渲染72行，而非用一个Models列压缩异质性。

新增缺失模型格、聚合/逐模型矛盾、借用Q2/Q3证据支持Q0以及uncovered列表漂移反例；comprehensive定向
回归为`50 passed`。该修改只强化既有报告合同，不新增family、不重选模型/层，也没有读取运行中科学效应。

### Q2 block 22 fold-1 mechanically closed

Q2 block-22 fold-1已完整闭合，D2固定bundle从43/60增至44/60。该包覆盖3,918个fold-1请求、78,864个
score rows；`complete_finite_score_coverage=true`、`result_eligible=true`、`identity_passed=true`，最大
identity delta为0。重算full/null baseline的最大低精度差均为`0.0625`，已由注册的path-local BF16 bound
审计通过；它不是native scorer identity阈值，也不是科学效应。最终score SHA为
`97da21ab7ba74da09c7b6fa520b508817732324505e8b3e22bc3fb53c9998e89`，文件实际SHA与metadata一致。
scorer保持`qrels_read=false`、`source_test_opened=false`，队列已自动切换到Q2 block-23 fold-1。该闭合
审计没有读取patch、margin或NDCG效应。

### Execution coverage now names runs, models, endpoints, and fold roles

继续逐条对照comprehensive report plan发现，第1节原先列出了19个formal deliverables、21个supplements、
D2包数与机械失败，但没有直接显示formal run declaration总数，也没有把模型、端点和fold角色作为固定执行轴
列出。这会让“全面覆盖”在报告中缺少可核对的实验坐标。

现将formal execution census原样带入最终报告，显示declared runs、run status counts与result-eligible
completed runs；同时加入固定execution-axis census：Q0--Q3四模型、primary target margin、secondary
NDCG@10、fold-0注册定位/发现角色与fold-1固定转折确认角色。边界明确说明qrels-blind descriptive
supplements可以没有注册endpoint，fold-1不能重新选择transition。该合同不重算metric，也不把描述性证据
伪装成确认性结果。comprehensive定向回归仍为`50 passed`；下一轮全套回归将一并覆盖。

### Every optimization opportunity now carries evidence bytes

最终报告计划要求每个优化方向同时给出supporting formal deliverables、supporting supplements及SHA。旧schema
验证了证据ID与组件/model scope，却只在全局admission区保存文件SHA，方向行本身没有绑定字节身份；如果引用
列表与全局表发生漂移，读者需要人工拼接。

现由最终builder在formal closeout与supplement registry全部终态后自动附加
`supporting_evidence_identities`，逐项包含evidence ID、formal/supplement类型、path和64位SHA-256，顺序与
原引用列表完全一致。缺文件identity、非completed状态或伪造SHA都会fail closed；人工worksheet不负责抄写
SHA。Markdown优化机会表也显示完整`evidence_id@sha256`。新增合法绑定、输入不变与伪SHA反例后，
comprehensive定向回归为`51 passed`。该步骤只绑定已审计文件，不读取或重算科学结果。

### Opportunity rows expose expected benefit and risk separately

最终计划还要求每个优化方向写明预期收益、风险与最小证伪实验。旧opportunity虽然已有reason、冲突证据、
`do_not_infer`和falsification gate，但没有把expected benefit与key risks做成独立必填字段，容易让排序理由
代替风险披露。

现新增非空`expected_benefit`与非空`key_risks`，两者均禁止绝对layer/head/neuron编号；既有
`utility_gain_established=false`继续强制，因此expected benefit只能是待检验设计预期，不能写成已经建立的
NDCG收益。最终Markdown将预期收益、关键风险和证伪门并列展示。新增缺字段、空风险列表和风险文本夹带精确
层号反例后，comprehensive定向回归为`52 passed`。

### “Not recommended” now states why it is not recommended

计划要求明确区分某方向是被注册结果反证、因果证据不足、存在位置/测量混杂、只停留在描述层，还是只有
机械non-result。旧`basis`是任意字符串，无法机器区分这些理由。现固定五类basis：
`registered_refutation`、`insufficient_causal_evidence`、`position_or_measurement_confound`、
`descriptive_only`、`mechanical_non_result`。任意模糊标签会fail closed；comprehensive定向回归为
`53 passed`。这防止把“还没证明”误写成“已经反证”，也防止把机械失败当科学负结果。

上述72格矩阵、执行轴、逐方向SHA、benefit/risk与not-recommended basis合同合并后，全仓回归为
`864 passed, 7 subtests passed`，`git diff --check`通过。

### Cross-model narrative cannot borrow another model's finding

72格组件表已按模型拆开，但Q0--Q3横向边界的`supporting_findings`此前只检查ID存在，没有检查finding的
model scope。这会允许Q0/Q1总结引用仅覆盖Q2/Q3的证据。现要求每个模型边界引用的每个finding都必须显式
包含该模型；未覆盖模型可以不引用finding并报告untested/unresolved，不能借用其他模型的结果。新增Q0借用
Q2/Q3 finding反例后，comprehensive定向回归为`54 passed`。

### Five explanatory layers cannot substitute for one another

最终报告的input、representation、routing、readout与training五层此前只要求finding ID存在，因此表示证据
理论上可以被复制到routing或training段。现将18组件完整且不重叠地映射到五个解释层，并要求每个被引用
finding至少对该层一个组件、一个实际model scope成立；无对应证据时允许保留空引用与remaining uncertainty，
不允许跨层补故事。Markdown同时列出每层包含的功能组件。新增representation finding伪装routing证据反例
后，comprehensive定向回归为`55 passed`。

同一scope约束现也覆盖`unresolved`组件聚合行与逐模型格：保守状态不再允许引用错误组件或错误模型的finding。
新增Q2/Q3 representation证据写入Q0 native-readout unresolved格/聚合行反例后，定向回归为`56 passed`。

`not_recommended` basis进一步与证据等级绑定：registered refutation至少需要S，descriptive-only至少需要D，
mechanical non-result至少需要M，position/measurement confound需要M或D，insufficient-causal-evidence则保留
D/S/N/G/U的不同“尚不足以支持方法”语境；空finding列表同样fail closed。这样basis不只是枚举标签，还必须
有语义相符的证据来源。

### Negative section must cover every weakened or rejected conclusion

`negative_and_conflicting_results`不再只要求列表非空。每条负/冲突结果必须引用至少一个finding；所有
weakened组件以及weakened/rejected H0--H5都必须与该清单共享实际supporting finding，否则最终报告拒绝
生成。新增“组件矩阵已weakened但负结果章节改引另一finding”和空finding条目反例后，comprehensive定向
回归为`57 passed`。这保证负结果与冲突不因优化方向排序而从正文中消失。

### Reproducibility appendix now retains every audited command and byte identity

closeout run declaration现从每个已审计metadata保留原`command`，formal report的evidence admission同时保留
全部run declarations。comprehensive最终生成时构造独立reproducibility ledger，包含：冻结plan/manifest/
control资产、19个formal deliverable、21个supplement、逐run metadata path/SHA/阶段/模型/状态/eligible/
command，以及dev-eval ledger身份和source/qrels审计标志。

完成且result-eligible的formal run若缺command、任一SHA不是64位十六进制、deliverable集合不等于注册19项、
run ID重复或文件状态非completed，最终报告都会fail closed。Markdown附录实际渲染全部文件身份和逐run命令，
不再只链接一份上游formal report。comprehensive/closeout/formal-builder定向回归为`91 passed`；没有重新执行
命令，也没有打开score或qrels读取科学效应。

supplement registry audit也开始从每个已审计输出保留并验证`command`；17个既有输出均已具备，4个pending
生成器也已在代码中写入`sys.argv`。comprehensive ledger与Markdown现在同时显示21项supplement命令；缺失或
空命令会fail closed。supplement/comprehensive/closeout/formal-builder联合回归为`95 passed`。

以上新增报告门合并后，全仓回归为`869 passed, 7 subtests passed`，`git diff --check`通过；deep-dive、
necessity V2、supplement registry及comprehensive plan/manifest七个冻结SHA均与注册值逐字节一致。

### Untested is derived from registered coverage, not chosen by the writer

最终72格矩阵现从formal deliverable的固定component/model topology与supplement registry scope自动构造
`registered_evidence_sources`。某格没有任何直接注册source时必须标`untested`；只要至少一个formal或
supplement直接覆盖，就不得标`untested`，即使科学结论仍可保持`unresolved`。Q0--Q3模型边界的
`uncovered_components`继续要求与这些untested格精确一致，Markdown每格显示source ID或
`not-directly-registered`。新增双向伪状态反例后，comprehensive定向回归为`59 passed`。

18行聚合组件状态也从72格按`supported > weakened > unresolved > mechanical_failure > untested`的保守优先级
校验：逐模型格已有支持/削弱时聚合行不能隐藏为unresolved；四模型全未覆盖时聚合行不能伪装成unresolved；
supported/weakened/mechanical聚合还必须给出非空model scope，并与对应格的status/level一致。这样功能链与
H0--H5使用的聚合状态不会和逐模型表分叉。

### Q2 RoPE block 13 replacement mechanically closed

Q2 RoPE block-13 v2已覆盖8,000个请求并完整闭合，其中7,254个为冻结content-neutral eligible请求；输出
160,753个条件score，`complete_finite_score_coverage=true`、`identity_passed=true`、最大identity与冻结
baseline delta均为0。common-offset低精度bound最大ratio为`0.00000770432811096255`且通过。最终score SHA为
`34223f83124c34a7f69ed3010b599b7632fca362fa2f252b3290a3f133604051`，实际文件SHA与metadata一致；
`qrels_read=false`、`source_test_opened=false`。这里只审计机械字段，未读取RoPE科学contrast。

GPU2随后遇到上午遗留的Q2 attention-group block-20 orphan partial：196/512行已同步落盘，但旧进程消失且
metadata仍为running，18:04回填队列因防并发写入规则拒绝接管。确认无活进程/锁后，直接使用scorer的注册
`--resume`路径；该路径在模型追加前重新审计row index、selection SHA、8-group覆盖、supplemental condition
集合与partial SHA。恢复已把196行绑定进`resume_lineage`并继续追加，另挂completion-gated watcher在其封口
后重启原短回填队列。该孤儿状态与恢复只算机械事件，不进入机制结果。

当前全仓回归为`870 passed, 7 subtests passed`，`git diff --check`通过。D2仍为44/60终态包，按请求量执行
进度已到`0.764267483409903`；formal 6/19、supplement 17/21，source test保持关闭，未读取未终态效应。

### The worksheet initializer now pre-fills all registered coverage without inferring outcomes

最终worksheet初始化器现直接读取冻结19项formal topology与21项supplement registry metadata，自动预填
18组件×4模型的72格注册覆盖及每模型`uncovered_components`。有直接注册source的格保持`unresolved/U`，
没有直接source的格标记`untested/U`；该步骤不读取score、metric、qrels或科学status，也不会改变传入template。
独立回归验证了72格完备、每模型未覆盖列表精确相等、聚合untested规则及输入不变性。实际smoke初始化显示
Q0/Q1各有8个未直接覆盖组件，Q2/Q3为0；这是注册实验边界，不是模型能力结论。

comprehensive/supplement/closeout/formal-builder联合回归为`97 passed`，随后全仓回归为
`871 passed, 7 subtests passed`，`git diff --check`通过。运行时attention-group block-20恢复继续前进到
263/512；Q2 block-23为3080/3918，Q3 block-17/18分别为1489/3918与1503/3918，四项均仍为running，
未读取任何效果字段。

### The final report now renders the complete localization evidence instead of only carrying it in JSON

comprehensive payload此前已经从formal report携带D2完整逐层与相邻节点profile，但Markdown第3节只显示
human narrative和“层扫描仅用于定位”的边界，未实际展开60个post-block格、56个相邻层变化、4个跨模型/
endpoint shape summary与24个相邻功能节点变化。这不改变机器证据，却会使最终读者无法核对“完整曲线、
局部/分布式形态和后续组件窗口”要求。

现将上述四张表逐行渲染，同时继续声明绝对block index只属lineage，功能节点与有符号因果行为才是跨模型
对齐单位。冻结观察也新增固定六行scope contract，逐项覆盖recurrence、strict transfer、other overlap、
full/null/wrong-user以及单seed/回溯人口/Q0--Q3边界；自由文本不能再让这些定义从最终报告消失。
该变更不重算结果、不读score/qrels、不新增实验family。定向回归仍为`97 passed`，全仓为
`871 passed, 7 subtests passed`，`git diff --check`通过。

同次机械快照中，attention-group block-20恢复到305/512；Q2 block-23为3375/3918，Q3 block-17/18
分别为1614/3918与1628/3918，均保持running和eligible，未读取效果字段。

### Every Markdown conclusion now resolves to an explicit finding and boundary

最终Markdown此前引用finding ID却没有finding ledger，逐模型边界没有显示untested组件，H0--H5、负结果、
not-recommended与五层解释也省略了各自finding引用；优化机会表则没有展示mechanism target、minimum level、
contradictory evidence、do-not-infer和diagnostic-patch flag。虽然这些字段都存在于JSON，这会使Markdown无法
独立审阅证据链。

现新增完整finding ledger（等级、模型、claim、formal/supplement source、冲突、禁止推断）、18组件聚合表，
并在72格、功能失败分类、五解释层、四模型边界、H0--H5、负/冲突结果和不建议方向中显示对应finding。
七个保留机械non-result逐run列出；优化机会表完整显示注册合同字段与evidence bytes；论文claim节逐项渲染
固定invariant并明确诊断patch、target margin与NDCG效用的边界。变更只影响报告可审阅性，不推断结果。

联合定向回归为`97 passed`，全仓为`871 passed, 7 subtests passed`，`git diff --check`通过。同期outcome-
independent overview登记84个formal run单元：74 completed integrity-checked、5 mechanical failure、4 running、
1 wall-time exhausted；17/21 supplements已完成。D2按请求量为`0.7722264760932448`，未读取科学效果。

### Q2 fixed-confirmation block 23 mechanically closed

Q2 post-block block-23 fold-1完成3918/3918请求，`complete_finite_score_coverage=true`、
`identity_passed=true`、`result_eligible=true`，qrels与source test均未打开。最终score文件SHA为
`0d48cd95b84211f97a68ad11ee2ee2d224386035892deeea38b6618d87a3b919`，与metadata逐字节一致；这里只
检查机械字段，没有读取full/null或相邻层科学contrast。原Q2 formal queue已自动进入block-24。

D2固定终态包由44增至45/60，请求加权执行比例为`0.7740258635358176`；当前in-flight为Q2 block-24、
Q3 block-17/18，Q2 block-27保留1807/3918断点并将在队列轮到时恢复。source test保持关闭。

### Comprehensive admission now proves every localization row is present

最终comprehensive builder现不再只信任formal report的`status=completed`。admission明确校验Q2/Q3×
target-margin/NDCG两个endpoint的4个shape summary、60个all-layer cell、56个adjacent-layer cell与24个
adjacent functional-node cell的精确笛卡尔覆盖，重复、缺失或多余任一行均fail closed。所有shape继续要求
`layer_scan_alone_authorizes_design=false`，逐层行要求不作为primary component attribution，功能节点行要求
`literal_hidden_state_sign_reversal_claimed=false`。Markdown同时显示两张profile的source path与SHA。

新增缺失all-layer行和伪造literal reversal反例后，联合定向回归为`98 passed`；全仓为
`872 passed, 7 subtests passed`，`git diff --check`通过。该审计检查结构和claim boundary，不读取新效果。

### Cross-model synthesis is functional and evidence-bound, not an index narrative

最终worksheet与Markdown新增结构化`cross_model_synthesis`，分别要求至少一条shared pattern与一条
heterogeneous pattern。每条必须覆盖至少两个冻结模型，列出功能组件、证据等级、finding、剩余外推边界；
每个声明组件×模型都必须由同等级、component/model-matched的finding支持。只引用另一个组件、只覆盖一个
模型、等级不匹配或在summary中用绝对block/layer编号对齐都会fail closed。

Q0--Q3逐模型untested列表继续保留，新的cross-model表只表达有直接证据的功能共同性/异质性，不能用Q2/Q3
结果填补Q0/Q1，也不能把相对深度相似写成固定层号方法。新增单模型、错等级、错组件和绝对block编号反例后，
联合定向回归为`99 passed`；全仓为`873 passed, 7 subtests passed`，`git diff --check`通过。

### CCF-A method readiness is separated from mechanism-report completion

论文claim节新增固定五门表：functional causal target、independent method instantiation、preregistered utility、
replication/generalization与baselines/ablations/efficiency。每门同时写明未来所需证据与当前stage边界：G级功能
节点只可排序候选，单向patch不是方法；target-margin mediation不是NDCG收益；单seed回溯KuaiSearch不能建立
泛化；source test只有在新授权/协议后才可打开；当前不生成方法比较表。

该表不新增实验或暗示当前已达CCF-A方法证据，只防止最终机制报告把“最可能方向”写成“已验证方法”。联合定向
回归保持`99 passed`，全仓`873 passed, 7 subtests passed`，`git diff --check`通过。

### Q2 attention-group block 20 orphan recovery mechanically closed

恢复任务完成全部512个注册row，`identity_passed=true`、最大identity与冻结baseline delta均为0、
`result_eligible=true`，qrels/source test均未打开。最终`groups.jsonl` SHA为
`304594551398e4ca44570d0e6e375aa3ed4ef734b1643025ada414648e35423e`，与metadata一致；未读取group
科学contrast。completion-gated watcher随后正确跳过已完成的Q2 b20 attention-head/group，启动
Q2 MLP-group block-20，确认孤儿恢复没有造成重复writer或GPU2空转。

恢复闭合后的outcome-independent census为86个formal run declaration：76 completed integrity-checked、
5 mechanical failure、4 running、1 wall-time exhausted；D2固定包45/60，请求加权执行比例
`0.7817168623447337`，formal 6/19、supplement 17/21。七个冻结plan/manifest/registry SHA均与注册值一致。

### Cross-model and opportunity claims cannot outrun the 18x4 matrix

跨模型pattern现与逐模型边界和72格双向绑定：每个pattern model必须在对应model boundary引用至少一个该
pattern finding；S/N/G pattern要求每个声明component×model格均为同等级`supported`，M/U也分别与
mechanical/unresolved状态相容。D级pattern可描述共同/异质几何，但不会把组件格升级为因果支持。

优化机会增加同一一致性门：`actual_evidence_level`为S/N/G时，对应18组件聚合行必须同等级supported，且机会
model scope不能超过组件scope。于是即使某项direct evidence存在，只要最终组件矩阵因冲突仍保留unresolved，
优化表就不能偷偷写成causal/design-qualified。新增model-boundary缺引用、因果pattern/cell冲突和causal
opportunity/matrix冲突反例；定向回归`99 passed`、全仓`873 passed, 7 subtests passed`，diff检查通过。

### Every finding now carries audited evidence bytes

最终builder原先只给optimization opportunity自动绑定formal/supplement path与SHA；finding ledger虽显示证据
ID，却仍需读者去附录人工映射。现同一binding函数在不修改validated worksheet的前提下，为每个finding按
formal-first、supplement-second固定顺序附加`supporting_evidence_identities`，逐项包含kind、path与64位SHA。
缺失终态identity、伪SHA或引用顺序漂移均fail closed。Markdown finding ledger直接显示`evidence_id@sha`。

新增finding identity覆盖、输入不变和伪SHA反例后，定向回归保持`99 passed`，全仓
`873 passed, 7 subtests passed`，`git diff --check`通过。

### Q2 MLP-group block 20 mechanically closed and GPU2 advanced to block 27

Q2 MLP-group block-20完成512个注册row，`identity_passed=true`、`result_eligible=true`、qrels/source test
均未打开；最终`rows.jsonl` SHA为
`53947aa4a1099d24eedcca12fce995c91ad5b203d58c9fcd667cd1c70ad53d47`，与metadata一致。没有读取
MLP科学contrast。短回填队列随后跳过已完成的Q2 b27 attention-head，启动Q2 b27 attention-group，GPU2
继续服务已注册breadth，不与selected-branch contract抢占。

### Negative evidence retains experiment coordinates and H0--H5 basis

每条negative/conflicting result现必填model、endpoint、surface、contrast、fold与seed scope，并要求model
scope不超过其finding证据。endpoint/surface/contrast/fold使用固定注册枚举，任意“best-looking slice”或重复
seed都会fail closed；Markdown用表格逐项显示，机械non-result仍在其后逐run保留。

H0--H5每行新增结构化`negative_evidence_basis`：registered refutation/weakening、跨模型或endpoint冲突、
测量/人口不稳定、因果证据不足。rejected必须带registered-refutation，非rejected不能伪称已反证，weakened
必须说明实际削弱来源。新增缺scope、非法surface、越界model、重复seed、非法basis和非rejected伪反证反例。
定向回归`101 passed`，全仓`875 passed, 7 subtests passed`，`git diff --check`通过。

### Every registered artifact now has an audited producer path

既有queue topology只证明bash语法、GPU静态所有权和canonical metadata单写者，没有证明19个formal deliverable
与21个supplement在脚本重命名或队列调整后仍各自拥有项目自有producer。新增outcome-independent producer
topology，将每个固定output path绑定到producer entrypoint、direct/queue/watcher入口与上游family；输出ID、路径、
producer、orchestrator或数据边界任一漂移都会fail closed。专用queue还必须逐字绑定对应producer，generic watcher
只能作为显式登记的通用入口。

真实仓库审计为formal `19/19`、supplement `21/21`，其中18项通过queue/watcher接管，failure为0；该门已接入
comprehensive readiness，但不改变任何科学family、效应判定或优先级。故障注入覆盖output-path drift、producer
丢失、专用orchestrator解绑和source-test越界。相关定向回归`20 passed`，全仓
`879 passed, 7 subtests passed`，`git diff --check`通过；审计不打开score、qrels或source test。

### Opportunity ranking now carries a per-candidate method-stage contract

对照mechanism M4与comprehensive plan后，原机会表虽有mechanism target、证据、风险和证伪门，却只在全局
CCF-A门槛中提到训练信号、消融和基线，不能保证每个候选方向分别回答这些问题。现每个opportunity必填
`hypothesized_innovation`、`training_signal_requirements`、`key_ablations`、
`closest_baseline_families`与`baseline_differentiation`；上述字段同样禁止绝对layer/head/neuron编号。
`architecture_implemented=false`同时成为逐机会字段和全局claim invariant，防止机制候选被写成已实现方法。

最终JSON/Markdown另新增冻结13-section contract：逐项绑定执行表、scope、完整层轨迹、18×4矩阵、功能因果链、
五解释层、Q0--Q3边界、H0--H5、负结果、机会、不建议项、claim边界和复现附录的实际payload路径；任一为空
即fail closed。comprehensive report plan本身以固定SHA直接进入复现ledger。新增缺字段、伪实现状态和缺章节反例
后，相关定向回归`78 passed`，全仓`886 passed, 7 subtests passed`，`git diff --check`通过；未新增实验family
或读取效应值。

### Free-text narratives are now evidence-bound instead of an escape hatch

此前finding、18×4格、五层解释、跨模型pattern和机会均绑定证据，但五段narrative仍是裸字符串；这允许在
executive summary或paper boundary中手写一个没有finding/SHA的结论。现`executive_summary`、冻结scope、
layer interpretation、cross-model boundary与paper claim boundary均必填`evidence_level`、非空
`supporting_findings`和`do_not_infer`。非U等级必须存在同等级finding；最终binding会对叙述引用的所有finding
去重并展开formal/supplement path与SHA，Markdown同时显示finding和evidence bytes。

自由文本禁止手抄`NDCG@10=...`、margin、CI、p/q或effect-size数值，数值只保留在admitted evaluator表；摘要、
跨模型与paper boundary也禁止绝对layer/block/head/neuron编号。逐层trajectory仍可把层号作为lineage而非设计。
新增空finding、等级升级、手抄metric、跨模型绝对层号和合法trajectory lineage反例后，定向回归`79 passed`，
全仓`887 passed, 7 subtests passed`，`git diff --check`通过。

### All human interpretation fields now inherit the same numeric/index boundary

进一步递归审计发现，即使五段narrative已封闭，finding claim、组件/模型summary、功能链diagnosis、H0--H5、
negative、机会与not-recommended字段仍可能手抄metric或把绝对层号写入结论。现统一枚举所有human-
interpretation字段并递归遍历mapping/list：出现NDCG、target margin、CI、p/q或effect-size的字面数值一律
fail closed，数值只能来自admitted evaluator table；出现带数字的layer/block/head/neuron也一律失败，唯一例外
是`narratives.layer_trajectory_interpretation.text`中的lineage描述。

新增finding手抄NDCG、组件summary层号和H0不确定性手抄CI反例；同时验证trajectory层号仍合法。叙述finding
binding还会去重并展开直接evidence path/SHA，Markdown显示这些bytes。定向回归`80 passed`，全仓
`888 passed, 7 subtests passed`，`git diff --check`通过。

### Q2 fixed-confirmation block 24 mechanically closed

Q2 post-block block-24 fold-1完成3918/3918请求与78,864个有限score row，
`complete_finite_score_coverage=true`、`identity_passed=true`、`result_eligible=true`，qrels/source test
均未打开；最大identity delta为0，冻结baseline low-precision ratio为0.0625。最终`scores.jsonl` SHA为
`fff3fb19f8b027bb9d25aa2fb5f6183cfff469b9270f465c84448718d86da8d2`，与metadata逐字节一致。这里只读取
机械字段，没有打开或解释科学contrast。

D2固定终态由45增至46/60，原Q2 queue在约20秒内自动进入block-25；同一快照中Q3 block-17/18分别为
3454/3918与3490/3918，Q2 block-27保留1807/3918断点。outcome-independent audit错误为0。

### Every admitted evidence item now receives an explicit scientific disposition

复现ledger已能列出19个formal与21个supplement的路径、SHA和命令，但此前仍允许某个完成证据只停留在附录，
既不进入finding/negative，也不解释为什么它不产生科学claim。最终worksheet现新增精确40项
`evidence_disposition`账本；每项只能标为`interpreted_in_findings`、`negative_or_conflicting`或
`bounded_no_scientific_claim`，并必填summary与`do_not_infer`。

前两种处置必须引用实际逐字引用该证据ID的finding；冲突处置还必须进入注册negative/conflicting表。第三种不得
引用finding，且任何已经被finding使用的证据也不能反向伪装成“无科学claim”。最终binding为40项逐一附加
kind、path和SHA，执行/证据章节将完整渲染该账本；13-section coverage也把它作为必需payload。缺项、证据种类
漂移、无关finding、隐藏claim、冲突漏表与伪identity均fail closed。模板机械核对为formal 19、supplement 21、
共40个唯一ID；相关定向回归`83 passed`，未读取效应字段、qrels或source test。

### Functional causal chain now exposes every primitive S/N/specificity/control/G gate

冻结comprehensive plan要求功能因果链分别列S/N/G，但原Markdown每个节点只显示一个汇总等级；虽然G在代码内由
原始门重推，读者仍看不到same-request充分性、position-preserving反向必要性、same-minus-wrong特异性、
cross-request与norm/direction/random控制各自是否通过。最终builder现把已审计
`component_functional_design_gate_synthesis`逐行规范成Q2/Q3×四功能节点的8行
`component_bidirectional_gate_matrix`，直接展示全部原始布尔门、combined state gate、设计资格与G门。

该矩阵不读取或重算效应值，只绑定supplement path/SHA；incoming state仍是上游control，block-output仍是完整状态
ceiling且强制`design_target_eligible=false`，不能回填为residual operator归因。admission audit新增claim-role、
design-eligibility和布尔类型校验，并继续从primitive gates重推state/G与跨模型节点。功能因果链章节把该矩阵设为
必需payload，明确N只表示移除已注册有害响应，G也不建立组件有益性、ranking收益、operator necessity或精确层设计。
相关定向回归`86 passed`；全仓`891 passed, 7 subtests passed`，`git diff --check`通过。

### Q3 fixed-confirmation block 18 mechanically closed

Q3 post-block block-18 fold-1完成3918/3918请求，`complete_finite_score_coverage=true`、
`identity_passed=true`、`result_eligible=true`，最大identity delta为0，qrels/source test均未打开。
最终`scores.jsonl`含3918个request级row，SHA为
`78becbfec0862c4d690cf9327aa8ee172a6504e1ad04c86ca8ef7d4e81a18920`，与metadata一致；未读取
candidate score或科学contrast。原Q3 lane自动进入block-20，另一lane的block-17同期已超过99%，队列接力正常。

Q3 block-17随后同样完成3918/3918请求，coverage/identity/result-eligibility全部通过，qrels/source test
保持关闭；`scores.jsonl`为3918个request级row，SHA
`3354d0f3802d4500b9bea95eb2cff98b7b3f4e57b98cbaea7319761385e93f92`与metadata一致。对应lane在约一分钟
内自动进入block-19。两项关闭均只核对机械元数据与字节身份。

### Model-scoped weakening cannot disappear behind a supported aggregate

18×4矩阵聚合优先级允许同一组件在一个模型supported、另一个模型weakened，此时聚合行合理地显示supported；
但原negative保留门只检查聚合行，可能让被supported遮住的单模型削弱不进入“全部负结果与冲突”。现为每个
component×model cell构造model-scoped negative finding集合；任一weakened格都必须在包含该model且引用同一
finding的negative row中出现。这样跨模型异质性既保留正证据，也不会删除相反模型/endpoint/surface的负证据。
新增“负表只写Q3却把Q2格标为weakened”的反例后，comprehensive定向回归`77 passed`；全仓
`892 passed, 7 subtests passed`，`git diff --check`通过。

### Q2 attention-group block 27 mechanically closed

Q2 attention-group block-27完成512/512个注册row，identity与冻结baseline最大delta均为0，
`result_eligible=true`，qrels/source test均未打开。`groups.jsonl` SHA为
`6d6bbbb2cecb4934dd79bbdfd2dd6d96d575811971ab45e2bc1d1fd946d495f3`，与metadata一致；未读取组效应。
短回填队列随后自动启动Q2 MLP-group block-27。Q2 D2 block-27的1807/3918断点仍由其canonical主队列持有，
没有重复writer。

### Negative design recommendations are component- and model-bound

优化机会已要求component/model-matched evidence，但`not_recommended`原先只有方向文本、basis和finding等级，
理论上可用表示层finding否定一个未被测试的attention或readout方向。现每项不建议方向必填
`functional_component`、`model_scope`、`dataset_scope=kuaisearch_dev`与`source_test_opened=false`；每个声明模型
都必须存在basis等级匹配且真正覆盖同一组件/模型的finding。Markdown同时显示组件、模型、数据边界和finding。
新增无关readout组件、越界Q0/Q1 scope与source-test越界反例，comprehensive定向回归保持`77 passed`。

### Optimization opportunities must be interpreted by component/model-matched findings

机会表原先绑定formal/supplement ID，却仍可能直接引用一个finding没有实际解释的额外证据，或用只覆盖部分模型的
finding扩张机会的model scope。现每个opportunity必须列出非空`supporting_findings`；其全部直接证据ID必须被
这些finding逐字引用，并且机会声明的每个模型都必须存在actual-evidence-level、component和model同时匹配的
finding。组件证据本身的model coverage仍先独立校验，因此finding绑定不能替代底层证据覆盖。

fixture新增专用`F_OPPORTUNITY`，避免一般描述finding被误当作机会finding；负结果fixture同时保留基础与机会
finding，使component/model-scoped weakening不会因测试变体而被意外删除。新增空finding、未解释的额外formal
证据和证据等级不匹配反例。comprehensive定向回归`78 passed`，三项报告组合回归`88 passed`，全仓
`893 passed, 7 subtests passed`，`git diff --check`通过；没有读取效果值、qrels或source test。

### Failure-mode labels are now bound to functional components and models

最终`failure_mode_diagnosis`此前只约束finding等级与model scope，理论上仍可把表示finding写成readout或
optimizer失败。现任何非`unresolved`模式必须列出`functional_components`；每个组件必须由diagnosis引用的
finding在至少一个声明模型上直接覆盖，每个声明模型还必须拥有与该failure-mode语义一致的组件证据。训练失败
必须落在loss/optimizer/LoRA组件，readout失配必须含native readout或score-nullspace，candidate transport必须
含routing/candidate interaction；`multiple_bottlenecks`则要求每个模型至少跨两个独立五层系统层面。

报告同时输出派生的component→model evidence map，防止全局组件列表隐藏单模型异质性；`unresolved`模式不得
反向断言功能组件。新增空组件、表示证据伪装训练失败和无finding训练组件反例后，comprehensive定向回归
`79 passed`，三项报告组合回归`89 passed`，全仓`894 passed, 7 subtests passed`，`git diff --check`通过。

### Idle GPU 2 resumed the canonical Q2 block-27 bundle

Q2 MLP-group block-27机械关闭后GPU 2短暂空闲，而Q2 D2 block-27停在1807/3918。该bundle仍是冻结注册单元，
与主Q2 queue共享canonical metadata writer lock；主queue当时仍在block-25且后续还有block-26。使用同一
`run_deep_dive_resume_loop.sh`和同一atomic run目录在physical GPU 2恢复，不创建新run、层、seed或选择，锁仍防止
任何双写。恢复后四卡均活跃，block-27已推进至2186/3918；同期Q2 block-25为2572/3918，Q3 block-19/20为
502/3918与541/3918。审计仍未读取科学效应、qrels或source test。

### Failure maturity and opportunity polarity can no longer overstate evidence

失败类型现在额外派生`diagnostic_resolution`：D=`descriptive_candidate`、S=`sufficiency_candidate`、
N=`necessity_mediator_candidate`、G=`bidirectionally_supported_failure_path`，U保持`unresolved`；机械M证据不能
命名科学failure mode。`multiple_bottlenecks`的跨两层约束新增专门反例，避免同一representation层内两个组件被
包装成多层瓶颈。

机会表方面，`utility_gain_established=false`与有害full-history注册行为共同意味着当前不能推荐
`preserve_or_strengthen_beneficial_state`；该polarity现一律fail closed。M级机械证据只能进入deprioritized或
not-recommended且保持diagnostic-only，不能排成科学优化候选。新增机械failure、单层伪多瓶颈、无收益却建议加强
状态和M级机会排序反例后，comprehensive定向回归`82 passed`，三项报告组合回归`92 passed`，全仓
`897 passed, 7 subtests passed`，`git diff --check`通过。

### Causal-role coverage is explicit in the 18×4 matrix

readiness已区分“任意artifact覆盖”与“注册/完成causal-role artifact”，但此前逐组件/逐模型主表只显示一般证据
ID，读者仍可能把描述几何当作S/N覆盖。最终payload现新增精确18项
`component_evidence_role_coverage`，逐组件绑定任意证据完成、causal-role注册/完成、Q2/Q3双模型完成以及注册/完成
model scope；所有布尔值、模型子集和14/18、3/18等summary count均由readiness反向核对。

Markdown aggregate与18×4格分别显示causal role注册/完成状态。当前机械审计明确指出token embedding、MLP feature
formation、optimizer effective update与LoRA parameterization只有描述/训练动力学覆盖，不能被写成因果已解析；
该状态本身不推断科学支持。覆盖漂移、完成scope越界与count伪造均fail closed。

### Five system layers and model boundaries are evidence-bearing claims

input/representation/routing/readout/training五层解释原来只在引用finding时检查跨层，却允许无status/level的自由
诊断。现每层必填`status`、`evidence_level`和`model_scope`；supported/weakened/mechanical结论必须在每个模型上
同时匹配同状态、同等级、同组件单元与同finding，unresolved严格为U。Markdown逐层显示状态、等级与模型范围。

Q0--Q3模型边界也增加直接证据门：只要某模型已有任何完成formal/supplement组件证据，summary就必须引用该模型
finding；无直接证据时才允许空引用。另将not-recommended的`registered_refutation`绑定同模型S级weakened格与
negative表，`mechanical_non_result`绑定M级mechanical-failure格，G级支持不能伪称“因果证据不足”。新增跨层借证、
层状态漂移、模型空finding和basis/status错配反例后，comprehensive定向回归`84 passed`，三项报告组合回归
`94 passed`，全仓`899 passed, 7 subtests passed`，`git diff --check`通过。

### Opportunity ranking is deterministic within each evidence class

综合机会原来只强制design-priority大类顺序，同一类内仍可按worksheet输入任意排序。现固定组内key为
`design priority → evidence strength (G > S/N > D > U > M) → model coverage`；较低证据或较窄模型scope不能排在
同类更强、更广候选之前，完全同key才保留人工顺序。`candidate_to_test`只接受D/S/N，U/M必须降级；输出逐项显示
evidence tier与ranking basis。跨模型范围反向排序与U级候选反例均fail closed。

正式deep-dive report固定的五项架构机会现在以原rank/status/model/evidence/rationale/falsification gate完整进入最终
第10节，随后才展示组件门控后的综合新排序。二者明确是“冻结formal候选”与“更严格综合更新”，不能用省略表示
反证或降级；section contract同时要求两张表非空。该批综合回归`94 passed`、全仓
`899 passed, 7 subtests passed`，`git diff --check`通过。

### GPU 2 will backfill registered Q3 block 25 after Q2 block 27

为避免GPU 2在Q2 block-27关闭后空闲，已挂outcome-independent watcher：只等canonical Q2 b27 metadata完成，
随后在同一冻结fold-0 selection、Q3 gate和fold-1协议下运行注册Q3 b25。没有抢b21，因为b21可能与正在运行b19的
主lane发生时间重叠；b25前还有b21/b23，具备充分安全裕量，canonical writer lock继续兜底。该调度不读取效应、
不新增run family/层/seed或选择。

### Q2 fixed-confirmation block 25 mechanically closed

Q2 post-block block-25 fold-1完成3918/3918请求，`complete_finite_score_coverage=true`、
`identity_passed=true`、`result_eligible=true`，最大identity delta为0，qrels/source test均未打开。
`scores.jsonl`为3918个request级row，实算SHA
`06f5e1155f53dbac262f5e74c51b1eecaefe6350913c36ac5a869c505272eb7f`与metadata逐字节一致，耗时
4541.996秒；未读取candidate score或科学contrast。主Q2 queue自动接力block-26。

D2固定终态由48增至49/60，固定请求加权执行84.11%；包含最多2个条件分支的最大科学单元resolution为
79.03%，请求加权上限口径为81.39%。同期Q2 b27为3585/3918，Q3 b19/b20为1037/3918与
1079/3918，outcome-independent错误为0。

### Formal-opportunity dispositions are mechanism-family bound

精确一次的predecessor ID覆盖仍可能把梯度预算映射到纯表示证据。现映射到综合opportunity的每个formal ID都必须
与目标条目的直接formal evidence在其冻结`OPPORTUNITY_ALLOWED_DELIVERABLES`内相交；映射到not-recommended时，
basis-level finding的formal证据也必须相交。fixture以d1 representation、d2 selected branch与d5 context覆盖
冻结五项所需family，同时仍由专用`F_OPPORTUNITY`解释，不能借一般F1扩张。

新增移除H4所需family和把H1 routing塞进纯表示not-recommended的反例；comprehensive定向回归`85 passed`、三项
报告组合`95 passed`、全仓`900 passed, 7 subtests passed`，`git diff --check`通过。

### Every frozen formal opportunity has exactly one comprehensive disposition

仅并列展示formal五项仍允许综合报告不解释其去向。现每个综合opportunity与not-recommended条目分别声明
`formal_predecessor_ids`；五个冻结`OP_H*` ID必须在两类目标中精确出现一次，可多对一合并，但不得遗漏、重复或
同时保留/否定。builder派生`formal_opportunity_disposition`，逐项输出
`mapped_to_comprehensive_opportunity`或`mapped_to_not_recommended`及目标ID；完全新方向可保持空predecessor。

第10节现在依次显示冻结五项、精确处置表、组件门控综合排序。遗漏、重复和合法转入not-recommended反例后，
comprehensive定向回归`85 passed`，三项报告组合回归`95 passed`，全仓
`900 passed, 7 subtests passed`，`git diff --check`通过。

### Q2 fixed-confirmation block 27 mechanically closed and GPU 2 handed off

Q2 post-block block-27 fold-1完成3918/3918请求，coverage/identity/result eligibility全部通过，最大identity
delta为0，qrels/source test保持关闭。`scores.jsonl`为3918行，实算SHA
`24105dcb2fe8ee881a6d5e4e2df002d8f5d21157f479919e25bf6a6dc2aefd76`与metadata一致，累计耗时
4533.330秒；未读取科学contrast。

预挂watcher随后按合同启动Q3 fold-1 block-25，首个机械快照为12/3918，physical GPU 2恢复活跃。D2固定终态
由49增至50/60，固定请求加权84.55%；最大62科学单元resolution为80.65%，请求加权上限口径81.82%。同期Q2
b26为405/3918、Q3 b19/b20为1207/3918与1239/3918，错误为0。

### GPU 2 selected-branch watcher collision was removed before launch

Q2 selected-branch shard-1的旧watcher原本只等待Q2 selected-branch contract与Q2 RoPE block-13；GPU 2已被
outcome-independent backfill用于注册Q3 block-25，因此若Q2 contract先生成，旧watcher可能在同一卡并发启动两个
scorer。审计确认旧watcher仍仅在`sleep 30`、Q2 contract尚不存在且selected-branch scorer没有启动后，只终止该
睡眠watcher及其sleep子进程，并以相同目标队列重挂；新watcher额外等待canonical Q3 block-25 metadata完成。没有
停止、重跑或更改任何实验，也没有新增层、seed、family或outcome-dependent选择。

修正后四卡利用率为96%--99%。机械D2审计为50/60个固定bundle完整、4个运行、6个尚未启动；按已处理请求计
固定执行51.187/60（85.31%），最大62科学单元口径51.187/62（82.56%）。运行快照为Q2 b26
1208/3918、Q3 b19 1536/3918、Q3 b20 1591/3918、Q3 b25 316/3918；进度审计继续声明未读取科学效应、qrels
或source test。综合报告定向回归`95 passed`，全仓`900 passed, 7 subtests passed`，`git diff --check`通过。

同一机械readiness快照显示40项正式/补充artifact中23项完成（57.5%）；21项补充证据已完成17项（80.95%），
19项正式deliverable完成6项（31.58%），剩余正式项多数由D2终态、selected-branch与其后评估依赖。18个组件中16个
至少已有一种完成artifact，14个已注册因果角色检验；只有3个目前拥有完成的单模型因果角色artifact，尚无组件在
Q2/Q3两模型都完成因果角色闭环。因此这些数字只表示证据可用性和队列进度，不能提前解释成科学支持或设计资格。

### Exact implementation interfaces can no longer hide behind 18 aggregate components

18类科学组件对报告解释足够稳定，但会把真实Qwen实现中的多个不同接口聚在一起。新增outcome-independent
`transformer_internal_interface_coverage`，精确登记32个接口：serialization/tokenization、tied embedding/readout
rows、causal mask、Q1 KV-cache phase、block input、两类RMSNorm、Q/K projection与QK norm、RoPE前后、V path、
attention logits/softmax edges、pre-O head output、16Q→8KV GQA grouping、O projection、两次residual、SwiGLU gate/up/SiLU/product/down、
final RMSNorm、candidate readout positions、native score、loss、AdamW effective update与LoRA path。

每项接口绑定正式/补充证据ID、角色（描述、edge intervention、S、N、G或训练动力学）、模型scope、完成状态与claim
boundary；readiness和最终Markdown均显示该清单。当前机械快照为32项中22项至少已有完成证据、23项已注册因果角色
证据、5项已有完成因果角色证据，只有2项的全部注册证据均已完成；所有计数明确声明不能推断科学支持。新增3项
专用测试并扩展readiness/report回归后，全仓`903 passed, 7 subtests passed`，`git diff --check`通过。

readiness同时显式列出尚无注册因果角色检验的9个精确接口：GQA grouping、Q1 KV-cache phase、MLP gate、up、SiLU、SwiGLU product、training loss、
optimizer effective update与LoRA adapter path。这些MLP formation及训练参数化证据仍是描述/训练动力学，不能
伪装成operator causality；这正是最终报告必须保留的unresolved boundary，而不是通过聚合18类组件隐藏。

同期D2固定请求加权执行升至85.91%，最大62单元请求加权口径83.14%；四个运行bundle分别为Q2 b26
1853/3918、Q3 b19 1794/3918、Q3 b20 1843/3918、Q3 b25 565/3918，机械错误为0。

### Live physical-GPU ownership is now machine-audited

此前producer topology只能证明每个注册artifact有生产者，不能证明运行时没有两条队列落到同一物理卡。新增只读
`gpu_ownership_audit`：仅扫描`/proc`中的project Python worker，联合解析`CUDA_VISIBLE_DEVICES`与`--device`，
检查每张物理卡最多一个活跃worker、每个run最多一个writer，并明确忽略只包裹命令的bash resume loop。它不读取
score、qrels或科学效应。

实时审计为4个worker精确映射到GPU 0/1/2/3：Q2 b26、Q3 b19、Q3 b25与Q3 b20；无同卡重叠、无run双写，
`all_expected_gpus_active=true`。新增4项审计测试后相关定向回归`95 passed`，全仓
`907 passed, 7 subtests passed`，`git diff --check`通过。

### Exact-interface evidence now has semantic component closure

接口登记若只校验evidence ID存在，仍可能把不属于该接口组件的证据借来制造覆盖。现每个formal引用必须属于接口
至少一个组件的`COMPONENT_ALLOWED_DELIVERABLES`；每个supplement引用也必须与冻结registry的component scope
相交，模型scope仍逐证据保留。真实registry闭包测试覆盖全部32接口。

该门当场发现并修正两处过宽绑定：Q1 KV-cache phase不能借Q0/Q1 branch sufficiency，因为该formal只覆盖
attention/MLP/residual输出而非KV phase；candidate readout positions不能借Q0/Q1 trajectory作为直接readout证据，
现改为绑定Q0/Q1 native final-readout family并显式包含native-readout组件。修正后精确接口为22/32至少有完成证据、
23/32注册因果角色、5/32已有完成因果角色；9项无注册operator-level因果证据被明确列出。相关定向回归
`96 passed`，全仓`908 passed, 7 subtests passed`，`git diff --check`通过。

最终Markdown的paper-claim section现单独列出上述9项operator-level因果债务，并逐项显示其组件与claim boundary；
描述或训练动力学覆盖不能再借“已有artifact”从18×4主表中静默升级。未来若把这些接口作为方法核心，报告明确要求
新的预注册干预和跨模型复现。该显示层定向回归`92 passed`，`git diff --check`通过。

### All 19+21 evidence now has exact-interface or cross-interface disposition

并非每项证据都应被强行映射到单个hook。接口审计现要求32项直接接口证据与8项明确cross-interface/scope-gate
证据不重叠且并集恰好等于全部40项：activation anisotropy、candidate block flow、candidate residual、preference
subspace、query causal floor、Q3 native gate、Q0/Q1 multi-branch breadth与Q0 all-layer trajectory均保留为跨接口证据。
每项仍显示模型scope、组件scope、完成状态与“不归因给单一operator”的边界。

真实readiness审计通过`32 direct + 8 cross = 40 registered`闭包；当前8项cross中前6项完成、后2项pending。
精确/跨接口处置测试及Markdown回归`92 passed`，`git diff --check`通过。

### Final completion is now a 12-requirement machine audit

新增终态审计器逐项验证：19 formal、21 supplement、D2 fixed/conditional终态、18组件artifact覆盖、32接口/40证据
处置、producer topology、source-test关闭、无readiness失败、final human worksheet、formal report、comprehensive
JSON的13节合同与Markdown。它只看完成字段、结构和文件SHA，不重算/选择科学效应，也不打开qrels。

当前真实审计为4/12完成：接口/证据闭包、producer topology、source-test boundary与无审计失败通过；其余8项保持
pending，主要由运行队列及其后human interpretation/report生成依赖。新增终态审计测试后相关定向回归
`93 passed`，全仓`909 passed, 7 subtests passed`，`git diff --check`通过。此审计将作为未来调用goal complete前的
必要证明，而不是用测试绿灯替代科学终态。

### Q2 fixed sweep closed and selected-branch decomposition started

Q2 post-block block-26 fold-1完成3918/3918请求，`score_rows=78864`、coverage/identity/result eligibility全部
通过，qrels/source test保持关闭，耗时4535.535秒。`scores.jsonl`为3918个request row，实算SHA
`b1270b5d91107de871c15ce64a93c3bd5b1f640f6050d9ded8ed69a5adb70997`与metadata一致。至此Q2全部15个
fold-1固定block终态，D2 fixed从50增至51/60；没有读取candidate score或科学contrast。

主队列随后完成fold-1 confirmation并按冻结gate生成Q2 selected-branch contract，branch scoring eligible；selected
block只作为lineage记录，不能进入设计结论。1-request GPU smoke完成，21个score cells coverage/identity通过、SHA逐字节
一致，`result_eligible=false`、qrels/source test均未读；GPU0已自动进入fold-1 shard0。GPU2 shard1仍等待注册Q3
block-25终态，避免同卡并发。实时ownership继续为4张物理卡各1个worker、无run双写。

### Conditional branch progress now counts request-sharded partial execution

原D2机械进度器只把selected-branch记为`pending_branch`或完整bundle，导致双分片已经处理的请求在最大62单元口径中
被低估。现仅从每个shard的metadata/progress读取注册fold总量、分片合同、已完成请求与run-contract SHA，按
`completed_requests / fold1_request_count`累加；不读取score、contrast、qrels或source test。它同时验证method、fold、
evidence mode、7个冻结node、二分片奇偶规则、结果资格与数据边界，合同漂移时fail closed。

真实快照为D2 fixed 51/60完整、3个运行、6个尚未启动；fixed请求加权52.811/60（88.02%）。Q2 selected-branch
shard0已处理42/1959，按完整fold为42/3918（1.072% bundle），shard1按预定依赖继续等待Q3 block-25；Q3 contract
尚未生成。因此最大62单元请求加权执行为85.20%，完整终态仍为51/62（82.26%），二者不再混淆。物理GPU审计
同时确认GPU 0为Q2 selected branch、GPU 1/2/3为Q3 block 19/25/20，4卡均活跃、无重叠、无双写。

### Nine exact operator-causality debts have explicit falsification gates

32接口清单此前能列出9项无注册causal-role evidence，但仅有自由文本claim boundary，仍可能在机会排序时被描述性
证据借道升级。现新增结构化debt ledger并由终态审计强制闭包：6项inference operator（GQA grouping、Q1
KV-cache phase与SwiGLU gate/up/SiLU/product），3项training mechanism（loss、optimizer effective update、LoRA）。

每项逐一记录当前evidence role/model scope、所缺最小证据、最小证伪门、
`unresolved_no_operator_causal_claim`、`can_rank_architecture_from_current_evidence=false`与
`active_experiment_authorized=false`。未来门只作为最终报告的未解决边界：例如GQA需保持Q/K/V与o_proj不变的grouping
intervention；SwiGLU需gate/up/nonlinearity/product成对identity、specificity、direction和reverse-removal；训练路径需同数据、
update budget和多seed效用控制。它不新增当前实验family，也不允许临时outcome selection。

最终Markdown现输出完整9行表，completion audit把`9=6+3`、接口ID精确相等和“不授权新实验”作为原12项
interface/evidence requirement的一部分。相关回归`93 passed`，`git diff --check`通过。

### The exact-interface inventory is now bound to the real frozen Qwen topology

静态32接口若不读取真实checkpoint/config，仍可能讨论未启用的泛Transformer模块。新增effect-blind frozen architecture
audit，逐字节绑定base config、Q2/Q3 method config、Q2 saved model config、Q3 adapter config及两份training metadata，
且只读取结构与训练合同，不读取score/effect/qrels/source test。

真实审计确认两条主路径共享Qwen3-0.6B：28 blocks、hidden 1024、SwiGLU intermediate 3072、16 query heads/
8 KV heads（每KV两个Q heads）、head dim 128、Q/K norm、RoPE theta 1,000,000、RMSNorm epsilon 1e-6、
tied embedding/readout、attention dropout 0、无sliding window与attention bias。Q2为full-parameter，Q3为rank-8、
alpha-16、q_proj/v_proj-only LoRA；Q3 trainable 1,146,880 / 597,196,800（0.1920%），两者均967 optimizer steps但
objective不同。

26项可由config直接证明存在/启用的实现接口全部包含在32接口清单中；其余接口由project-owned runtime hook、输入/
readout或phase合同证明。readiness与最终13节报告现嵌入该审计，terminal completion的interface requirement同时要求
architecture audit completed、0 failures与26/32 config-backed closure。真实completion仍为4/12，新增2项测试后相关
回归`91 passed`，全仓`912 passed, 7 subtests passed`，`git diff --check`通过。

同一审计现进一步绑定Q0--Q3四个canonical D0 identity smoke，而不是只凭config推断hook可用。四模型均在真实
SDPA backend上验证18个patch node/model（block-13全部16个branch/state接口加final RMSNorm前后），逐节点identity
score error为0、capture/wrapper no-op为0；block input、attention、O projection、SwiGLU、block output、RoPE norm与
final RMSNorm的BF16 algebra recomposition均在预定bound内。固定详细深度13/20/27只作执行覆盖，不作设计层号。

最终报告的reproducibility appendix现逐一列出7份model/config/training identity以及4份runtime smoke的path/SHA、
backend、hook count、identity error和recomposition状态；completion要求`4 models × 18 nodes` runtime closure。失败的
旧Q1 wrapper v2并不属于冻结formal mechanical-failure七项，现作为独立superseded runtime mechanical lineage保留：
`failed_identity`、最大identity error 14.8617大于1e-5 tolerance、result ineligible、qrels/source-test关闭，并逐字节
绑定canonical v3 replacement。它不能进入科学证据或增加formal run count，但不会从最终复现附录中消失。

### Tied parameters no longer conflate input embedding with native readout

五层聚合审计发现原32接口中的`token_embedding_and_tied_rows`虽共享同一权重矩阵，却把两个不同计算接口合并；
native lm-head readout intervention会因此错误地让input embedding看似已有causal-role coverage。现拆为
`token_embedding_lookup`（input）与`tied_lm_head_rows`（readout）。前者只保留representation/geometry描述证据，
后者绑定Q0--Q3 native-readout证据；readout结果不得借给embedding输入算子。

因此精确接口从32收紧为33，config-backed从26变为27，最低causal-role debt从9变为10（7 inference、3 training）。
新增embedding最小证伪门要求在token IDs、positions、mask不变时分别干预query/history/candidate embedding state，带
identity、same/wrong-history、scale/sign/random controls及Q2/Q3 replication。这个debt ledger现显式标记为lower bound；
不在表中的23接口也不会自动获得operator attribution。

最终报告新增五层量化表。当前真实机械状态为：input 2接口/1 causal registered/0 completed/1 debt；representation
10/6/0/4；routing 13/11/0/2；readout 5/5/5/0；training 3/0/0/3。该表只表示artifact role availability，绝不
等于科学支持。completion的exact-interface requirement已按33接口、10 debt、五层和tied-use分离更新并通过；相关
回归`95 passed`，`git diff --check`通过。

### Exact interfaces now separate artifact claim ceilings from scientific support

仅有`causal_role_registered`仍不足以区分D/S/N/G。现对11种interface evidence role作完整映射：mechanical confound
audit→M，descriptive/localization/training dynamics→D，context/edge/position/readout/sufficiency intervention→S，reverse
necessity→N，cross-model design gate→G。每个接口同时给出`registered_claim_ceiling`与
`completed_artifact_claim_ceiling`，但明确标记这些只是“若统计门全部通过时的最高允许级别”，绝不从artifact存在推断
实际scientific evidence level。

33接口的注册上限分布为D=10、S=19、N=1、G=3；当前已完成artifact上限分布为none=10、M=1、D=17、S=5、
N=0、G=0。尤其当前没有任何接口因为未来注册了G gate就被写成G级结果。五层表和逐接口表同时显示registered/
completed ceilings，terminal audit要求总数、role覆盖与`actual_scientific_evidence_levels_inferred=false`闭合。相关回归
`93 passed`，`git diff --check`通过。

### Frozen architecture closure now covers all Q0--Q3 configs and checkpoints

此前architecture audit只逐文件验证Q2/Q3，Q0/Q1仅由runtime smoke覆盖。现加入Qwen3-Reranker-0.6B base config、
Q0/Q1 method config、saved model config与training metadata；审计共绑定19个结构/运行文件。两个base artifact具有相同
28×1024/3072、16Q/8KV、RoPE/RMSNorm/SwiGLU拓扑，但Q0 specialized reranker使用不同冻结base weights与manifest，
不可与Q1--Q3 General Qwen当成同一checkpoint。

四路径均训练967 optimizer steps，但目标/参数化不同：Q0 pointwise BCE/full parameter；Q1 normalized candidate-response
NLL/full parameter；Q2 0.5 RankNet + 0.5 tie-aware ListNet/full parameter；Q3 recommendation-alignment NLL/rank-8 q/v LoRA。
最终报告现逐项显示base artifact、adaptation、objective和steps，不能再从相同block topology推断相同训练机制。

completion的exact-interface/architecture requirement要求2个base artifacts、4条精确model pathways、全路径967-step lineage、
4个runtime smokes与33接口闭合。真实审计0 failures，相关回归`91 passed`，`git diff --check`通过。

### Five distinct losses replace one misleading aggregate training interface

四路径闭合后确认原`training_loss_components`把Q0 BCE、Q1 normalized response NLL、Q2 RankNet/ListNet与Q3 alignment
NLL混成一个接口，隐藏了目标函数的真实异质性。现拆为5个model-scoped接口；Q2两个0.5-weight loss term分别绑定
objective/gradient evidence，Q3 loss只绑定Q3 training dynamics。跨Q2/Q3的common objective nullspace改列为第9项
cross-interface evidence，不能归到任何单一loss operator。

更重要的是，inventory现允许“implementation存在但registered evidence为空”。Q0 BCE与Q1 response NLL因此明确显示
`registered_evidence_count=0`、claim ceiling=`none`，而不是从Q2/Q3训练分析借证据。每个接口新增独立
`implementation_model_scope`，Q1 KV cache只属于Q1，Q3 LoRA只属于Q3，artifact model scope在接口处取与真实实现
scope的交集。

精确清单由33收紧为37：35项最终可由现有40证据覆盖，Q0/Q1 loss两项保持零证据；direct/cross evidence disposition
由32+8修正为31+9=40。最低causal-role debt为14（7 inference + 7 training），注册claim ceilings为none=2、D=12、
S=19、N=1、G=3；当前完成artifact ceilings为none=12、M=1、D=19、S=5、N/G=0。config-backed为31/37，
completion requirement与最终Markdown同步更新。相关回归`95 passed`，`git diff --check`通过。

### GPU0 breadth handoff no longer waits for unrelated shard1 evaluation

调度审计发现Q2主队列在GPU0完成selected-branch shard0后，只等待GPU2 shard1、CPU merge和shared evaluation；
旧lane2 watcher却以整个主队列PID退出及selected eval为交接条件，会让GPU0在没有scorer ownership时空等。lane2的
Q2 block-20 attention/head/group/MLP/RoPE、Q0 breadth和Q2 optimizer replay均为预注册、固定run ID，不依赖selected
block或任何effect，因此等待evaluation没有科学必要。

现handoff只要求immutable Q2 shard0 metadata=`completed`；eligible=false时contract本身即释放GPU0。旧进程树精确
确认为两层bash加`sleep 30`、无Python/scorer后终止，并以新watcher PID 3652934重挂；当前shard0运行不受影响。
静态queue topology 21项测试、producer topology 19/19 formal+21/21 supplements、实时4-GPU ownership均通过。
按当前机械吞吐，shard0约比等待shard1+merge/eval早释放约2小时；这只是利用率优化，不改变实验、family、顺序内
选择、seed、层、效应或证据资格。全部改动后全仓`912 passed, 7 subtests passed`，`git diff --check`通过。

### Q3 LoRA query routing and value transport are separate training interfaces

Q3冻结adapter同时作用于`q_proj`与`v_proj`，但二者分别改变attention query/key routing和value transport；原来的
`lora_adapter_path`把两个不同算子合并，可能让共享训练动力学描述被误读成对同一个operator的诊断。现拆为
`lora_q_projection_adapter_path`和`lora_v_projection_adapter_path`，两者都严格保持Q3-only scope，并继续只绑定
已注册的LoRA path、optimizer replay与head geometry artifacts。

两个接口分别保留独立的最小未来证伪门：固定另一条adapter路径、数据、初始化、优化器与update budget，仅隔离
q-path或v-path，并要求parameter/effective-update匹配、多seed和冻结utility gates。该债务登记不授权当前新增
experiment family，也不从artifact完成推断科学支持。

精确实现清单因此由37收紧为38，config-backed为32/38；最低结构化causal-role debt为15（7 inference + 8
training），注册artifact claim ceilings为none=2、D=13、S=19、N=1、G=3。全部40项证据的31 direct + 9 cross
处置不变。真实architecture audit为0 failures，completion仍为4/12；相关定向回归`95 passed`，`git diff --check`
通过。随后全仓912项测试完整通过，`git diff --check`再次通过。

### Four frozen native-score computations no longer share one readout claim

代码级复核发现原`native_lm_head_score`仍把四种不同的candidate scalar混在一个实现接口：Q0的final-token raw
Yes−No logit、Q1经共享prompt KV cache计算的candidate-response mean token log-likelihood、Q2的final-token raw
Yes−No logit，以及Q3在三个causal states上的四个teacher-forced log-prob terms组成的两路径均值差。尤其Q2
native-readout已经完成时，聚合行会把artifact-level S上限显示在整个混合接口，掩盖Q0/Q1/Q3仍pending的事实。

现将其拆成四个model-scoped exact interfaces，并保留`tied_lm_head_rows`作为物理共享权重用途；共享参数不再等同于
共享score algebra。拆分没有新增实验、读取effect或改变40项证据，只让每个readout completion严格落在其真实模型
scope。新增synthetic partial-completion回归专门验证：仅完成Q2 readout时，Q0/Q1/Q3不能借得causal completion。

精确清单由38收紧为41，config-backed为35/41；注册claim ceilings为none=2、D=13、S=22、N=1、G=3，
causal-role registered从23变为26，最低15项causal debt不变。真实当前artifact ceilings为none=14、M=1、D=21、
S=5、N/G=0：Q2 score path已有S级artifact可用，Q3只有D级artifact，Q0/Q1仍无完成的score-path artifact。
这仍只是artifact availability，不是科学效应结论。architecture audit为0 failures，completion保持4/12。

### Scaled QK-logit formation is separate from softmax edge use

attention复核发现`attention_logits_and_softmax_edges`仍把两个不同阶段合并：Q/K点积与`1/sqrt(d)`缩放形成
pre-mask logits，随后才应用causal mask与softmax得到edge weights。现有D3 logit-mask/value-zero可以支持特定
edge route的使用，却不能把因果性归给QK dot-product/scaling算子；描述性attention mass同样不能完成该归因。

现拆为`attention_scaled_qk_logits`与`attention_softmax_edge_weights`。前者只绑定Q/K stage geometry并新增
lower-bound operator debt；最小未来证伪门要求在Q、K、V、mask、softmax、dropout、o_proj合同不变时干预完整
pre-mask scaled-logit tensor，并通过recomposition identity、same/wrong-history、scale/sign/random、reverse removal
和Q2/Q3功能复现。既有edge mask明确不算作该operator test。后者继续绑定已注册edge intervention，但claim
boundary禁止把edge依赖升级为softmax非线性本身的独占原因。

精确清单由41收紧为42，config-backed为36/42；最低causal debt为16（8 inference + 8 training），注册
claim ceilings为none=2、D=14、S=22、N=1、G=3。真实当前artifact ceilings为none=14、M=1、D=22、S=5、
N/G=0，routing层为14接口、3项lower-bound debt。相关定向回归`96 passed`，真实completion仍为4/12，
GPU/qrels/source-test边界不变。随后全仓913项测试完整通过，`git diff --check`通过。

### LoRA A, B, and gauge-invariant effective deltas are separate interfaces

Q3训练路径复核进一步发现，即使q/v已分开，原两条adapter path仍把A down-projection、B up-projection与
`(alpha/r)·B@A`有效函数增量合在一起。冻结D7实际对28层×q/v的A/B分别记录梯度与step-501 replay，并对
B@A做SVD、orthogonal-gauge identity、A-only/B-only/joint及二阶interaction审计；聚合接口不足以表达这些不同
可识别性边界。

现将q/v各拆为A factor、B factor与effective delta三个接口，共6项。A/B坐标本身受LoRA gauge变换影响，任何单个
rank coordinate、方向或factor norm均不可进入架构排序；只有B@A是gauge-invariant函数对象，但其几何和单步更新仍
只是D级训练诊断，不证明该parameterization导致transfer失败或改善utility。每个接口都有独立最小未来证伪门：
A/B需orthogonal-gauge等价、另一factor固定、effective-update norm匹配和多seed；B@A需与parameter-budget及
function-update-norm匹配的full-rank/alternative-low-rank control比较，并保持另一q/v路径固定。

精确清单由42收紧为46，config-backed为40/46；最低causal debt为20（8 inference + 12 training），注册
claim ceilings为none=2、D=18、S=22、N=1、G=3。真实当前artifact ceilings为none=14、M=1、D=26、S=5、
N/G=0；training层现有12个精确接口且12项都没有registered causal-role evidence。该拆分不新增当前实验或读取
effect，相关定向回归`96 passed`，completion保持4/12。
随后全仓913项测试完整通过，`git diff --check`通过。

### Q0/Q1 final RMSNorm evidence now lands on the exact readout nodes

跨模型scope复核发现，`d6_q0_q1_readouts`正式实验同样干预`final_rmsnorm_input`与
`final_rmsnorm_output`，但这两个精确接口此前只绑定Q2/Q3 native-readout evidence；Q0/Q1的最终norm/readout
覆盖会因此在接口表中消失。现把同一formal deliverable以其冻结Q0/Q1 model scope绑定到两个节点，Q2/Q3
绑定保持不变。

在全部formal完成口径下，这两个节点的`model_scope_registered`现精确覆盖Q0--Q3；不会用Q2完成状态冒充Q0/Q1，
因为每项evidence仍分别保存source与implementation交集scope。接口、claim ceiling与debt数量不变，相关定向回归
`94 passed`，`git diff --check`通过。随后全仓913项测试完整通过。

### Optimizer replay stages no longer share one effective-update claim

`optimizer_effective_update`原来同时表示microbatch accumulation/global clipping、Adam一二阶moment预条件、
decoupled weight decay及scheduler-scaled实际delta。冻结D7 replay明确逐段恢复和记录这些量，因此一个聚合接口会
让已有的final parameter/update geometry错误地替代尚未完成的clipping、moments和decay replay证据。

现拆为`gradient_accumulation_and_global_clip`、`adam_moment_preconditioned_direction`、
`decoupled_weight_decay_term`与`learning_rate_scaled_effective_parameter_delta`。前三项只绑定formal optimizer replay；
只有最终delta可继续引用Q2 parameter-update geometry与anisotropy描述证据。四者各自保留独立多seed最小证伪门，
要求固定其余gradient/optimizer/schedule/data/init/budget并匹配effective或integrated update norm，最终仍需冻结surface
与ranking-utility gate，不能从一步replay推断方法收益。

精确清单由46收紧为49，config-backed为43/49；最低causal debt为23（8 inference + 15 training），注册
claim ceilings为none=2、D=21、S=22、N=1、G=3。真实当前artifact ceilings为none=17、M=1、D=26、S=5、
N/G=0；training层15接口中，clip/moments/decay当前正确保持none，而不是借final geometry得到D。相关定向回归
`96 passed`，completion保持4/12，未读取effect/qrels/source test。随后全仓913项测试完整通过，
`git diff --check`通过。

### Frozen autoregressive visibility is not the diagnostic history-edge mask

基础mask审计发现原`causal_attention_mask`把冻结autoregressive visibility topology与D3诊断时额外施加的
history-edge logit mask混为一个接口。D3 edge mask/value-zero可以证明允许的history route是否被使用，但没有改变
或隔离模型原生causal topology，因而不能支持“自回归mask已被因果检验”或“它不是transfer瓶颈”。

接口现改为`autoregressive_causal_attention_mask`，只绑定attention-pattern描述证据；D3因果edge证据仍归入
`attention_softmax_edge_weights`。新增的最小未来证伪门要求仅改变query/history/candidate prefix内部visibility，
同时严格保留answer/continuation causal boundary、token/position、label isolation、identity/leakage audit、
same/wrong-history specificity、Q2/Q3复现、多seed与冻结utility。这暴露了一个真实但当前未授权实现的新方向：
输入前缀的可见性拓扑，而不是简单删除某条history edge。

接口总数仍为49、config-backed为43/49；最低causal debt从23增至24（9 inference + 15 training），注册claim
ceilings从D=21/S=22纠正为D=22/S=21。真实当前artifact ceilings不变为none=17、M=1、D=26、S=5、N/G=0；
相关定向回归`96 passed`，completion保持4/12且未读取effect/qrels/source test。
随后全仓913项测试完整通过，`git diff --check`通过。

### Functional causal-role availability no longer masquerades as operator attribution

精确接口表此前已经说明24项`operator_causal_debt`只是“连causal-role artifact都没有”的下界，但机器字段仍只在
全局给出`operator_attribution_inferred_for_other_interfaces=false`。这不足以逐接口阻止一种常见误读：例如在
`attention_o_projection`输出状态做S/N/G patch能识别一个功能state mediator，却不能由“产物已完成”自动归因给
`o_proj`矩阵乘法本身；同理，完整block state、RMSNorm边界或native readout path的artifact可用性也不等于算子
原因已经成立。

现为49个接口逐一增加`operator_attribution_status_from_artifact_availability`与固定false的归因标志，并增加三项
聚合不变量：由artifact availability推断的operator attribution必须为`0/49`，未由completion解决的接口必须为
`49/49`，25个已有functional causal-role的接口必须单列为“有功能干预、但没有由完成状态推断算子归因”。24项
lower-bound debt与其未来最小证伪门保持不变；这不是说最终科学解释永远不能识别RoPE/edge/readout等路径，而是
禁止readiness/completion机械标志替代逐项effect解释。最终Markdown也显式报告0/49与49/49。

该收紧不新增实验、不读取effect/qrels/source test，不改变49个接口、40项证据处置或24项lower-bound debt。
接口、readiness、completion与report builder相关定向回归`94 passed`；随后全仓`913 passed, 7 subtests passed`。

### All 49 exact interfaces now have implementation provenance

architecture audit此前准确报告43/49项`config_backed`接口，但没有机器化解释余下6项为何不由静态配置证明。
这六项不是遗漏，而是动态执行合同：`serialization_tokenization`、`kv_cache_phase_boundary`、三个residual
边界以及`candidate_readout_positions`。若只展示43/49，最终读者可能把“动态来源”误读成“未审计实现”。

现新增冻结的dynamic-interface contract：序列化绑定project-owned baseline/prompt source；Q1 cache phase绑定
原生scorer与Q1 trajectory source；candidate readout绑定Q0/Q1及共享prompt-position source；三个residual边界
除`transformer_instrumentation.py` SHA外，还要求Q0--Q3四个runtime identity smoke在block 13对应节点全部精确
score identity。architecture audit逐项输出source SHA、binding kind、runtime node/model scope与failure列表。

最终implementation provenance现在是`43 config + 6 dynamic = 49/49`，无重叠、无缺口；completion将49/49和
六项零failure设为硬门，Markdown附录会逐项列出动态接口源码字节。该溯源仍只是实现/机械证据，不推断科学
support。真实审计为0 failures，相关architecture、interface、readiness、completion与report回归`96 passed`。
随后全仓`913 passed, 7 subtests passed`，`git diff --check`通过。

### The 18-by-4 matrix now exposes all 72 coverage dispositions

逐模型coverage复核确认，最终18组件×4模型矩阵不能只检查“有72个键”，还要机器化区分每个cell是否真正注册、
是否只有描述证据、以及是否具备当前阶段的causal-support路径。冻结注册的实际结构是：Q0 `10/18`、Q1
`10/18`、Q2 `17/18`、Q3 `18/18`，合计55/72个directly registered cells；其中31个具有causal-support-
capable deliverable，24个只有描述性注册，另17个明确`not_directly_registered`。这些空白不能用Q2/Q3结果
或共享base model自动填充。

现将上述`31 + 24 + 17 = 72`写入outcome-independent coverage debt，并要求所有cell都有且只有一个处置；
formal report会直接显示总数。最终completion还会对comprehensive JSON硬验18个component key及每个component
恰好4个model key，防止human worksheet漏行或跨模型借证据。该改动不读取effect/qrels/source test，也不把coverage
称为support；overview、formal report、completion与comprehensive report相关回归`108 passed`。
随后全仓`913 passed, 7 subtests passed`，`git diff --check`通过。

### Residual addition and RMSNorm operators are no longer hidden behind state boundaries

继续按“状态证据不能冒充算子归因”复核时发现，49接口清单虽然分别列出pre/post residual与RMSNorm output，
但仍未把两次elementwise residual addition以及三类RMSNorm变换本身列成独立implementation interfaces。
这与冻结V2计划明确写出的边界一致：absolute state patch不能证明residual addition必要，RMSNorm output也不是
operator bypass。只保留输出状态会让“post-block才变化”被误读为已识别addition或normalization算子。

现新增五项精确接口：`attention_residual_addition`、`mlp_residual_addition`、input/post-attention/final三处
`rmsnorm_variance_rescale_and_gain`。现有D0四模型smoke已分别验证`r+a`与`u+m`的BF16-bounded algebra
recomposition，final RMSNorm公式也通过机械重组；D2 norm flow和D4 geometry提供描述证据。但这些都不隔离算子
原因，因此五项全部进入lower-bound inference-operator debt，不能借相邻state的S/N/G completion。

每项新增独立最小未来证伪门：两次residual需在固定两输入时只改变composition rule/coefficient，并通过alpha=1
identity、norm/direction/random、same/wrong-user、reverse removal、Q2/Q3与utility门；三处RMSNorm需分开variance
rescale与learned gain，固定上下游、使用output-norm-matched controls并复现。债务账本不授权当前新增实验。

精确接口由49增至54；实现溯源为`46 config + 8 dynamic = 54/54`；lower-bound debt由24增至29
（14 inference operator + 15 training mechanism）；注册artifact ceiling为none=2、D=27、S=21、N=1、G=3。
25个已有functional causal-role的接口不变，artifact availability推断的operator attribution仍为0/54。
相关inventory、architecture、readiness、completion与双报告回归`118 passed`；随后全仓
`913 passed, 7 subtests passed`，`git diff --check`通过。

### Q/K head RMSNorm operators are distinct from their pre/post states and RoPE

attention内部继续下钻后确认，Qwen3在`q_proj/k_proj`与RoPE之间分别应用head-dimensional `q_norm/k_norm`。
原清单已有pre-norm与post-norm states，也有RoPE后的Q/K，却没有把两个RMSNorm operators本身列出；这会让
D5的post-norm RoPE phase intervention或D3 pre/post geometry被误读为已经检验q_norm/k_norm。实际冻结forward
明确是`q_norm(q_proj(h))`与`k_norm(k_proj(h))`，两者各有独立learned gain。

现新增`q_head_rmsnorm_variance_rescale_and_gain`与`k_head_rmsnorm_variance_rescale_and_gain`，绑定project-owned
instrumentation源码SHA及Q0--Q3四模型对应post-norm runtime identity节点。现有Q/K stage geometry只给D级描述；
两个算子都进入lower-bound debt。最小未来门要求固定projection output、另一Q/K/V路径、RoPE phase、mask、
softmax与o_proj，只分别干预variance rescale和gain，并通过identity、magnitude/direction、same/wrong-user、reverse
removal和Q2/Q3复现；RoPE干预不能借作Q/K norm operator evidence。

精确接口由54增至56；implementation provenance为`46 config + 10 dynamic = 56/56`；lower-bound debt为31
（16 inference + 15 training）；注册ceiling为none=2、D=29、S=21、N=1、G=3。25个functional causal-role
接口不变，artifact availability推断operator attribution仍为0/56。相关定向回归`96 passed`。
随后全仓`913 passed, 7 subtests passed`，`git diff --check`通过。

### The frozen forward graph is now audited as an executable primitive census

在56个精确接口都具备实现溯源后，继续复核发现“接口全集”仍不能机器证明实际forward没有漏掉一整段执行逻辑：
Qwen3 decoder之外还存在project-owned serialization/candidate position、position ID与RoPE basis、causal mask
construction、SDPA内部GQA/softmax/value aggregation、Q1 cache update、tied lm-head以及四种native score algebra。
若没有按真实执行顺序串联这些步骤，人工接口清单即使数量完整，也可能在模型库升级或backend切换后失真。

现新增38项冻结semantic primitive census，从输入序列化一直覆盖到Q0--Q3各自native score；它们精确映射全部
41个非训练接口，结果为`41/41 mapped, 0 missing, 0 extraneous`。15个training接口明确排除在inference forward
census之外，继续由loss、gradient、AdamW与LoRA独立审计。审计同时对安装的Transformers 5.12.1绑定8个实际
source objects及其file/object SHA和执行sentinel，覆盖Qwen3Model、DecoderLayer、Attention、MLP、RMSNorm、
RotaryEmbedding、CausalLM与SDPA；删除任一独占primitive（回归用SiLU）会fail closed。

另显式验证9条冻结不活跃路径：attention dropout=0、sliding window及其dispatch关闭、Q/K/V/O bias关闭、
untied lm-head关闭、dynamic/non-default RoPE scaling关闭、MLP三投影bias硬编码关闭、native SDPA不物化
attention weights、四模型alternative backend均非冻结score路径。它们只是排除当前实现分支，不是机制反证。
机器字段固定声明该清单是semantic primitive
census而非kernel-instruction census，且不能从coverage推断operator attribution。

architecture与completion真实审计均为0 failures，exact-interface requirement继续completed；相关
architecture、readiness、completion与comprehensive report定向回归`92 passed`。本项不读取scientific effect、
qrels或source test，也不授权新的实验family。随后全仓`914 passed, 7 subtests passed`。

### The frozen training update is now audited from objective to effective LoRA function delta

与38项inference forward primitive对称复核训练链路时，发现原15个training接口漏掉一个真实活跃分支：Q3
冻结配置与checkpoint adapter config均声明PEFT 0.19.1、`lora_dropout=0.05`，训练态执行路径是在q/v两类
adapter的A down-projection前对输入做dropout，evaluation态为identity。原有A/B因子、AdamW replay与
gauge-invariant `B@A`几何会经过或汇总这条随机路径，但都没有把dropout本身隔离为算子归因。

现新增`lora_training_input_dropout`精确接口，并把它登记为第16项training-mechanism lower-bound debt。现有
D7训练动态产物只给D级availability ceiling；当前没有dropout-only causal或utility结论。最小未来证伪门要求
固定data、initialization、q/v targets、rank、loss、AdamW、schedule、update budget与RNG protocol，只改变
adapter dropout，在可行时匹配integrated gauge-invariant function-update norm，并使用多seed、冻结surface、
null recovery与ranking utility门。债务账本仍不授权当前新增训练family。

训练执行现由22项semantic primitives机器化覆盖：Q3 dropout与q/v A/B前向、Q0 BCE、Q1/Q3 sequence NLL、
Q2 RankNet/ListNet及0.5/0.5组合、loss scaling/backward、microbatch accumulation、unscale/global clip、Adam
一二阶moment/bias correction、AdamW decoupled decay、linear warmup/decay、effective parameter delta，以及
q/v两类gauge-invariant function delta。结果为`16/16 mapped, 0 missing, 0 extraneous`；删除dropout primitive
会fail closed。另绑定11个project/Torch/Transformers源码对象的file/object SHA与执行sentinel，以及1个冻结
PEFT adapter artifact；实际审计0 failures。

精确接口现为57，implementation provenance为`47 config + 10 dynamic = 57/57`；lower-bound debt为32
（16 inference + 16 training），注册ceiling为none=2、D=30、S=21、N=1、G=3，artifact availability推断
operator attribution仍为0/57。机器边界明确该覆盖只是single-step semantic primitive census，不是multiseed
causal attribution，也不读取scientific effect、qrels或source test。相关定向回归`98 passed`。

### Q3's active PEFT forward branches are no longer hidden inside q/v output states

在57接口通过后反向对照冻结PEFT 0.19.1源码，确认Q3 evaluation forward并非普通`q_proj/v_proj`：未merge且
adapter active时，q/v两路都执行`base_layer(x) + B(A(dropout(x))) * scaling`；evaluation的dropout为identity，
`scaling=alpha/r=2`，结果再转回base projection dtype。原清单只有注入后的`q_pre_norm`/`v_projection`状态以及
training侧A/B/`B@A`更新几何，因而漏掉了两个真实活跃的inference operators。

现新增`q3_q_lora_scaled_adapter_injection`与`q3_v_lora_scaled_adapter_injection`，forward primitive census分别在
q_norm前与value transport前显式列出identity-dropout→A→B→scale→base-add。两者都有D7 implementation/
training-dynamics与geometry产物，但没有branch-specific bypass、reverse necessity或utility，因此各自进入
inference-operator lower-bound debt；A/B norms与更新几何不能借作其因果证据。

同时纠正source provenance环境：此前审计使用当前shell的Transformers 5.13.0源码，而冻结Q0--Q3训练/评分
metadata实际绑定`/home/gkl/miniconda3/envs/pps-kuaisearch`中的Torch 2.6.0+cu124、Transformers 5.12.1和
PEFT 0.19.1。现从checkpoint记录的Python环境直接解析并SHA绑定Qwen3、SDPA、PEFT `Linear.forward`与
`LoraLayer.update_layer`，再绑定project-owned Q3 loader；loader source还fail-closed排除`merge_and_unload`/
`disable_adapter`，并确认`PeftModel.from_pretrained(..., is_trainable=training)`及`model.train(training)`。
forward现为40 primitives、`43/43` inference interfaces、11 source bindings，checkpoint source environment=
true；删除q-adapter primitive会明确报告该接口missing。

### BF16 autocast and non-reentrant checkpoint recomputation are explicit training mechanisms

继续对照四个冻结training config与shared trainer时发现，`dtype=bfloat16`与
`gradient_checkpointing_enable(use_reentrant=False)`此前只存在于配置/loader，没有进入training exact-interface
与causal-debt矩阵。BF16决定forward/loss/gradient数值路径；non-reentrant checkpointing在backward重算decoder
activation，并需明确审计Q3 LoRA dropout的RNG/mask保持。它们都不能被gradient clipping或AdamW replay自动覆盖。

现新增`bfloat16_autocast_training_forward`与`nonreentrant_gradient_checkpoint_recomputation`。前者已有actual
optimizer replay的D级availability ceiling，但没有FP32 precision-matched control；后者当前无registered evidence。
对应最小未来门分别要求固定data/order/init/dropout/optimizer后的BF16-vs-FP32 raw-gradient/effective-delta等价审计，
以及checkpoint on/off逐参数gradient/update与Q3 dropout-mask/RNG等价审计；机械等价未通过前不得形成utility claim。

training census现为24 primitives、`18/18` training interfaces、12 source bindings与2 artifact bindings；第二个
artifact binding证明当前shared trainer/contracts SHA与冻结训练metadata identity完全一致。全接口现为61，
implementation provenance=`51 config + 10 dynamic = 61/61`；lower-bound debt为36（18 inference + 18 training），
注册ceiling为none=3、D=33、S=21、N=1、G=3，availability推断的operator attribution仍为0/61。相关定向回归
`99 passed`，随后全仓`916 passed, 7 subtests passed`。

### Frozen-disabled training branches and Q3's hidden gradient/dtype bridges are explicit

训练图不能只列active primitives，否则最终设计排序仍可能把冻结配方实际关闭的分支当作候选机制。现新增7项
inactive-training audit并全部机器验证：FP16 autocast/dynamic GradScaler关闭（四模型均BF16）、history dropout
augmentation关闭（概率0）、legacy reentrant checkpoint engine关闭（统一`use_reentrant=False`）、Q3 full-parameter
optimization关闭、LoRA bias关闭、DoRA/RSLoRA/QALoRA/BDLoRA/aLoRA变体关闭，以及Q3 adapter merge/disable路径
关闭。每行都输出config/source/artifact observed值、expected值、failure与scientific boundary；completion与最终
Markdown把`7/7`列为硬门。这些排除只限定冻结实现，不是机制反证或设计优先级。

同一loader复核还发现Q3专属`enable_input_require_grads()`。Q3 base embedding冻结且non-reentrant checkpointing
开启，这个hook使embedding outputs保持可微，令checkpointed q/v LoRA模块接收backward signal；Q0--Q2全参数
路径不依赖同一桥。现新增`q3_input_activation_requires_grad_bridge`与对应primitive/causal debt。现有D7只证明下游
gradient/update存在，不能隔离bridge；未来最小门需在固定microbatch、dropout mask、checkpoint、autocast、loss和
optimizer下，与checkpoint-safe reference逐参数比较q/v LoRA gradient和effective delta。

进一步读取safetensors header（不加载tensor value）确认一个不同于普通BF16 autocast的Q3数值边界：Q0训练
checkpoint为595,776,512个FP32参数，Q1/Q2各596,049,920个FP32参数；Q3 adapter的1,146,880个A/B参数为FP32，
而Q3 frozen base artifact的751,632,384个serialized elements均为BF16。PEFT forward源码相应先把input cast到
LoRA A dtype，再把scaled adapter result cast回base-result dtype。现新增
`q3_fp32_lora_bf16_base_cast_boundary`，并以两个primitive分别登记BF16→FP32与FP32→BF16 cast。独立artifact
binding记录五套checkpoint dtype/shape census及digest；测试把Q3 adapter伪改为BF16时会fail closed。

全接口现为63，implementation provenance=`53 config/artifact + 10 dynamic = 63/63`；forward仍为40 primitives、
`43/43` inference interfaces，training扩为27 primitives、`20/20` training interfaces、12 source与3 artifact
bindings。lower-bound debt为38（18 inference + 20 training），注册ceiling为none=3、D=35、S=21、N=1、G=3，
availability推断operator attribution仍为0/63。定向回归`99 passed`；随后全仓
`917 passed, 7 subtests passed`，`git diff --check`通过。

### Terminal report completion now revalidates the current evidence chain

最终completion audit原先对四个human-closeout文件只检查存在性、顶层`completed/final`、13节计数与18×4
矩阵键覆盖。该检查可以阻止缺文件，却不能阻止陈旧report、手改JSON或“空矩阵+completed标志”在当前证据
字节已经变化后继续通过。这个缺口不影响正在运行的科学任务，但会削弱最终机制结论与优化方向排序的可复核性。

现把所有claim invariants提取为builder与completion共享的单一常量；当worksheet、formal JSON、comprehensive
JSON和Markdown四者齐备时，completion会重新执行当前terminal readiness、19项formal与21项supplement覆盖、
design-gate/necessity model scope、worksheet语义约束、formal→comprehensive outcome边界、40项证据的逐项
disposition与evidence identity绑定。它还重新构造readiness admission、component bidirectional gate matrix、
18×4 coverage、63接口coverage、reproducibility ledger、13节contract和claim invariants，并逐字段与综合JSON
比较；Markdown必须与当前JSON重新render的字节完全相同。

formal admission中的冻结assets、19项deliverables、每个run metadata和dev-eval ledger，以及design supplement
均要求repository-relative path、不能逃逸root、目标存在、64位SHA合法且当前字节SHA完全一致。审计只重验
已注册gate/claim结构与byte identities，不重算scientific effect，也不打开source test或新增实验family。
专门红队测试确认：证据文件单字节变化、`../`root escape，以及四个“看似completed”的伪终态文件都会
fail closed。相关completion/comprehensive回归`88 passed`；随后全仓`919 passed, 7 subtests passed`，
`git diff --check`通过。

### Completed update geometry constrains, but does not choose, the causal attention hypothesis

在等待D2与attention/MLP正式任务期间，只读取已终态、qrels-blind且预注册为descriptive的D7产物，对训练
容量分布作跨适配边界复核。Q2全参数base→final的全局relative update Frobenius为0.00245；28层
per-layer update RMS的变异系数为0.0735、最大/最小为1.33，四个七层区域的transformer update-energy share
依次为0.292、0.243、0.208、0.257。它反对“训练更新几乎只集中在某一个晚层”的简单叙事；update RMS与
candidate/common flow描述量的Pearson仅0.25--0.42，也不能用参数移动大小替代功能归因。

Q3使用gauge-invariant `DeltaW=(alpha/r)BA=2BA`按真实Q/V输出head分组。final checkpoint的Q-head
normalized participation ratio从前七层均值0.892降至末七层0.750；同一几何下Q2全参数Q update对应为
0.956与0.910，Q3-minus-Q2在末段达到-0.160。相反，Q3 V-head participation在四区域约
0.980/0.976/0.982/0.956，与Q2约0.975/0.969/0.956/0.958接近，未显示同等程度的late-head concentration。
step500、final及final-minus-step500均保留“Q比V在末段更集中”的方向，但block-flow相关仅为中等或更弱。

最窄解释是：Q3 LoRA的晚层query-projection函数更新比value-projection更新分布更窄，这是一个训练容量
约束，不是head utilization、capacity collapse或transfer harm的因果证据。该模式只能与冻结D3 attention
edge/group及D2 selected-branch结果做effect-blind对照；不允许据此追加head、改变层集合或提升为adapter
设计。只有后续Q-routing/attention-output在same-request、wrong-user、cross stress及双向干预上共同通过，
它才可成为“query routing adapter capacity”方向的佐证；否则保留为描述性异质性。

### The Q2 rank losses agree on candidate-relative direction; their shared nullspace is not ranking harm

继续只读复核两个已终态、qrels-blind的D7描述性产物。对recurrence、strict-transfer与other-overlap各96个
request，以及对应的within-request label-shuffle控制，RankNet与ListNet在score空间的梯度余弦均值为
0.959--0.981、中位数为0.993--0.994。映射到九个参数族后，base/final、三surface与两control共12个cell的
平均梯度余弦为0.892--0.986；任何参数族的RankNet-minus-ListNet平均squared-gradient share绝对差均未达到
0.05，观察到的最大值仅0.00303。两种loss的主要能量分配也一致，最大族通常是`mlp_down`，随后为
`attention_v`、`mlp_up`及attention q/k之一。因此现有证据不支持“Q2的pairwise与listwise目标把历史排序
信号沿不同模块方向互相抵消”作为主要失败解释。该结论只排除粗粒度全局/参数族冲突，不能排除单请求、
单层或族内signed冲突，也不等价于optimizer effective update。

同一审计还验证了一个精确代数边界：对RankNet、ListNet及0.5/0.5组合，在每个request的全部candidate score
上同时加137，loss变化最大仅约`2.02e-14`，gradient sum、Hessian-times-ones及common-direction curvature均为
数值零量级，且每个cell恰有一个common-shift null eigenvalue。这说明训练目标确实不约束per-request共同分数
偏移；但共同偏移本身不改变request内排序，所以不能直接造成transfer ranking harm，也不能据此提出“减去共同
offset”的方法。真正可能有害的仍须是candidate-relative margin方向在attention、MLP、residual composition或
readout中的衰减/错配；D2双向分支与组件necessity结果才负责定位该方向。

训练前后参数族share确有小幅重分配：观察surface上`mlp_down`约增加0.018--0.034、`mlp_up`约减少
0.019--0.027，embedding/readout约增加0.009--0.023；但label-shuffle控制出现相近变化，故当前只能解释为
checkpoint-state相关的梯度几何，不是历史语义选择性。这个结果把设计空间从“更换RankNet/ListNet混合权重”
收窄到候选相对信息的路由、形成、残差保存与readout对齐，同时保持该优先级为待因果门确认的工作假设。

### The frozen observation is now a byte-bound premise, not hard-coded report prose

终态综合报告此前会固定陈述“Q0--Q3在单seed回溯式KuaiSearch确认中呈recurrence-dominant history use，
strict-transfer gain未建立”，但报告证据链只直接绑定deep-dive plan/manifest、19项formal与21项supplement。
deep-dive manifest内部虽然记录首轮证据SHA，completion只验证manifest自身字节，未递归验证这些被引用文件；
因此首轮summary/results/diagnosis若后来漂移，硬编码的frozen-observation prose仍可能继续通过。

现新增7项显式`frozen_observation_source` identity：首轮`protocol.yaml`、机器summary、冻结结果登记、首轮机制
plan/probe manifest及首轮诊断JSON/Markdown。综合JSON单列`frozen_observation_evidence`，13节contract要求该
字段非空，Markdown逐项显示path与SHA，reproducibility ledger也把它们纳入frozen assets。builder与terminal
chain都会验证repository-relative path、root边界、文件存在性和当前SHA；任何来源字节漂移都fail closed。
这不新增实验、重算指标或改变冻结观察，只保证最终机制解释确实建立在当前source-of-truth字节上。

真实仓库已验证全面报告计划加7项来源共8个静态identity；伪造SHA的红队测试按预期失败。report/completion
定向回归`89 passed`，随后全仓`920 passed, 7 subtests passed`，`git diff --check`通过。同期结果盲进度为
D2固定56/60、请求加权93.68%，Q2 selected branch 85.66%；四张物理GPU各有且仅有一个writer，ownership
审计0 failures。

### The frozen recurrence-transfer premise is reconstructed from evaluator output

仅绑定首轮summary字节仍不能保证最终报告的定性文字与机器值一致。现新增
`frozen_observation_machine_snapshot`：builder只读取已绑定的首轮机器summary，并逐模型复制Q0--Q3的full
NDCG、full-minus-null/full-minus-wrong-user在overall/recurrence/strict-transfer/other-overlap上的mean与
normalized-query cluster CI，以及recurrence/strict/other-overlap人口贡献。每个模型的原始shared-evaluator
evidence JSON也按summary声明的path/SHA验证；不读取其内容、qrels或score bundle。

snapshot机械验证首轮population恰由`288+1079+290+266+2077=4000`重构，pilot seed为20260714、没有第二
训练seed、bootstrap为normalized-query/5000 draws/seed 20260715、source test关闭。更关键的是，冻结定性
前提被展开为8个必须全真的布尔合同：四模型的full-minus-null与full-minus-wrong-user overall、recurrence
CI均为正；两种strict-transfer CI均跨零；full-minus-null other-overlap CI均为负；每个模型recurrence人口
贡献均大于strict-transfer。任何值或区间改变使这些合同不再成立时，综合报告不能继续沿用旧文字。

全面报告第2节现在渲染四模型的精确机器表、population counts、seed/bootstrap、evaluator evidence bytes和
single-seed/retrospective边界；13节contract要求snapshot存在，terminal audit会从当前绑定summary重新构造并
与最终JSON逐字段比较。这不是重算paper metric，而是防止人手抄写与定性前提漂移。定向回归`90 passed`，
随后全仓`921 passed, 7 subtests passed`，`git diff --check`通过。

### The first-round and M0--M3 premises remain first-class evidence in the final report

首轮机器snapshot进一步递归验证每个Q0--Q3 evaluator evidence内部声明的4项shared-evaluator artifact：metadata、
metrics、per-request输出与target-aware surface manifest；summary中的`metrics_sha256`必须与真实metrics identity
精确相等。另验证pre-qrels score-bundle audit为passed且未读qrels、qrels hash lock在shared evaluator前已验证，
以及四模型共同的4k confirmation records path/SHA与split/request count。shared evaluator本身按协议读取qrels，
但本snapshot只做字节验证，不打开qrels或score bundle。

同时发现仅保留deep-dive的19+21证据会让最终H0--H5丢失首轮M0--M3起点。现新增
`prior_mechanism_diagnosis_snapshot`，从已绑定的首轮诊断JSON保留H0/H5 unresolved、H1--H4 weakened的原始
状态、每项component status/rationale/support/opposition/triangulation/remaining uncertainty，以及全部10条
contradiction。首轮artifact registry的18项M0 data/power、M1 behavior/token、M2 representation/patch和M3
gradient/matched-control产物也逐项验证当前path/SHA，并保留stage/kind/run identity；不重算scientific effect。

最终第8节会先渲染prior M0--M3 hypothesis/contradiction ledger，再渲染deep-dive更新后的H0--H5，防止新内部
组件结果静默覆盖数据功效、输入、跨模型异质性或训练控制的旧反证。13节contract和terminal chain均要求该
snapshot存在并从当前冻结字节重新构造。定向回归`91 passed`，随后全仓
`922 passed, 7 subtests passed`，`git diff --check`通过。

H0--H5使用的11个逻辑证据ID（`E_M0_*`到`E_M3_*`）与18个文件artifact ID并非同一命名空间；只保留两端
仍无法从假设追到文件。snapshot现额外验证完整`evidence_index`：每个逻辑证据必须`valid_result=true`、具备
stage/summary/scope，引用的artifact ID必须存在，11行合计必须恰好覆盖全部18个artifact；hypothesis的support/
opposition与10条contradiction引用也必须属于这11项。最终Markdown逐行渲染logical-evidence→artifact映射以及
18项path/SHA/run/stage/kind，故H0数据功效、H1输入路由、H2/H3表示与patch、H4梯度/匹配训练控制均能从结论
追溯到具体字节。补充映射后定向回归仍为`91 passed`，全仓仍为`922 passed, 7 subtests passed`。

### History-span evidence is now separated from endpoint-vector evidence

进一步审计尚未闭合的D4 SwiGLU formation时确认：系统并非所有family都以同一粒度“观察历史序列”。
D3 attention observation/edge family在`history_summary`与native readout query rows上，显式解析完整
query/history/candidate token span，并按head/GQA group汇总mass、value contribution与固定edge干预；D5
context/position family也对完整保留history-content span作position-preserving content、visibility或RoPE
phase干预。相反，D1全层trajectory主要读取`query_end`、`history_summary_end`与每个candidate native
readout，D4 feature formation在固定blocks 13/20/27读取同样语义endpoint上的gate-pre、SiLU gate、up与
product。后两者能定位carrier或feature formation，却不是每个历史event/token的逐层MLP轨迹。

为防止最终报告把endpoint vector扫描冒充event-level sequence attribution，新增8行
`history_signal_observation_scope_contract`，逐行固定evidence IDs、model scope、实际token/span、粒度、
能回答的问题与`not_observed`：完整history content span、attention span edges、全层endpoint trajectory、
native-readout component mediation、SwiGLU endpoint formation、position/RoPE span controls、四模型native
score readout及Q0/Q1 model-specific sequence breadth。component 18×4章节现在必须渲染该表；其中明确写死
`history_summary_end`不是tokenwise event trace，SwiGLU extension不是per-history-event MLP trajectory，
attention mass不是preference content，readout mediation也不能反推上游event。

该增强只收紧最终解释边界，不修改冻结family、sample、block、position或GPU队列。terminal completion audit
会从代码常量重构整表并逐字段比较最终JSON，section contract同时要求字段非空；修改JSON/Markdown不能静默
通过。新增覆盖测试后report/completion定向回归为`92 passed`，全仓为
`923 passed, 7 subtests passed in 31.85s`，`git diff --check`通过。

### Prior design hypotheses cannot be silently rewritten by the final ranking

首轮M0--M3诊断已经给出五项architecture opportunity，但此前final snapshot只保留H0--H5、contradiction、
11项logical evidence与18项artifact bytes；正式deep-dive又从独立catalog重新表述同五个ID。两者概念一致，
但如果只显示后者，首轮的原始innovation target、architecture requirement、necessary modules、training
signals/data、key ablations、falsifiable predictions与prior-work差异仍可能在最终报告中被合理化改写或静默
遗漏。

`prior_mechanism_diagnosis_snapshot`现从已绑定的首轮诊断JSON完整读取五项
`architecture_opportunity_matrix`。每项必须属于冻结五ID、引用已验证的11项logical evidence与H0--H5，
priority只能是注册状态，implementation必须保持`not_started_not_authorized`且evaluation contract不变；
CoPPS/BATA/HMPPS/MemRerank四个comparator的shared ground、substantive difference与source ref必须完整。
所有模块、train-only data、训练信号、消融与预测逐项保留，不把后来的catalog wording倒灌回首轮记录。

最终第10节因此形成三段可审计链：首轮M0--M3机会假设→formal deep-dive五项排序→综合component-gated
处置。13节contract要求prior opportunity matrix非空，terminal chain从冻结首轮字节重构整个snapshot并
逐字段比较，所以后续结果只能改变处置与优先级，不能改写原始设计理由。这不实现架构或增加实验family。
report/completion定向回归保持`92 passed`，全仓为
`923 passed, 7 subtests passed in 32.77s`，`git diff --check`通过。

### Opportunity priority changes now have an exact longitudinal lineage

虽然prior、formal与comprehensive三阶段现在都保留五个冻结ID，读者仍需跨三张表手工拼接prior priority、
formal rank/status与最终component-gated target。现新增机器派生`opportunity_lineage_matrix`：每个首轮ID
必须在prior snapshot、formal ranking与`formal_opportunity_disposition`各出现且只出现一次；formal target可
映射到一个综合opportunity或not-recommended direction，但target必须真实存在，重复direction或丢失target均
fail closed。

每行同时显示prior bottleneck hypotheses/logical evidence、formal rank/status/evidence、最终disposition、
target kind/ID、functional component、design priority或not-recommended basis、actual evidence level、model
scope与supporting findings。utility gain和architecture implemented在整个lineage固定为false；该矩阵只解释
证据处置如何变化，不把优先级变化伪装成已训练方法收益。第10节先显示纵向总表，再完整显示首轮设计合同、
formal五项排序与综合机会细表。

terminal completion audit使用当前prior/formal/worksheet重建同一矩阵并要求顶层字段逐字相等，13节contract
也将其列为optimization section必需字段。新增missing-target红队测试后定向回归为`93 passed`，全仓为
`924 passed, 7 subtests passed in 32.50s`。

### Selected-branch synthesis is bound to evaluator tables and evidence bytes

D2七节点分解的两模型综合原先已经冻结planned family size和BH，但综合器对每模型evaluator的`results`、
`family_rows`以及上游score bundle/pre-qrels audit/per-request NPZ只做了有限结构检查。现将综合入口收紧为
fail-closed链：96项family row必须精确覆盖48 contrasts×2 endpoints，所有p/mean/CI有限且与`results`逐项
一致；fold/qrels/bootstrap/population/family policy必须符合注册合同；merged bundle metadata/scores、
pre-qrels audit和per-request contrasts必须与声明SHA对应。pre-qrels七项检查、selected block、method、
implementation digest和bundle identity也必须逐字段一致，qrels只验证evaluator已经记录的SHA格式，综合器
自身不打开qrels。

最终synthesis input identity会保留上述每个path/SHA和`evaluator_tables_cross_checked=true`，因此后续两模型
BH或报告不能在不被发现的情况下引用被篡改score、漂移表格或错误per-request产物。新增正常链、result/row
漂移与bundle字节篡改红队测试；selected-branch/report定向回归`141 passed`，全仓
`927 passed, 7 subtests passed in 32.41s`，`git diff --check`通过。

### Reverse-necessity design gates now rederive statistics and bind evidence bytes

V2 position-preserving component removal尚未产生结果，因而在其启动前继续红队审计design synthesis。原综合器
能验证32项family coverage、selected-parent lineage和布尔gate的部分必要条件，但没有重新计算BH，也没有把
`results`与`family_rows`的mean/CI/p/q逐项相等、pre-qrels audit、per-request NPZ及qrels identity声明全部
绑定。这会允许某些表格漂移在布尔gate未同步变化时进入后续报告。

现综合入口对每个endpoint的16-unit BH从原始p重新计算；completed/gate-stopped inference分别验证有限值或
固定p=q=1，nested result与family row必须逐项一致，positive removal、neutral-primary gate及NDCG等价标志
均从mean/CI/BH重新派生。pre-qrels八项检查、extension manifest、两模型input identities、per-request bytes
和三个qrels SHA声明也全部fail closed；综合器仍不打开qrels或score bundle，后续shared-parent audit才按已
声明identity验证necessity与sufficiency共享同一selected-branch字节。

新增伪造BH和篡改per-request bytes红队，并将旧null-sensitivity测试改为真正满足正向CI/BH但仍不能替代
position-preserving primary gate。定向回归`105 passed`，全仓
`929 passed, 7 subtests passed in 32.40s`，`git diff --check`通过。

### Q2 selected-branch merge closed and exposed a pre-qrels fallback-audit bug

Q2 block-15 selected-branch两shard完整合并为3,918个fold-1请求、78,864 candidate rows，14项identity
最大误差0、path-local BF16 ratio 0.0625、merged scores SHA
`702d58e2b917e3a5325c6957377c76843bccf439497bd71cfe410355c8de2f7e`。首次shared evaluator调用在
打开qrels之前正确停止：计划和scorer明确规定wrong-user不合格请求写回`frozen null score`，但evaluator
错误地把该值与当前instrumented path的`baseline_null`重算值作exact比较。366个不合格fold-1请求的6,847
candidate rows中，5,632行存在允许的BF16路径差，最大0.0625；这不是score错误或科学结果。

只修正evaluator，不修改已冻结scorer/merge bytes：从merged metadata绑定的full-population frozen-null
root重新验证method/checkpoint、16万余candidate顺序、metadata/scores SHA和有限分数，再让每个不合格
wrong-history condition逐候选exact等于该frozen-null值。新增测试明确构造recomputed null=0.125、frozen
null=0.1并要求fallback接受后者、拒绝任何偏离。修复后pre-qrels七项audit全部通过，shared evaluator才
打开fold 1；fold 0未打开，strict-transfer为1,088请求且全部wrong-user eligible。metrics SHA为
`4f8081a38721f9ea137e1cd38846ce661154cc52bf853acfc84c0c16c66b89ca`，per-request、input bundle与
dev-eval-log identities全部交叉验证。全仓`930 passed, 7 subtests passed in 33.80s`。

Q2-only原始fold-1表出现一个值得后续两模型family确认的composition形态：same-request target-margin从
attention output的`-0.00246`到post-attention residual的`-0.01175`，相邻差`-0.00929`
（raw CI不跨0）；到post-attention RMSNorm output回到`+0.00148`，相邻差`+0.01323`；MLP increment为
`+0.00187`，但block output又为`-0.00893`，相邻差`-0.01080`。这更像skip/residual composition使有害
signed margin state出现、normalization branch暂时衰减、随后完整residual state重现，而不是attention或MLP
increment单独足够。

该模式尚不能写成组件结论：same节点的两模型planned BH未闭合；post-attention/block-output的same-minus-
cross和same-minus-wrong-history方向均不满足注册的negative specificity预期，random-direction control也强烈
相反；七节点same NDCG CI均跨0。因此当前只保留“Q2 component-composition localization candidate”，等待Q3
selected branch、两模型BH和position-preserving reverse necessity，禁止据此提前选择residual架构。

### All frozen-baseline fallback evaluators now audit ineligible rows before qrels

Q2 selected-branch的pre-qrels错误提示了一个可推广风险：D3 attention edge、D5 RoPE与D5 contextual
scorer都把content-control不合格请求的全部condition精确写回冻结full score，但三个shared evaluator此前只
验证score hash、完整有限coverage、eligibility数量与跨模型一致性，没有重新绑定冻结bundle并逐候选验证
fallback值。这会允许不合格行漂移后仍以“copied baseline”进入descriptive full-population结果。

现由单一共享审计重载metadata声明的frozen baseline root，验证method/checkpoint、完整candidate顺序、有限
分数以及metadata/scores SHA逐字节identity；每个不合格request的每个candidate、每个condition必须exact等于
对应冻结分数。三个evaluator还要求`ineligible_scoring=copy_frozen_baseline_score`与逐request布尔eligibility，
并在各自pre-qrels audit新增
`ineligible_conditions_equal_bound_frozen_baseline=true`。这些修改只收紧尚未运行的evaluator，不改变任何
active scorer、已生成score bundle或implementation digest。

微型测试覆盖合法fallback、单condition漂移和冻结identity伪SHA；真实qrels-blind预演进一步重审Q2/Q3的
attention block-13及RoPE block-13四个bundle，每个均为8,000 requests、7,254 eligible，全部通过，期间未
打开qrels。全仓932项测试按互斥A--M/N--Z分片完成：`709 passed, 7 subtests passed`与`223 passed`；
`py_compile`和`git diff --check`均通过。

同一全仓fallback census还发现尚未启动的component-necessity evaluator虽已逐请求验证不合格neutral removal
精确等于full-to-full identity，却只信任scorer metadata声明的full/null baseline BF16最大误差。现该入口也从
`frozen_full_baseline`与`frozen_null_baseline`identity重载全8,000-request冻结bundle，验证完整candidate顺序和
两套文件SHA；再对fold-1每个candidate自行重算full/null absolute delta及path-local BF16 ratio，并要求三项
最大值与metadata逐字相等且ratio不超过1。逐candidate的neutral eligibility还必须是JSON boolean，缺字段不能
再静默解释成ineligible。该修改仅在未来evaluator中，不进入冻结necessity scorer implementation digest。
新增手算ratio与缺失baseline反例后定向回归`20 passed`，全仓互斥分片更新为
`710 passed, 7 subtests passed`与`223 passed`，共933项测试；`py_compile`与`git diff --check`继续通过。

继续审计eligibility本身发现，D3/D5 evaluator此前只要求eligible总数为7,254且六bundle跨模型/层完全一致；
若所有bundle都把相同数量的request等量替换，数量与一致性仍会通过。现共享入口重新读取冻结deep-dive
manifest及Q2/Q3 content-control rows，验证artifact当前SHA、metadata identity与8,000-request顺序，并构造
逐request expected mask；attention、RoPE、context和component-necessity的score-row eligibility必须与该mask
逐项相等。pre-qrels新增`eligibility_matches_bound_frozen_control_rows=true`，因此注册population不再由score
bundle自报。

微型测试专门构造同eligible count但identity漂移；真实qrels-blind预演再次审计Q2/Q3 attention/RoPE四个
block-13 bundle，均以7,254/8,000的逐request绑定通过。相关定向回归`27 passed`，未打开qrels、未改写任何
scorer bundle。

attention-group虽是qrels-blind exploratory localization，不进入confirmatory family，但其512-row均值包含
`neutral_history_kv`，不合格行按零效应回退；因此eligibility或sample漂移仍会改变描述性局部化。该evaluator
现重新验证冻结sample manifest/rows与content-neutral manifest/rows的当前SHA，逐输出行要求selection、request、
candidate和ordinal等于冻结sample，`neutral_history_eligible`等于冻结control；ineligible行的八个GQA group
neutral score还必须逐项exact等于native baseline。pre-qrels明确记录三项新检查，不能用“exploratory”掩盖输入
population漂移。

Q2/Q3真实attention-group block-13各512行均通过新sample/mask审计；定向回归`17 passed`。一次测试命令误列
不存在的`test_attention_group_scoring.py`而未收集测试，随后已用仓库实际存在的evaluator/intervention测试重跑，
该命令错误不产生artifact或科学结果。

上述两轮mask/sample加固后的全仓测试按互斥分片显式取得退出码0：A--M为
`712 passed, 7 subtests passed in 29.77s`，N--Z为`223 passed in 9.91s`，共935项测试。
tracked diff与本轮相关untracked source/test逐文件no-index whitespace审计均无错误。

attention-group的supplemental cross-request history-summary K/V还依赖确定性donor映射；原evaluator只信任
metadata中的mapping SHA与输出行自报donor。现从content manifest绑定的8,000-request records重新构造
`_cross_request_mapping`，按冻结sample candidate ordinal重算每行donor request/candidate，并要求metadata
mapping SHA和512个输出donor逐项一致。Q2/Q3真实block-13均得到同一mapping SHA
`c4cc09d23da015b0b3f4c4ddeec0a02f4dd54395e0c48dd886f2df888cd9ac42`且512/512 donor lineage通过；
定向回归`12 passed`。这只封闭cross-stress provenance，不把exploratory donor effect升级为用户特异性结论。

D4 SwiGLU group localization也使用同一冻结512-row candidate sample和cross-request donor；原evaluator却只要求
每个bundle恰有512个有限行、16组和共同implementation digest，没有逐行重载冻结sample与donor mapping。
这会允许样本或donor identity整体漂移后仍生成描述性group均值。现evaluation入口从immutable deep-dive
manifest重载sample manifest/rows和8,000-request `records_dev`，验证三套当前字节SHA，自行重构
`_cross_request_mapping`，并逐输出行要求ordinal、request、candidate ordinal/item、block以及donor
request/ordinal/item全部相等；metadata还必须绑定records、sample、mapping和deep-dive manifest五项identity。

Q2/Q3真实MLP-group block-13 v2各512行均通过逐行lineage审计，mapping SHA同样为
`c4cc09d23da015b0b3f4c4ddeec0a02f4dd54395e0c48dd886f2df888cd9ac42`。微型测试同时构造donor item与target
ordinal漂移，定向MLP group回归为`9 passed`。该增强只收紧qrels-blind evaluator，不修改active runtime、
scorer、bundle字节或implementation digest；因而后续SwiGLU差异可以排除sample/donor漂移，但仍保持
descriptive-only claim ceiling。

加入该lineage红队后，全仓单次回归取得退出码0：`936 passed, 7 subtests passed in 33.97s`；相关
`py_compile`、tracked diff whitespace与三份本轮文件的trailing-whitespace检查全部通过。

### Block-13 MLP groups do not yet show a cross-model request-specific carrier

在Q2/Q3 block-13 v2各512行通过冻结sample/donor lineage后，对16个预先固定的SwiGLU group作一次不读
qrels、不选择group的早期横向摘要。Q2与Q3的group均值再平均后，`same_minus_null`分别为
`-0.0008850`与`-0.0009079`，`cross_minus_null`为`-0.0003891`与`-0.0006714`，因此
`same_minus_cross`仅为`-0.0004959`与`-0.0002365`。Q2的same-minus-cross有8/16组为正，Q3为6/16；
group-level跨模型相关为`-0.4659`。单行绝对效应并不为零（Q2/Q3 same-minus-cross平均绝对值约
`0.0349/0.0226`），但没有same donor系统性优于cross donor或稳定跨模型group pattern。

该结果只允许记为block-13 qrels-blind descriptive interim：它反对“一个固定SwiGLU group已经形成稳定、
request-specific carrier”的强表述，但不能证明MLP不重要，也不能替代固定block 20/27、selected-block
七节点family、两模型BH或reverse necessity。正式D4结论必须等待六个固定bundle共同进入evaluator；当前
不依据该摘要选择group、层、模型或后续条件。

### Block-13 GQA groups likewise do not isolate a shared fixed carrier

对已经通过冻结sample、eligibility、ineligible fallback和cross-request donor lineage的Q2/Q3 block-13
attention-group bundles作同样的512-row、全8-group、qrels-blind摘要。`history_to_readout_logits_mask`的
8-group均值再平均为Q2 `-0.002686`、Q3 `-0.000153`，跨模型group相关仅`0.1995`；对应value-zero为
`+0.000763/+0.000076`，相关`0.0192`。neutral-history K/V均值为`-0.01465/-0.00261`，group相关
`-0.1692`。三类干预单行绝对效应仍明显高于signed mean，说明intervention能够扰动native score，但正负
抵消且没有固定GQA group跨模型复现。

不分group的formation/summary stress也没有给出大signed shift：Q2/Q3 query-to-history logits-mask均约
`-0.001343`，value-zero约`+0.000977/+0.000122`，cross-request history-summary K/V约
`+0.000366/-0.000854`。这不能反证attention：Q2 selected-block整条attention-output state的充分性候选
仍可能来自跨group分布式作用、block-13与selected layer错位或后续residual composition。正式区分必须等待
固定blocks 20/27的attention edge/group，以及Q3 selected branch和reverse necessity；当前不选择head/group。

### Block-13 RoPE is broadly path-sensitive but not directionally shared

对已通过pre-qrels frozen baseline/eligibility审计的Q2/Q3 block-13 RoPE v2全8,000-request bundle，使用
7,254个冻结eligible requests作不读qrels的candidate-centered摘要。对每个request先计算compression-minus-
expansion的candidate向量并减去request均值，因此统计排除不影响排序的共同score offset。readout-Q、
history-K、paired-QK三类contrast在Q2的candidate-centered RMS分别为`0.07421/0.07990/0.08081`，
非恒定request比例为`93.20%/95.22%/94.78%`；Q3 RMS为`0.03751/0.03807/0.03788`，比例均约
`99.9%`。所以position phase操作确实广泛改变candidate-relative score，而非仅平移logit。

但这不是共同方向机制。相同candidate的compression-minus-expansion delta在Q2/Q3间相关仅
`0.00675/-0.00008/0.00082`。模型内部三条路径也几乎正交：Q2 readout-Q对history-K/paired-QK为
`-0.0042/-0.0032`、history-K对paired-QK为`0.1461`；Q3三对仅`0.0045/0.0126/0.0179`。这反对
“三种RoPE条件只是同一个generic perturbation”，同时也没有支持一个跨模型一致的position failure direction。

该摘要只能说明block-13 path sensitivity，不能判断变化对strict transfer有利或有害；正式position解释仍需
六个固定bundle、36-cell family、compression-vs-expansion BH、compression-vs-baseline support gate和NDCG
`±0.005`等价条件。当前不根据RMS大小选择path、层或模型。

### Block-20 group decomposition narrows a Q2-specific value-transport candidate

Q2/Q3 block-20 MLP-group与attention-group四个bundle也已完成；在解释前先用加固后的evaluator逐行重审
冻结512-row sample、candidate、cross donor、eligibility和ineligible fallback，四bundle均通过。加入block 20
后，MLP `same_minus_cross`的16-group均值在Q2为`+0.001747`且13/16组为正，但Q3仅
`-0.0000076`，跨模型group相关`-0.3413`。block 13到20的group identity在Q2/Q3也不稳定（相关
`0.1930/-0.1193`），因此仍不支持共同固定SwiGLU carrier；Q2弱signed shift只能保留为model-scoped
descriptive pattern。

attention中较值得后续正式检验的是block-20 history-to-readout value-zero：8-group均值为Q2
`-0.009323`、Q3 `-0.000153`，跨模型group-order相关`0.5583`。然而Q3 magnitude接近零且正负各半，
所以group-order相似尚不等于共同功能效应。neutral-history K/V则明确分叉：Q2均值`-0.02261`且8/8组
为负，Q3为`+0.00997`且3/8组为正。logits-mask仅为`+0.001724/-0.000244`，跨模型相关`0.2281`。

这些值是固定sample上的native-score descriptive shifts，不是target-aware ranking endpoint，也没有qrels、
BH或跨block formal family。因此当前只能把“Q2中层attention value transport可能参与后续residual effect”
列为待b27和正式evaluator验证的model-specific candidate；不能称共同transfer failure mechanism，更不能据此
选择head/group或修改队列。

### Q2 fixed-depth groups favor distributed layerwise recomposition over one persistent group

Q2 block-27 MLP/attention group bundles同样通过加固后的512-row sample、candidate、donor、eligibility与
fallback逐行审计，因此可把Q2固定`[13,20,27]`作完整qrels-blind深度摘要。history-to-readout
value-zero的8-group均值由`+0.000763 → -0.009323 → -0.028748`，但block 20到27的group-order相关仅
`-0.0870`；后层native score对value transport更敏感，却不是同一固定GQA group持续负责。neutral-history
K/V在block 20为`-0.02261`且8/8组为负，到block 27衰减为`-0.00366`且仅3/8组为正，block-20
content-state pattern没有简单延续。logits-mask到block 27变为`+0.00873`，但group-order稳定性也仅
`0.0497`。

MLP `same_minus_cross`从block 13的`-0.000496`变为block 20 `+0.001747`、block 27
`+0.002853`；block 20到27的group相关`0.4451`，但block 27只有8/16组为正，且same/null与cross/null
的group range同时显著扩大。这更像late-score sensitivity与部分group-order延续，尚不是request-specific
MLP carrier；Q3 block-27仍缺失，不能作跨模型解释。

综合固定深度只支持一个待验证的Q2局部化：history-conditioned branch effects在中后层增大，但active group
identity随层改变，符合distributed interaction/layerwise recomposition，也可能包含common-score或低精度敏感性。
它不等同“信号反转”或“被某固定group抹掉”。后续必须由Q3 b27、target-aware D3/D4 evaluator、selected-
branch七节点family和position-preserving reverse removal区分有用信号稀释与有害state写入。

### Observed late attention contribution predicts Q2 value sensitivity, but not universally

正式attention-head observation与pattern synthesis已覆盖Q2/Q3 × blocks 13/20/27的全部16 query heads和
8 GQA groups，不作best-head选择。它显示block 27时两模型history attention mass与o-proj contribution norm
都收敛到相同query-head top-3集合`{7,13,15}`及GQA top-3集合`{3,6,7}`；Q2/Q3 query-head top-3 mass
share分别为`0.777/0.649`，GQA为`0.841/0.748`。12个cell/axis中mass与contribution norm相关全部为正，
平均`0.902`。这是late routing concentration的结构共性，但norm没有signed preference direction。

为检查“高mass/norm是否真的更功能重要”，将每个已完成attention-group cell的8个observed GQA values与同一
固定sample上logits-mask、value-zero、neutral-K/V的group mean score effect逐格关联。相关不具有跨cell
稳定性：例如Q2 block 13 contribution norm对value-zero signed/absolute effect为`0.155/-0.300`，block 20为
`-0.352/0.297`；Q3 block 13为`0.875/0.617`，block 20为`-0.040/0.440`。因此attention mass或
contribution norm不是通用causal-importance proxy。

唯一较强的待复现候选是Q2 block 27：contribution norm对value-zero absolute effect相关`0.9370`，对signed
effect相关`-0.7348`；mass对应为`0.8487/-0.7310`。这与Q2后层value-zero native-score sensitivity一致，
提示late history contribution magnitude可能预测value transport被移除后的影响，但每个相关只有8个固定groups、
endpoint仍是qrels-blind raw score，且Q3 block-27 group intervention未完成。故只能记作Q2-specific late-value
cross-link candidate；不得选择top head/group，也不能称attention mass解释transfer failure。Q3 b27和正式edge/
target-aware family是必要复现门。

### Candidate-relative energy is not erased; Q2 is diluted and both models lack stable semantics

为直接检验“中间信号被某层弄没”而非margin反转，读取已完成cross-component synthesis中按request拆分的
candidate-common与candidate-relative全层轨迹。两个模型四个固定阶段的mean candidate-relative energy
change全部为正；late stage平均`fraction_requests_candidate_relative_energy_decreased=0`。因此没有证据表明
candidate-relative hidden energy在后层被普遍擦除。任何“signal disappeared”表述都必须改成更精确的
composition/semantic/readout问题。

Q2确实存在relative fraction的late dilution：hidden candidate-relative energy fraction由early
`0.0586`升至mid-late `0.0822`，随后在blocks 21--27降至`0.0232`；同时绝对relative energy change
却从`0.00453`升至`0.03788`，candidate-over-history delta RMS从`0.0648`增至`0.3403`。也就是说common
component增长得更快，使relative share下降，但relative signal本身没有消失。Q3不复现该模式：relative
fraction从`0.0612 → 0.0878 → 0.1274 → 0.1520`持续增加，late absolute change也为正`0.02660`。
故“common-mode dilution”最多是Q2-specific secondary mechanism，不能解释共同transfer failure。

两模型共同的问题更像semantic organization/readout mismatch。Q2 history-to-candidate delta cosine从
`0.0233`增到`0.5567`，Q3从`0.0119`增到`0.0889`，说明history-conditioned state会逐渐贴近candidate
difference；但真实brand/category相对random-subspace的isotropic multiple很小且不稳定。late Q2 candidate-
relative brand/category仅`0.0119/0.0058`，Q3为`0.0107/0.1030`；candidate-readout category real-minus-
random甚至为`-0.142/-0.251`。这些proxy不证明完整用户偏好语义，但明确反对“更多relative energy=更多可用
preference signal”。

当前更合理的机制边界是：模型保留并放大大量candidate-relative variation，却未将其稳定组织为request-
specific preference direction或可靠映射到native readout；Q2另有late common-mode相对稀释。selected-branch
与reverse necessity将检验有害state是否经residual写入，native-readout diagnostics则检验最后的semantic-to-
score mapping。该结论是descriptive geometry，不宣称具体算子已被证明。

### Native readout amplifies candidate differentiation without reliable transfer utility

继续检查frozen logit-lens与exact scalar rank-null audit后，可以排除“最终readout把所有candidate variation压成
相同分数”的简单解释。Q2 hidden candidate-relative fraction在late降至`0.0232`，但同区间frozen native-
score rank-effective fraction为`0.1837`；Q3分别为`0.1520/0.5104`。更直接地，full-history相对null-
history的candidate-relative score RMS ratio在late为Q2 `1.6333`、Q3 `2.4277`。历史使两模型的候选分数
分化更强，而不是更弱。

这与冻结strict-transfer缺乏稳定增益合起来，更符合“misdirected/non-specific personalization”：模型利用历史
制造了candidate-relative score variation，却没有稳定把variation对齐到正确用户偏好。late common-history
对native-lens score的cosine在Q2仅`0.0623`，Q3为`-0.3576`；same-sign fraction为`0.473/0.320`，说明
history-common score path本身没有可靠一致方向。Q2 late residual native-direction energy为isotropic expectation
的约`47.9×`，Q3约`10.3×`，进一步表明readout方向可被hidden residual强烈激活，但激活强度不等于正确排序。

精确边界保持不变：只有最终scalar candidate score的request-common shift对RankNet/ListNet是数值验证的exact
rank-null（loss delta最大约`2.0e-14`）；hidden common displacement可经下游非线性转为relative score，不能称
因果浪费。Q2 lens是exact native single-position yes/no direction，Q3现有lens只覆盖首个answer token而非完整
teacher-forced likelihood。因此当前支持readout/semantic mapping candidate，不证明readout operator是原因；
Q3 native readout与position-preserving reverse tests仍是必要门。

### Q2 selected-branch effects localize residual sensitivity, not a history-specific harmful carrier

Q2预注册selected block 15的七节点fold-1 evaluator已经终态，可把此前的“residual composition候选”收紧。
same-request full-to-null patch的target-margin效应依次为：block input `-0.00082`、input RMSNorm
`-0.00106`、attention o-proj `-0.00246`、post-attention residual `-0.01175`、post-attention
RMSNorm `+0.00148`、MLP down `+0.00187`、block output `-0.00893`。其中post-attention residual
CI为`[-0.01978,-0.00340]`，block output为`[-0.01732,-0.00060]`；相邻节点差在attention
o-proj→post-attention residual为`-0.00929`、post-attention residual→RMSNorm为`+0.01323`、
MLP down→block output为`-0.01080`，三者raw CI均不跨零。这确实把状态敏感性定位到两次residual
addition附近，而不是attention o-proj或MLP down单独输出。

但预注册的history-specific harmful gate没有通过。该gate要求same、same-minus-cross和same-minus-wrong-user
均为负。attention o-proj的same本身CI跨零，same-minus-cross虽为`-0.01486`，same-minus-wrong却为
`+0.00896`；post-attention residual的后两项为`+0.01331/+0.00834`，block output为
`+0.01435/+0.01096`，均至少有一项与期望方向相反。same与adjacent的NDCG CI也全部跨零。更强的机械警告
来自random-direction-at-recipient-RMS：post-attention residual与block output的target-margin差分别高达
`+0.6615/+0.5888`，说明这些接口对任意大幅方向替换具有强非特异敏感性；不能把absolute-state patch大小
解释成历史信息贡献。

direction-vs-norm factorial进一步支持“方向敏感、尺度非主导”的局部描述：post-attention residual的direction
contrast为`-0.01343`且CI不跨零，而norm contrast仅`+0.00132`且CI跨零；block output相应为
`-0.01133/+0.00213`。但这里的direction只是activation vector方向，不是已验证的preference direction。
相邻节点差也不是可加Shapley贡献。因此当前Q2结论只能是：两次residual composition会放大或改变patched
state对margin的影响，但尚未证明它们写入了正确用户特异的有害历史信号，更没有“反转/抹除”结论。
Q3同一七节点family、两模型BH与position-preserving reverse removal仍是必要门。

### Q2 native readout preserves matched-state utility but exposes endpoint-level misalignment

已终态的Q2 native-readout确认family提供了比frozen logit lens更直接的target-aware证据。在final RMSNorm
input与output两个固定causal cells，same-request state patch都与原full score逐candidate精确相同，
same-minus-full的target margin和NDCG均为exact zero；这同时验证了patch identity，并说明这两个接口都能
完整恢复冻结full behavior。两cell的注册统计完全一致，因此没有证据把full→score效应的改变局部归因于final
RMSNorm本身。

相对null marker，same/full state在冻结`target_nonrepeat_no_candidate_overlap`人口上的NDCG为
`+0.01173`，CI `[0.00373,0.01925]`、BH `q=0.00540`，两个query folds同为正；但同一请求的
target margin却为`-0.03137`，CI `[-0.04079,-0.02183]`、BH `q=0.00080`，两fold同为负。
所以native readout没有抹掉history-conditioned ranking variation，也不能简单称full history在该人口完全无用；
更准确的是，它对完整slate的重排与单一target-margin方向发生分叉。该分叉与前述candidate-relative能量增加
一致：更多分化不保证所有偏好端点沿同一方向改善。

same-minus-cross donor stress同样强：NDCG `+0.03149`，CI `[0.01835,0.04450]`；target margin
`+0.10222`，CI `[0.07602,0.12780]`，两者BH均为`0.00080`且fold 0/1同符号。这反对“native
score只响应任意history state、完全不关心匹配状态”的极端解释。不过cross donor同时改变request/candidate/user，
按预注册边界它不是wrong-user specificity检验，不能据此声称读出已经编码正确用户偏好。

因此Q2的简单readout-collapse假设被明显削弱：最终state到native score的通路能保留matched-state差异和部分
listwise utility。剩余更像candidate-level semantic allocation或margin/readout目标之间的错配，而非所有信号在
最后一步消失。该结论仍只属于Q2；Q3的完整两tokenteacher-forced native score含共享prompt、`P+Yes`、
`P+No`和joint四个cells，必须等其正式结果后才可判断跨模型readout机制。

### Q2 endpoint divergence is tail-sensitive candidate allocation, not only a readout sign error

为拆开上述两个endpoint，进一步读取同一已注册Q2 native-readout score bundle的逐candidate full-minus-null
delta；不新增condition、不重跑scorer。strict surface共有`2,195`个request，其中`2,194`个target-margin
有限。request-common raw score shift均值为`+0.53898`，它是精确rank-null，不解释排序变化。减去每个slate
自身均值后，选定的第一个最高gain target相对增量均值为`+0.02502`，而零gain候选为`+0.39839`；有并列最高
gain时，其余同级正例为`+0.44625`，低于最高gain的正例为`+0.40492`。这表明历史主要增加candidate-relative
分化，并且给非选定候选更大的相对提升；不是把所有candidate variation压成同一分数。

但对每个request把target与所有低于最高gain的候选逐一比较，平均pairwise target-minus-lower delta仍为
`+0.02857`（median `+0.01691`，54.28%为正）；其最不利的lower candidate delta均值为`-0.21419`，仅
10.53%为正。也就是说，多数候选的平均相对排序可以改善，同时少数hard-negative tail获得更大的提升，恰好
拉低“target对最佳低gain competitor”的registered target-margin。该模式比统一的history方向反转更符合
非均匀candidate allocation。

endpoint分布支持这一解释：NDCG增量为正的request占`31.68%`、负的占`27.48%`，target-margin为正的占
`41.89%`、负的占`54.10%`；`10.71%`的request同时出现NDCG正而margin负，反向组合仅`4.65%`。
两endpoint逐request Pearson/Spearman分别为`0.432/0.421`，所以并非完全不同信号。去掉1%、5%、10%
两端极值后，NDCG均值仍为`+0.01178/+0.01124/+0.00904`，margin仍为`-0.03157/-0.02972/-0.02832`；
fold 0/1也分别保持NDCG约`+0.01234/+0.01111`与margin约`-0.03779/-0.02485`。

并列正例会放大target-margin的解释问题：tied-highest组（`783` requests）NDCG为`+0.01701`而margin为
`-0.04439`；但single-positive组（`1,262` requests）仍为NDCG `+0.00783`、margin `-0.02177`，说明
该分叉不能完全归因于target tie-breaking，仍有hard-negative tail成分。该分析只提供Q2 qrels-gated
descriptive geometry，不改变D6的预注册family结果；它把后续机制问题收紧为“candidate-relative tail/calibration
与正确用户特异性是否一致”，而不是笼统的readout sign reversal。

### Training geometry does not support a shared objective-conflict or late-collapse explanation

把冻结的Q2 objective-gradient、full-parameter update、channel anisotropy与Q3 LoRA path/head geometry并列后，
训练侧可以进一步排除两个过强解释。第一，Q2 RankNet与ListNet在strict-transfer上的score-gradient cosine从
base的`0.9661`升至final的`0.9840`，对应cluster CI分别为`[0.9489, 0.9807]`与
`[0.9736, 0.9924]`；recurrence和other-overlap也保持约`0.892--0.984`的高对齐。映射到参数族后，
observed RankNet/ListNet share差的绝对最大均值仅`0.00303`，request-level最大TV为`0.01287`，没有cell
达到预注册的`0.05`阈值。label-shuffle差异也未形成BH支持。因此gross loss conflict不支持作为Q2失败主因，
更不能解释Q2/Q3共同的strict-transfer缺失。

第二，Q3的rank-8 Q/V LoRA没有出现训练后半程突然退化或有效秩坍缩。56条projection path在step 500和final
均为effective rank 8；step-500 `Delta W` norm相对final的逐path平均比例为`0.9059`，范围
`[0.8215, 1.0032]`，Q/V均值分别为`0.8991/0.9127`。step-500到final函数增量方向的path平均cosine为
Q `0.9463`、V `0.9546`，late七层也为`0.9502/0.9477`。所以最终LoRA函数方向大部分在训练中段已经形成，
没有证据支持“后半程把已有迁移方向反转或抹掉”。这仍不等价于step 500已经收敛，也不能排除错误方向在早期
形成；注册optimizer replay负责区分raw gradient、Adam moments、clipping、weight decay和effective delta。

参数化异质性仍然存在，但目前是模型特异候选：Q3 late Q-head participation约`0.750`，低于其early
`0.892`及Q2 late同几何`0.910`；V则在Q3 late仍为`0.956`，与Q2 `0.958`接近。Q2全参数更新本身跨层较
均匀（layer update-RMS CV `0.0735`、max/min `1.33`），attention Q/K/V/O与MLP gate/up/down均获得广泛
更新；late MLP channel participation缩窄和K-norm participation增长只能作为训练几何描述，不能由更新幅度
推出功能伤害。

因此当前跨模型优先级应放在前向的semantic organization、request-specific direction和residual/readout
composition，而不是简单改loss或把Q3 LoRA容量限制当共同根因。Q3 query-adapter concentration可以解释模型
间差异的一部分，但只有Q-routing、attention output和reverse necessity在同一目标层共同通过后才能获得因果
角色；否则它保留为伴随现象。该综合不新增实验family、不选择head/layer，也未读取source test。

### Interim MLP-group readout (descriptive, not a registered evaluator result)

在等待六个正式D4 MLP-group bundle齐备期间，对已经完成的rows进行固定分组的无qrels描述性汇总，作为
工程/科学监控，不改变D4 family的注册判定。Q2在block 13的最大绝对`same-minus-null`组均值约`0.00745`，
block 20约`0.01160`；但block 27出现多个大幅正负组（最大约`+0.148`、最小约`-0.137`），且对应
`cross-minus-null`也同向达到约`+0.121/-0.120`。这种后层分组敏感性不能直接解释为“某个MLP组携带有害
历史”，因为它还可能是候选/方向替换的非特异性响应，且尚未经过六节点的统一qrels-gated evaluator。

Q3在已完成的block 13/20分组中，组均值幅度整体更小（约`0.0034`到`0.0054`），没有与Q2 block 27相当的
单组异常。这个跨模型差异只是当前已完成bundle的描述性线索；它既不能把MLP提升为主要损失模块，也不能
把Q3的b27缺失填成零。正式D4 evaluator、Q3 b27以及随后的位置保持necessity仍是决定性门。所有汇总均
使用已有score rows，没有读取qrels、没有新增condition，也没有按组挑选后启动实验。

### Interim attention-edge rank geometry at block 13 (descriptive only)

同样对已完成的D3 block-13 edge rows做了候选内几何汇总，不打开qrels。这里把每个request的
`history_logits_mask`、`history_value_edge_zero`和`neutral_history_kv`相对`baseline_full`的变化拆成
request-common shift、去均值后的candidate-relative RMS，以及候选pair的顺序翻转率。Q2的三种干预
candidate-relative RMS约为`0.0669/0.0667/0.0860`，但pair flip率只有`1.09%/1.05%/1.82%`；
Q3的RMS约为`0.0354/0.0354/0.0529`，pair flip率却为`22.74%/22.70%/27.10%`。这提示Q3在该
早层的候选分数间隔更容易被attention edge扰动，Q2则表现为较大但相对同序的分数位移；不过两模型
raw score尺度、候选间隔和tie结构不同，不能直接比较RMS大小或把Q3翻序率称为transfer损失。

该结果只说明“attention edge对候选排序的敏感方式可能模型特异”，不是attention路径的因果必要性，也
不能由block 13外推到选定后层。正式D3 edge evaluator仍需六个model×block bundle，并且要与Q2/Q3的
selected-branch、context和position-preserving necessity结果合并后，才可进入H1/H3设计判断。分析没有
新增condition、没有按翻序率挑选层或head，也没有改变注册family。

### Matched surface-balanced control does not rescue the transfer endpoint

补查已冻结的M3 Q2 matched-control/DID synthesis（该控制是诊断训练，不是论文方法）。在固定
`target_nonrepeat_no_candidate_overlap`的`2,195`个request上，surface-balanced相对original mixture的
history-response NDCG DID为`+0.00432`，query-cluster CI为`[-0.00623,0.01509]`，BH `q=0.4215`，且
两fold方向不一致（fold0 `-0.00280`，fold1 `+0.01157`）。因此不能说重平衡采样恢复了transfer utility。

同一DID的target-margin coherence endpoint却为`-0.08123`，CI `[-0.08978,-0.07260]`，BH `q=0.00080`，
两fold均为负（约`-0.0814/-0.0811`）。也就是说，单纯把训练 surface exposure 做平衡，不能同时修复
candidate-level utility 与 target-margin，反而保持/放大了两endpoint的分叉。这削弱“只要调整 recurrence/transfer
采样比例，问题就能解决”的H4强版本；它不能排除更细的pair construction、hard-negative或optimizer状态效应，
但把后续优先级进一步推向candidate-conditioned allocation/readout与可验证的偏好表征路径。该结果使用注册
matched control、共享 evaluator 产物和固定两fold synthesis；没有把诊断控制包装成新方法，也没有读取source test。

在解释性surface拆分里，`target_nonrepeat_other_candidate_overlap`的DID为NDCG `+0.01810`、但
target-margin `-0.45976`；`target_repeat`为`+0.01140/-0.01776`；`target_nonrepeat_no_history`两者均为
exact zero。这进一步把DID的endpoint分叉指向“候选竞争/重叠条件下的relative allocation”，而不是没有历史
时的统一退化或单纯的历史读取缺失。由于这些是解释性surface，不能替代预注册strict-transfer主检验，也不
单独授权模型设计排序。

### 2026-07-20 next-wave extension frozen before outcomes

现有四卡队列仍在运行 Q2 D3 attention-edge（blocks 20/27）与 Q3 D2 fold-1 post-block
确认（blocks 26/27），四张物理 GPU 均被独立 worker 占用；没有打断或抢占这些任务。为响应
“继续深入且扩大范围”的授权，新增并冻结
`experiments/motivation/transformer_next_wave_plan_v1.md` 和对应 manifest。下一波不覆盖
既有 19 项 formal 或 21 项 supplemental，而是补两个明确的机制缺口：

1. N8 joint attention×MLP state removal，计算单组件效应的非线性交互，避免把最终
   `block_output_residual` 覆盖误称为某个上游组件的唯一原因；
2. N9 formation→transport→candidate-readout 的 position-preserving 联合路径，补齐既有
   history→readout edge formal branch 对 query-to-history formation 的覆盖不足；
3. N10 作为后继子波，固定覆盖 Q3 全 28 层 q/v LoRA rank path 与四模型 final-readout
   candidate-gap geometry，不按结果挑 rank、coordinate 或绝对层号。

该 manifest 在效应值、qrels 和 source-test 均未读取前冻结，四卡排程固定为 Q2/Q3 的 N8/N9
首波，随后才进入 N10。下一波仍只产出机制诊断与设计约束，不实现 transfer 架构，也不把
diagnostic patch 作为论文方法。

### 2026-07-20 N9 path-level implementation

N9 now has a separately hashed protocol at
`experiments/motivation/transformer_n9_history_path_manifest_v1.yaml`
(`a35c176b792feabc3b83c1a4bb83fb2fbfafc6966de2774c3f9f52475b74382b`). It
preserves the Q2/Q3 models, standardized internal-dev population, fold-1
evaluation boundary, candidate slate, token IDs, and position IDs. It adds a
path-level diagnostic at blocks 13/20/27: history-summary formation
(`history_summary` query against the semantic query span), history-to-readout
transport (`native_readout` query against the semantic history span), their
same-forward joint mask, and a joint zero-delta identity. The scorer remains
qrels-blind; only the shared evaluator may open fold-1 qrels after score,
hash, and eligibility checks.

The CPU continuation watcher is
`scripts/run_deep_dive_next_wave_n9_queue.sh` (PID 363280 when registered) and
waits for the same terminal sentinel as N8. When released, it uses GPU2/GPU3
for Q2/Q3 while N8 uses GPU0/GPU1. No N9 effect value has been read yet; the
current four-card D2/D3/D5 workers remain untouched.

### 2026-07-20 N10 LoRA rank-path implementation

While the current Q3 post-block workers remain active, the first N10 subwave was
implemented and frozen at
`experiments/motivation/transformer_n10_q3_lora_rank_manifest_v1.yaml`
(`10373ff6ad55d0e23af739c479dce0d9748af66de45d519aa61344f4919e23f6`). The
qrels-blind scorer temporarily replaces all 56 q/v LoRA factor pairs in the
loaded Q3 checkpoint, preserving either one rank-one function contribution
`B[:,r:r+1] @ A[r:r+1,:]` or a registered factor control. It restores every
parameter after each condition and retains all eight rank groups, rather than
selecting a favorable rank. The follow-up queue waits for the N8/N9 shared
evaluators; no N10 model or qrels effect value has been read yet.
## 2026-07-20 N10 candidate-gap/normalization geometry registration

在 N8/N9 等待现有四卡波次闭合期间，补齐了此前仅在计划中出现、尚未实现的 N10
candidate-gap geometry。冻结 manifest 为
`experiments/motivation/transformer_n10_candidate_gap_manifest_v1.yaml`，覆盖 Q0/Q1/Q2/Q3
四个 native scorer 的 final RMSNorm input/output。每个请求保留 full/null 基线、full-minus-null
方向、candidate-common slate-mean 方向和 deterministic orthogonal 方向；方向按逐行
full-minus-null norm 做 0.10 norm-matched perturbation。输出只报告 common score shift、
candidate-relative L2、pairwise order flip 和 mean absolute shift，不打开 qrels。

实现为 `candidate_gap_scoring.py`、`candidate_gap_runtime.py`、`candidate_gap_evaluator.py`
及对应 CLI/queue；Q0/Q1/Q2/Q3 分别复用项目自有 native scorer、Q1 prefix-cache 与 Q3
Yes/No readout，保持 token、candidate 顺序和请求边界不变。candidate-gap 队列作为独立
CPU watcher 已登记，等待 N8/N9 shared evaluator 后接管 GPU1--3；rank-path 队列继续独占
GPU0。四项方向单元测试和全套回归测试通过，尚无 candidate-gap 科学效应值可读。

同一等待窗口完成了一次 effect-blind topology 复核：冻结 Qwen 架构审计覆盖 63 个精确
Transformer interfaces（53 个 config-backed、10 个 runtime/source-backed），18 个 hookable
nodes、43 个 forward interfaces、20 个 training interfaces，forward/training 均无 missing
interface，failures 为空。producer topology 复核登记的 19 个 formal producer 全部有覆盖，
当前 18 个仍由 queue/watcher 接管；qrels 文件未由 audit 打开，scientific effect values 仍为
false。这一复核确认“还有哪些架构接口完全没有入口”不是当前主要缺口，后续重点应放在
已经可 hook 但尚未完成因果分解的 normalization geometry、joint composition 与 path-level
transport，而不是重复做静态接口枚举。

实时 GPU ownership audit 随后确认 4/4 物理卡各有且仅有一个 worker，physical GPU 映射为
GPU0=Q2 RoPE b20、GPU1=Q3 post-block b27 fold1、GPU2=Q2 attention-edge b27、GPU3=Q3
post-block b26 fold1；failures 为空，source test 未打开。N8、N9、N10 rank 和 candidate-gap
四个 CPU watcher 均未抢占这四张卡。

## 2026-07-20 N11 scaled-QK-logit operator registration

在下一波之后再补一层明确的 Transformer 内部算子：既有 attention-edge mask/value
intervention 能回答“历史边是否被使用”，但不能回答 pre-softmax scaled QK logit 的
scale/sign geometry 是否造成信号消失。为此新增并冻结
`experiments/motivation/transformer_n11_attention_logit_manifest_v1.yaml`（SHA256
`9c70ff17a3c5fbef3eb309b5bc0b44ddf9eac1740d2a1e1f43dd7a3120d89275`）以及独立计划
`experiments/motivation/transformer_n11_attention_logit_plan_v1.md`。

N11 保持 Q/K/V、RoPE、causal/additive mask、softmax 后的 value 和 output projection
不变，只在固定 readout rows 对完整 pre-mask scaled QK logits 做 identity、half-scale、
double-scale、sign-flip；Q2/Q3 均覆盖 block 13/20/27、full/null 两条路径和 8000 条
internal-dev 请求。新实现为 `attention_logit_{interventions,scoring,runtime,evaluator}.py`
及对应 CLI/四卡队列，scorer 仍 qrels-blind，evaluator 先做完整 finite coverage 与 identity
审计后才打开 qrels。N11 只提供算子级诊断，不能把 scale、sign、层或 head 直接升级成方法。

截至记录时四张 GPU 仍由上一波四个 worker 占用，N11 仅完成协议、实现和机械测试，尚无
N11 科学效应值。额外的 Q2 block13 CPU smoke 已完成（`result_eligible=false`，
`maximum_identity_delta=0.0`，6 个 candidate rows 全部 finite）；由于 CPU 与冻结 BF16
GPU baseline 的数值路径不同，smoke 的 `maximum_frozen_baseline_delta=2.875` 只记为
机械环境差异，不能作为科学或 transfer 结果。

## 2026-07-20 N12 SwiGLU stage operator registration

为覆盖 attention 之外此前只有 formation/分组观测、尚未正式因果化的 MLP 内部阶段，新增
并冻结 `experiments/motivation/transformer_n12_mlp_stage_manifest_v1.yaml`（SHA256
`e700cd2413b0176f1a0c1b16ca0309574741e2c0d4841eb31d298e9535f185ac`）和对应计划。N12
直接 hook 选定 token rows 的 `gate_proj` 与 `up_proj` 输出：gate donor 仍经过原生 SiLU，
up donor 仍进入原生 down projection；因此可在相同 residual、mask、position、candidate
边界下比较 gate-only、up-only 与 joint composition，并保留 symmetric reverse 与 sign
controls。它不选择 neuron/group，也不把 stage patch 当作方法。

实现为 `mlp_stage_{interventions,scoring,runtime,evaluator}.py` 及 CLI/四卡队列；scorer
不读 qrels，evaluator 先审计完整 finite coverage、native SwiGLU recomposition 和
identity，再打开 qrels。N12 队列等待 N11 evaluator；截至记录时四卡仍被 D2/D3/D5
worker 占用，尚无 N12 effect value。

随后完成 Q2/Q3 block13 的 CPU one-request smoke。两者均 `status=completed`、
`result_eligible=false`、6 条 candidate rows finite，且新加入的 identity fast-path
使 `maximum_identity_delta=0.0`；CPU 与冻结 BF16 GPU 的 baseline 数值差异
（Q2 0.125、Q3 0.0625003）仍仅作为环境机械差异，不进入 transfer 结论。正式 N12
GPU 队列仍等待 N11 evaluator，不重复占用当前四张卡。

## 2026-07-20 N13 Q/K/V projection-stage registration

为了把 N11 的合并 QK-logit 敏感性再拆开，新增并冻结
`experiments/motivation/transformer_n13_qkv_projection_manifest_v1.yaml`（SHA256
`aaea7d8c0943e8110d151c7a7bd2de1817d59a7fdde73d6f45400ed1e189bf82`）及独立计划。
N13 分别对选定 readout rows 的 Q projection、history span 的 K projection 和 V
projection 做 identity/0.5x/2x/sign-flip，其余行、RoPE、mask、softmax 和 o-proj
保持原生。它只是 projection-stage 诊断，不选 head 或层，不升级为方法。

Q2/Q3 block13 CPU one-request smoke 均完成，6 条 candidate rows finite、identity delta
均为 0；CPU 与冻结 GPU baseline 差异仅作机械非结果记录（Q2 2.875、Q3
0.6911208）。N13 四卡 queue 已登记为 N12 evaluator 之后的下一波。全套回归测试现为
`958 passed, 7 subtests passed`。

## 2026-07-20 N14 history-embedding-stage registration

继续向模型入口扩展，新增并冻结
`experiments/motivation/transformer_n14_embedding_stage_manifest_v1.yaml`（SHA256
`c24c92febef125adef09cdfacab3918cf4f4bd3b2aacd45b4e9e33f70de7cc02`）。N14 不改变 token
IDs 或 history span，而是在 `embed_tokens` 输出上只对 history rows 做 identity、0.5x、2x、
sign-flip、zero；query/candidate rows、position、mask、Transformer blocks 和 native readout
均保持原生。它用来区分入口 embedding attenuation 与后续 Q/K/V、attention、MLP 或 residual
阶段的变化，仍是诊断，不是方法。

N14 已完成 scorer/runtime/evaluator/CLI/queue/test 代码登记，正式队列等待 N13 evaluator；
Q2/Q3 block13 CPU one-request smoke 均已完成，6 条 candidate rows finite，identity delta=0；
CPU baseline 与冻结 GPU 的差异（Q2 2.875、Q3 0.6911208）仍只作机械非结果记录。N14
尚未产生 formal scientific effect value。

另审计到 D4→component-necessity→design-synthesis 的旧 watcher 依赖了不存在的前置 run ID，
会使后续 N8/N9 永久等待。没有改动任何冻结协议或科学条件；新增
`scripts/run_deep_dive_mlp_formation_recovery_queue.sh`，仅在当前四个正式 GPU worker 全部
completed 后，用四卡生成原计划的六个 D4 MLP-formation run，随后复用既有 evaluator 和
necessity/design queue。watcher 当前只占 CPU 等待，不重复占用 GPU。

## 2026-07-20 N15/N16 residual-composition and RMSNorm operator registration

冻结架构审计显示 63 个 Transformer interface 均已有入口，当前欠缺的是 operator 级因果隔离，
而不是静态模块枚举。为此新增 `experiments/motivation/transformer_n15_n16_operator_manifest_v1.yaml`
和 `transformer_n15_n16_operator_plan_v1.md`，登记下一组不依赖结果选层的 Q2/Q3 诊断：

- N15 在 block 13/20/27 的固定 readout rows 上，只改变 attention branch 或 MLP branch
  increment 的 residual coefficient（identity、0.5x、2x、sign-flip、zero），保留同一次
  forward 的 incoming residual、branch tensor 和所有下游权重；
- N16 在 input、post-attention、final RMSNorm 分别只改变 variance rescale 或 learned gain，
  以同一 hidden input 做 FP32 重算，identity 始终返回原生输出。

新原语为 `residual_composition_interventions.py` 与 `rmsnorm_interventions.py`；tiny-Qwen
测试覆盖 attention/MLP 两个 branch 和三种 norm scope，8 个测试全部通过，identity logits
最大误差为 0，active control 能改变注册位置输出。正式 qrels-blind scorer/evaluator 仍待
N14 evaluator 之后接入四卡队列，当前不把 smoke 变化当作科学效应。

随后已补齐 N15/N16 的 qrels-blind scoring/runtime/evaluator 公共内核：
`operator_stage_{scoring,runtime,evaluator}.py` 和 `score/evaluate_deep_dive_operator_stage.py`。
N15 队列 `scripts/run_deep_dive_next_wave_n15_queue.sh` 已在 CPU 后台等待 N14 evaluator；N16
队列 `scripts/run_deep_dive_next_wave_n16_queue.sh` 等待六个 N15 scope evaluator。两者均在
启动前检查物理 GPU 全部释放，并按四卡 disjoint waves 调度，避免同一张卡并发两个模型。此时
尚无 N15/N16 formal effect value。

N15 Q2 attention/block13 CPU one-request smoke 和 N16 Q2 post-attention-RMSNorm/block13
CPU one-request smoke 均 `status=completed`、`result_eligible=false`、identity delta=0，
6 个 candidate rows 全部 finite；两者与冻结 GPU baseline 的差异均为 2.875，只记为低精度
环境机械非结果。该 smoke 同时验证了新 runtime 的 manifest hash、resume bundle、条件集合和
shared scorer 调用链。

## 2026-07-20 formal wave handoff audit

最新运行状态显示 Q3 post-block fold-1 b26 与 b27 均已完成（各 3918/3918）。完成触发后，
Q3 selected-branch fold-1 的两个 1959-request shard 已自动接管 GPU1/3；Q2 RoPE b20 与
b27 继续运行于 GPU0/2。四卡 ownership audit 为 4/4 active、无 physical-GPU collision，
且 qrels/source-test 均未被 worker 打开。D4 recovery queue 仍正确等待全部当前 worker 终态，
N8→N16 watcher 依赖链未错误提前启动；截至本记录仍没有 N8–N16 formal evaluator metrics，
因此不报告任何 effect value 或模块排序。

同一时点的 frozen architecture audit 仍为 63 个精确接口、53 个 config-backed、10 个
runtime/source-backed、18 个 hookable node，forward missing interface 与 failures 均为空。
在排除已注册但尚未有科学 effect 的 operator 后，剩余最明确的深层债务是 q/k head RMSNorm、
GQA repeat-KV grouping、Q3 完整 adapter contribution 和 Q1 cache phase boundary；这些已写入
N15/N16 plan 的 N17 后续边界，暂不在当前四卡波次中 outcome-driven 插入。

## 2026-07-20 N17--N20 boundary pre-registration

为避免当前四卡等待期间出现无约束扩展，新增并冻结“只登记、不启动”的
`experiments/motivation/transformer_n17_n20_boundary_manifest_v1.yaml`（SHA256
`e13177bd48c422002359dd3ba1e98f21c9fe8be7c3c460a12860af0074f01760`）及
`transformer_n17_n20_boundary_plan_v1.md`。四个 family 的边界被固定为：

1. N17：q_norm/k_norm 的 variance-only 与 gain-only operator，分开覆盖 Q2/Q3 的固定
   blocks 13/20/27；
2. N18：repeat-KV 边界的 identity/cyclic/reverse/seeded permutation，完整覆盖 8 个 KV
   groups，不挑单组；
3. N19：Q3 全 28 blocks 的 q/v 完整 scaled LoRA contribution，分开保留 base、adapter、
   sum 与 shared-prompt controls；
4. N20：Q1 prefix-cache materialization、cached continuation 与 answer-token phase 的
   phase-matched replacement，严格保留 token/position/cache-key 边界。

四者均标记 `architecture_authorized=false`、`effect_values_read_before_freeze=false`，并要求
N8--N16 evaluator closeout 后才能抢卡。当前 readiness audit 为 40 个注册 artifact 中
24 个已闭合（0.60，纯 artifact closure，不代表科学支持）；19 个 formal deliverable 中
7 个已闭合，故 N17--N20 目前只完善覆盖边界，不提前制造新的结果叙事。

在不占用四张正式 GPU 的 CPU 窗口中，先把 N17/N18 的最小 operator 原语落地并做 tiny-Qwen
identity/active smoke：`qk_head_rmsnorm_interventions.py` 覆盖 q_norm/k_norm 的
variance/gain/zero 分支，`gqa_grouping_interventions.py` 覆盖 repeat-KV 的 cyclic/reverse/
seeded-permutation；对应测试分别为 3/3 与 2/2 通过。两者尚未接入 scorer、qrels 或正式
queue，故这些通过只证明 hook、形状和 native identity 可执行，不是 transfer effect。

同一窗口进一步落地 N19 的 `q3_lora_branch_interventions.py`：它在 PEFT-shaped q/v
projection 上重新组合 `base_layer(x)` 与完整 scaled `B(A(x))`，再只对固定 readout 或
history rows 应用 identity/zero/scale/sign/random-norm-matched contribution。fake-PEFT
unit smoke 为 2/2 通过，并显式审计 native re-add residual；目前没有加载 Q3 正式 checkpoint，
也没有接入 scoring queue，因此不把该原语的可执行性解释成 adapter 的 transfer 作用。

最后补齐 N20 的 cache-side mechanical boundary：`q1_cache_phase_interventions.py` 对
Transformers `DynamicCache` 提供 native identity、same-request rebuild、zero-prefix 和
donor-prefix replacement，先验证层数、key/value shape、batch、prefix length 与 finite
coverage，再允许 cached-continuation 使用。DynamicCache tests 2/2 通过；该模块仍不执行
Q1 scoring、不跨请求借用未审计 cache，也不把 cache replacement 视作模型方法。

为覆盖此前尚未隔离的训练侧内部边界，另行预注册
`transformer_n21_n24_training_boundary_{plan,manifest}_v1`（manifest SHA256
`09ba9e48b56fbd7f0d695d064aaa60c3c8349aea7700c470682615d8434f200d`）：N21 Q3
FP32-adapter/BF16-base cast、N22 LoRA input-dropout、N23 gradient-bridge/checkpoint
recomputation、N24 objective/optimizer effective update。四个 family 均要求至少两个 seed、
forward/gradient mechanical gate 和 frozen utility 对照，且明确
`diagnostic_training_is_paper_method=false`；在 N8--N20 closeout 前只作为 inactive boundary，
不抢占当前四卡或依据结果扩展训练 sweep。

在四卡正式波次仍运行期间，补齐了不依赖模型/数据的训练边界测量原语
`src/myrec/mechanism/training_boundary_diagnostics.py`：它分别提供 mixed-dtype
recomposition residual、可固定 mask 重放的 LoRA inverted-dropout、bridge/recompute
gradient coverage/cosine，以及 raw-gradient 到实际 applied-update 的 family 分解。新增
`tests/test_training_boundary_diagnostics.py` 的 7 个手算断言全部通过；该代码不读取 qrels、
不执行 optimizer step，也没有改变 N21--N24 的 inactive 状态。当前四卡仍优先完成既有
inference wave，训练诊断只有在 N8--N20 closeout 后才可进入正式队列。

随后把 N17/N18 从“只有 hook 原语”推进到可执行的 qrels-blind bundle 链：新增
`routing_boundary_scoring.py`（q/k head RMSNorm 与 GQA repeat-KV 条件）、
`routing_boundary_runtime.py`（独立 resume/manifest/hash/finite-coverage contract）和
`scripts/score_deep_dive_routing_boundary.py`。共享 evaluator 已扩展到
`n17_head_norm`/`n18_gqa_grouping`，仍在完整 score integrity 后才打开 qrels；新增
`scripts/run_deep_dive_next_wave_n17_n18_queue.sh` 固定覆盖 block 13/20/27、q/k 两侧和
全部 GQA permutation，等待 N16 closeout 后才占用 GPU。该链路的静态编译、shell 语法和
routing/evaluator tests 均通过，目前没有提前启动或读取任何 N17/N18 效应值。
当前 N17/N18 watcher 已挂起于 `tmp/20260720_n17_n18_queue.log`，其唯一启动门是
`runs/20260720_kuaisearch_mech_n16_final_eval_v1/metrics.json` 且随后还会再次检查所有物理
GPU 为空；最新相关回归集合为 27 tests passed。

为继续覆盖 Q3 适配器内部而不把 A/B 几何误当成完整路径，新增
`q3_lora_branch_scoring.py`：它在固定 block、固定 readout/history rows 上分别重组
`base(x) + (alpha/r)B(A(x))`，保留 full/null、identity、zero、scale、sign 和
norm-matched random 条件，并复用 Q3 shared-prompt identity。当前该 kernel 及原语测试
通过；N19 的正式 runtime/queue 仍受 N17/N18 closeout 门控，尚未读取效应值。

随后补齐 N19 的 shared qrels-gated evaluator `q3_lora_branch_evaluator.py` 及 CLI：它先
审计完整 8000-request score bundle、identity/finiteness、Q3 method/checkpoint 和
content-control coverage，再打开冻结 dev qrels，输出 strict-transfer 的 target margin、
NDCG@10、full/null/operator-gap contrasts、normalized-query cluster inference 与 BH 校正。
因此 N19 现在具备 scoring/runtime/evaluator 三段式链路；28 blocks × q/v 的正式 queue 仍
待 N17/N18 evaluator closeout 后登记运行。

N19 的固定四卡队列 `scripts/run_deep_dive_next_wave_n19_queue.sh` 现已挂起运行，等待
全部 N17/N18 block/component evaluator metrics；门控打开后按两 block 四 lane 波次覆盖
完整 28 blocks × {q,v}，每个 bundle 独立 resume 目录，随后逐 bundle 走 Q3 evaluator。
当前 watcher 只占 CPU 等待，没有读取 qrels 或占用 GPU。

当前四卡波次运行期间又补齐了 N20 的 qrels-blind cache-phase scoring primitive
`q1_cache_phase_scoring.py`：native cache、same-request rebuild、zero-prefix、严格匹配
长度的 wrong-user prefix，以及 full-sequence no-cache rebuild 共用同一 Q1 answer-token
targets，并检查 token-position/cache-key integrity。它仍是 inactive boundary；正式 bundle
runtime/evaluator 与 GPU queue 会等 N17--N19 closeout 后再接入，避免抢占当前波次或把 cache
替换误写成 transfer 方法。

当前等待 GPU 的 CPU 窗口又落地了 N25 的第一层实现：
`swiglu_formation_interventions.py` 能在固定 token rows 上分别 hook
`gate_proj`、`up_proj`、SiLU gate 和完整 SwiGLU product，支持 identity/zero/scale/sign/
norm-matched-random，并强制 exactly-once fire；`swiglu_formation_scoring.py` 已把四个
operator 的 full/null 条件组成 qrels-blind scoring kernel。Tiny Qwen 的四个 operator
identity smoke 与注册条件测试共 5 项通过，尚未占 GPU 或读取 qrels。

为避免 N20 之后又回到“只看层或只看状态”的窄路径，新增 inactive follow-on
`transformer_n25_n29_followon_{plan,manifest}_v1`（manifest SHA256
`d2581e308f7478a2f91130868f407303e405fca7158494be29b86d2916e202a2`）。N25 固定拆分
SwiGLU 的 gate/up/SiLU/product，N26 固定拆分 final RMSNorm 与 native readout，N27 固定
测试 causal visibility/softmax topology，N28 测完整 pre-mask scaled QK-logit formation，
N29 用 attention×MLP 2x2 factorial 估计非加性 residual interaction。五个 family 都要求
identity、reverse/随机方向、full/null/wrong-user 和 Q2/Q3 replication，且只在
N17--N20 evaluator closeout 后进入四卡调度；当前没有启动这些新 family。

随后 N20 已闭合为可恢复的正式链路：`q1_cache_phase_runtime.py` 绑定冻结 Q1 full/null
 baseline、wrong-user mapping、候选/请求 hash 和完整 8000-request coverage；
 `q1_cache_phase_evaluator.py` 在 identity、phase-integrity、finite-coverage 审计后才读取
 dev qrels，并报告 cache rebuild、zero-prefix、wrong-user-prefix、no-cache rebuild 的
 target-margin/NDCG/transfer-gap 对照。`run_deep_dive_next_wave_n20_queue.sh` 已挂起在
N19 evaluator sentinel，当前不占用 GPU；CPU/py_compile/shell/test smoke 均通过。

N25 随后也接成了四卡可执行链：`swiglu_formation_runtime.py` 绑定 Q2/Q3 冻结
full/null baseline、content-neutral eligibility、固定 blocks 13/20/27 和 50 个完整
条件；`operator_stage_evaluator.py`/CLI 已接入 `n25_swiglu_formation`，并新增
`run_deep_dive_next_wave_n25_queue.sh`，按“两 block × 两 model”填满四卡；每个 bundle
一次记录四个 operator，避免重复 forward，等待 N20 evaluator 后再启动。该 queue 当前
只占 CPU 等待，没有读取效应值。

状态审计发现旧 N8--N16 watcher 实际被历史 D4/D7 sentinel 卡住（D4 MLP formation
metadata 尚未生成），因此单纯等待不会推进 N17。新增并启动
`run_deep_dive_recover_d4_mlp_formation_queue.sh`：它先等待当前四卡波次的四个原始
bundle 完成，再以四 lane 补齐原注册的 D4 Q2/Q3、13/20/27 MLP formation run IDs，
不改变协议、不抢卡、不新选层；完成后原有 component-necessity→N8→N16 queue 会按
既定 sentinel 自动接续。

继续核对 Transformer inventory 后，发现 N25--N29 之后仍有五类不能由 state
patch 借代的 operator debt：token embedding lookup、input/post-attention RMSNorm
的 variance/gain、attention/MLP residual addition、GQA query-to-KV grouping，以及
Q3 q/v adapter contribution。已预注册 inactive
`transformer_n30_n34_followon_{plan,manifest}_v1`（manifest SHA256
`e882079b6eb072c3f06fbb4b5fdea1645ca53559560d4d37c7a7487b1588f3f9`），固定 Q2/Q3、blocks 13/20/27、
identity/position/finite/qrels-blind gates 和四卡 wave A/B/C 调度。这些是下一阶段的
机制诊断边界，不改变当前 D4→component necessity 优先级，也不根据结果选择新的
层、头、token 或方法。

在等待 GPU 的 CPU 窗口又先闭合了 N31 的 operator 原语：新增
`rmsnorm_operator_interventions.py`，对固定 block 的 input/post-attention RMSNorm
分别支持 variance-rescale、learned-gain、sign 和 output-norm-matched-random，
只改选定 token rows，native coefficient/identity 直接复用原输出，并强制 exactly-once
hook fire 与 finite/shape 审计。新增 9 个纯 CPU 手算测试全部通过；尚未接入正式
runtime/evaluator，也没有占用 GPU 或读取 qrels。

随后补齐 N30 的 embedding lookup 原语 `embedding_interface_interventions.py`：
固定 query/history/candidate token rows，支持 zero/scale/sign/
output-norm-matched-random，identity 直接复用 native embedding，并在 hook 内冻结
并核对 input_ids，防止 token/position 边界漂移。embedding 与 RMSNorm 原语联合
CPU 回归 16/16 通过；两者仍未接入正式 scoring queue。

又闭合了 N32 residual-addition 原语 `residual_addition_interventions.py`：在同一
forward 捕获 block input、native attention increment、native MLP increment，保持两个
子模块返回值不变，只在 block output 的选定 rows 重组 `r+a'+m` 或 `r+a+m'`。identity
直接返回 native output，避免 BF16 重组漂移；zero/scale/sign/random controls 共用
identity-bound RMS matched direction。CPU 原语回归累计 19/19 通过，仍未占 GPU。

随后补齐 N33 的纯张量 GQA grouping 原语 `gqa_grouping_interventions.py`：固定
Qwen 的 16 query heads/8 KV heads，显式定义 native repeat、cyclic group permutation
和 within-group rotation，并在 remap 前后保持输入 K/V、head 数和每头向量可审计。
N30--N33 CPU 原语回归累计 25/25 通过；正式 attention wrapper/runtime 仍待后续
协议链路接入。

随后闭合 N27 的 CPU/operator 原语 `mask_softmax_interventions.py`：visibility
替代必须是 native causal mask 的子集，arm 阶段拒绝未来 token，wrapper 阶段再次检查
answer/continuation leakage；temperature half/double 只作用于有限 pre-softmax logits，
保留 `-inf` 拒绝项。N30--N33 加 N27 的联合 CPU 回归 30/30 通过，尚未运行正式
score bundle。

N28 的纯张量形成层也已补齐为 `qk_formation_interventions.py`：对完整
`[batch, query, head, key]` pre-mask scaled-QK tensor 支持有限 key 居中的 half/double、
centered sign-flip，以及按 head 独立 RMS 匹配且 identity-bound 的随机方向；所有
原生 `-inf` mask entry 保持不变。一次 CPU 回归中发现并修复了 masked key 被误计入
random RMS 的问题，当前 N27/N28 及 N30--N33 联合回归 37/37 通过。

同时核对到 N34 的 Q3 q/v adapter contribution 已由 N19 的完整
`q3_lora_branch_{scoring,runtime,evaluator}` 链覆盖，因此把 N34 标记为先整合 N19、
仅在明确 unresolved operator cell 时才允许最小 replication，避免重复运行昂贵的
Q3 adapter sweep。

N29 也已补上独立的 `attention_mlp_interaction.py` bookkeeping 原语：固定
`native_native`、`removed_native`、`native_removed`、`removed_removed` 四个核心格，
逐 request 保留 `both - attention_only - mlp_only + native` interaction，另外把
matched-scale/sign controls 作为完整性细胞而不是混入估计量。当前内部原语联合
回归 41/41 通过；正式 N29 scoring/evaluator 仍等待前序门控。
