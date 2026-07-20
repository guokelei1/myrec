# Motivation Transformer deep-dive plan

状态：2026-07-18，用户在首轮机制诊断完成后明确授权继续扩大深度与广度。本计划扩展
`mechanism_analysis_plan.md`，但不修改其冻结的首轮 probe manifest、产物或结论。当前阶段只做
机制诊断与架构约束，不实现新的 transfer 方法，不更换数据集，不打开 source test。

## 1. 已知起点与新问题

首轮诊断已经建立三个约束：

1. Q2 在部分层存在可解码的 brand/category 偏好代理，但 Q3 不稳定复现；
2. Q2/Q3 的正确 block-13 full state 移动方向与负的 full-minus-null margin 相反，而正确
   block-27 state 重现有害 full response；
3. post-block patch 混合了 attention、MLP、两次 residual、RMSNorm 与最终 readout，因而不能
   说明符号反转由哪个 Transformer 组件产生。

新阶段回答两个互补问题：

- **纵向深挖**：从 embedding 到 native readout，符号、用户特异性和候选相对性在哪一层、哪一
  分支丢失或翻转？
- **横向补齐**：attention head/edge、MLP、RMSNorm、RoPE、Q0/Q1 readout、loss 分量、AdamW
  有效更新与 LoRA rank path 中，哪些此前未测组件能解释现有行为？

本阶段不把“某个模块有非零响应”自动称为 preference mechanism。一个 preference-specific
结论仍必须同时通过 same-request、identity、cross-request、frozen null-marker 与相应结构负控。

## 2. 不可变边界

- 数据仅使用 `full_confirm_preceding40k_v11` 的 train 与 label-free internal dev；legacy 2k、
  new 4k 与 source test 不参与 probe 选择、调参或正式评估。
- Q0--Q3 config、checkpoint、第一轮 score、`protocol.yaml`、`probe_manifest.yaml` 与
  `motivation_mechanism_first_diagnosis.*` 均冻结，不覆盖原 run。
- scoring/instrumentation 不读取 dev qrels；只有共享 evaluator 在 score/coverage/hash/qrels
  边界检查后读取 dev qrels。train-only gradient/optimizer probe 可读取 train qrels。
- 原始 item ID 只用于完整性、candidate 对齐、surface 与负控 donor 排除，不序列化为模型特征。
- 单个连续 GPU job 不超过 13,500 秒，必须以完整 shard/cell 原子续跑；每个 GPU job 有独立
  run directory 和 lineage。
- smoke、identity 失败、OOM、hook 未触发、backend 不一致或 coverage 不完整只记 mechanical
  non-result，不能进入机制统计。
- 不按结果增加模型、层、head、MLP group、position condition、seed、surface 或 endpoint。

## 3. 证据层级与统计规则

### 3.1 证据层级

1. `mechanical_smoke_non_result`：最多 32 个稳定哈希请求，只验证 hook、shape、resume 和边界；
2. `numerical_identity_gate`：最多 128 个请求，比较原 scorer 与 no-op/identity instrumentation；
3. `registered_mechanism_diagnostic`：完成预注册人口、全部条件与负控，才可进入综合；
4. `exploratory_localization`：预注册的高维 head/group 曲线全部保留，但不单独形成主要结论。

identity gate 要求：Q0--Q3 native score 最大绝对误差均不超过 `1e-5`。attention
instrumentation 必须用 no-op wrapper 委托冻结 runtime 实际选择的 backend；eager 只在短 prompt
上交叉核对在线摘要，不作为 formal scorer。任何 wrapper、explicit-position 或 native-position
路径不满足 gate 时，相应正式结果不得启动，不能放宽阈值。

score identity 与低精度代数重组分开判定：前者固定绝对阈值 `1e-5`；后者逐 tensor 固定为
`max_abs_error <= 4 * eps(dtype) * max(1, max_abs(reference))`，并同时报告 FP32 重组误差。这个
dtype-aware bound 只用于 BF16/FP16 的 residual、SwiGLU、o-proj、RMSNorm 与 RoPE norm 机械审计，
不能替代 scorer identity，也不能被解释成科学等价。

### 3.2 统计

- 主要人口为 internal-dev strict transfer；recurrence、other overlap、observed-positive 与 overall
  为解释面。
- 主要方向 endpoint 为 target-versus-best-lower-gain-competitor score-margin change；NDCG@10 为
  共同 utility endpoint。两者均使用共享 evaluator 与 normalized-query cluster bootstrap。
- bootstrap 固定 5,000 draws、seed `20260715`；两 fold 固定为 normalized-query SHA256 mod 2。
- 每个 stage 形成独立、预注册的 FDR family；全层/head/group 曲线全部报告，不选择最佳单元。
- 对负的 full-minus-null denominator，mediated fraction 只作带符号描述；主要推断使用
  `patch - null` 和 `patch - full` 的请求级差值，禁止把负 ratio 写成 recovery percentage。

确认性 family 在 manifest 中冻结如下；fold 只作发现/确认或方向门，不另算 hypothesis。identity、
精确重组与 no-op wrapper 是机械门，不进入 FDR。曲线、范数和高维几何若未列入下表，一律描述性
报告，不能据 `p > 0.05` 写成 weakened/rejected：

| Family | Hypothesis unit | Unit count |
|---|---|---:|
| D1 region decoding | model 2 × position 3 × label 2 × region 4 × contrast 2 | 96 |
| D2 Q3 all-native scientific gate | block 2 × patch-minus-null 1 | 2 |
| D2 Q3 position scope sensitivity | block 2 × all-native-minus-first-only 1 | 2 |
| D2 all-layer margin | model 2 × block 15 | 30 |
| D2 adjacent causal transition | model 2 × adjacent step 14 | 28 |
| D2 selected-block same | model 2 × main node 7 | 14 |
| D2 same-minus-cross stress | model 2 × main node 7 | 14 |
| D2 same-minus-wrong-history specificity | model 2 × main node 7 | 14 |
| D2 adjacent-node contrast | model 2 × adjacent main-node pair 6 | 12 |
| D2 direction/scale factorial | model 2 × main node 7 × scale contrast 3 | 42 |
| D3 attention aggregate causal | model 2 × block 3 × comparison 3 × endpoint 2 | 36 |
| D5 RoPE causal | model 2 × block 3 × phase mode 3 × compression-minus-expansion 1 × endpoint 2 | 36 |
| D5 contextual controls | model 2 × context condition 2 × endpoint 2 | 8 |
| D6 Q2 native readout | node 2 × comparison 3 × endpoint 2 | 12 |
| D6 Q3 native readout | state/joint cell 4 × comparison 3 × endpoint 2 | 24 |
| D6 Q0/Q1 branch extension | model 2 × block 3 × aggregate node 4 × comparison 2 × endpoint 2 | 96 |
| D7 Q2 objective conflict | state 2 × surface 3 × registered statistic 2 | 12 |

D1 的两个 contrast 是 real-minus-random 与 full-minus-null excess。D2 的 all-layer、transition、
node 与 scale 表先列 primary target-margin family；对应 NDCG 按同一 unit/count 建立独立 secondary
family。cross-request 只是同时改变 query/candidate/user 的 stress control，只有
same-minus-wrong-history 能支持 history specificity。D3 的三个 comparison 是 logits-mask、
value-zero 与 full-to-neutral K/V，各自相对 no-op identity；8 个 GQA group 的全量观测曲线和稳定
哈希样本因果定位是 descriptive，不进入该 family。D4 的 16-group activation与固定样本 patch也只
作 descriptive localization，full MLP branch 的确认性结果已在 D2。D6 comparison 固定为
same-minus-null、same-minus-full、same-minus-cross。D7 的 12-unit family只检验 Q2
RankNet/ListNet per-request gradient conflict；effective-update与 LoRA vector geometry 保持 exact
descriptive，不能单独改变 H0--H5 状态。

D7 的两个 registered statistics固定为：(a) 每请求 RankNet-gradient 与 ListNet-gradient cosine 的
cluster mean；(b) observed cosine minus within-request-label-shuffle cosine。确认 states固定为
base initialization与 frozen final checkpoint；step-500用于 effective-update replay并作描述。
cosine SESOI固定为 `±0.1`：冲突需要区间完全低于 `-0.1`，practical equivalence需要区间完全落入
`[-0.1,+0.1]`。parameter-family update-share difference仅以 `±0.05` 作描述性敏感度带，不做
显著性或可加 component claim。

bootstrap p 值固定为双侧 tail probability并带 `+1` 修正：
`min(1, 2*min((1+#draw<=0)/(B+1), (1+#draw>=0)/(B+1)))`，之后在上述 family 内做 BH。D2
fold-0 只用于选 `j`；selected-block 的确认性 bootstrap、p 值与 BH 只使用 fold 1。identity失败会
阻断整个对应 subfamily；模型/qrels 读取前已冻结的 anchor eligibility 之外，机械失败 cell 不可换
层、换 token、换 population或缩小 family，报告为未启动/机械缺失而非科学 null。所有未运行、
门控停止或机械缺失的预注册 hypothesis cell 固定记 `p=1`，BH 的计划 family 大小 `m` 保持上表
不变，不能删除缺失 cell 后重算。

科学上的“无效应”等价性仅对 NDCG 使用首轮已注册的最小效应尺度 `±0.005`；区间完全落入该
范围才可称 practically equivalent。target margin 没有跨模型可辩护的统一 SESOI，因此只允许
方向/反方向结论，不允许因区间跨零声称等价或无效应。机械 identity 使用上述 `1e-5` 数值阈值，
不得与科学等价性混用。

## 4. D0：instrumentation、identity 与容量审计

### 4.1 通用节点接口

在项目代码中为 Qwen3 暴露以下可直接 module-hook 的节点，不能改写冻结 ranker 的训练或
evaluator：

- `block_input_residual`；
- `input_rmsnorm_output`；
- `q_pre_norm`、`k_pre_norm`、`q_post_norm_pre_rope`、`k_post_norm_pre_rope`、
  `v_projection`；
- `attention_head_output_pre_o`、`attention_o_projection`；
- `post_attention_residual`；
- `post_attention_rmsnorm_output`；
- `mlp_gate_projection`、`mlp_up_projection`、`mlp_swiglu_product`、
  `mlp_down_projection`；
- `block_output_residual`；
- `final_rmsnorm_input`、`final_rmsnorm_output` 与 native lm-head input/output。

post-RoPE Q/K 和 attention edge 不是 `nn.Module` 输出，必须由 project-owned
`AttentionInterface` wrapper 在收到已旋转 Q/K/V 后捕获或干预；不能声称普通 forward hook 已
覆盖。`attention_head_output_pre_o` 必须按冻结 config 显式 reshape 为 16×128，再按真实
`o_proj` column slice 重组。lm-head 只保存注册 token logits、logsumexp 或 top-k，不落全词表。

每个 hook 必须记录 shape、dtype、device、layer、token span、触发次数和 implementation digest。
所有正式 patch 只作用于明确 token position/span，禁止广播到未注册 token。

D0 除 score identity 外必须逐项验证：28 层 hook 触发次数；
`block_out = block_in + attention_increment + mlp_increment`；
`down_proj_input = SiLU(gate_proj) * up_proj`；head slices 经 `o_proj` 权重重组等于原输出；RoPE
前后 Q/K norm 保持；final RMSNorm 重组一致；wrapper no-op 与原 backend 一致。Q1 的 prefix 与
continuation 是多次 forward，触发次数按 phase/call 登记，不能沿用单次 forward 假设。

### 4.2 四卡 smoke

| GPU | 模型 | smoke 重点 |
|---:|---|---|
| 0 | Q2 | residual/attention/MLP/final-norm capture 与 patch identity |
| 1 | Q3 | PEFT wrapper、全部 answer prediction positions 与 patch identity |
| 2 | Q0 | specialized reranker 的层解析与 single-token readout identity |
| 3 | Q1 | listwise prompt、KV-cache 与 multi-token likelihood identity |

D0 所有 identity 通过之前，不启动正式 attention backend、position 或 readout结果。

## 5. D1：全 28 层表示与状态轨迹

### 5.1 固定设计

- 主 anchors：Q2、Q3；hidden state indices 固定为 `0..28`，不再只采五个 state。
- 位置：`query_end`、`history_summary_end`、每个 `candidate_readout`。
- train probe：原 manifest 的稳定哈希 train population；condition=`full`。
- internal dev：完整 8,000 requests；conditions=`full,null`。
- labels 与 split 原样继承首轮 category/brand probe；random-label、state-0 均保留。

### 5.2 端点

- 每层 real-label 与 random-label balanced accuracy；
- full-minus-null excess decoding 与两 fold 方向；
- full-minus-null L2/sqrt(hidden)、cosine 与 RMS ratio；
- 固定四个层区间 `0--6`、`7--13`、`14--20`、`21--27` 的平均曲线摘要；
- 不报告 outcome-selected best layer 作为确认性结论。

### 5.3 四卡并行

首波 GPU 0/1/2/3 分别负责 Q2-dev-full、Q2-dev-null、Q3-dev-full、Q3-dev-null；较短的 train
bundle 在对应 dev job 完成后的同卡队列续跑。若单 bundle 超过四小时，按完整 request shard
续跑，不能换 population 或减少 state。全人口只落 29 个 residual states 与预注册在线摘要；
Q/K/V、o-proj 和 MLP 的所有 raw tensor 若全落盘预计超过 1 TB，故只在稳定哈希 512-row
mechanical sample 保存 raw vectors，正式全人口保存 scalar/head/span summaries。该 512-row sample
在 model/qrels前固定为 internal-dev中 `history_present AND candidate_id not in original_history_ids` 的
全部 candidate rows，再按
`SHA256("deep-dive-fixed-candidate-rows-v1",request_id,candidate_id,ordinal)` 排序取前512；文件和
request/candidate identity由 deep-dive manifest绑定。

## 6. D2：native-position gate、全层因果定位与 Transformer 分支分解

### 6.1 Q3 native-position gate

分支归因前先重跑 block 13/27 的 Q3 `first_position_only` 与 `all_native_positions`
same/identity patch，覆盖 native score 的三个不同 hidden
states：共享 prompt 最后位置、teacher-forced `Yes` 后位置和 teacher-forced `No` 后位置。原 M2
prompt-state patch 在 block 27 只影响首 token，不等价于完整 Q3 readout mediation。新 gate 必须
对两个 target branch、四个 log-prob term 全部满足 identity。科学 gate 另成独立两单元 family：
`E13_all = margin(all_native_same_b13) - margin(null)` 预期 `>0`，
`E27_all = margin(all_native_same_b27) - margin(null)` 预期 `<0`；两者都必须 point estimate 为
预期符号、fold 0/1 同为预期符号且 BH `q<0.05`，Q3 才进入 b13--27 sweep。
`all_native_positions - first_position_only` 在 b13/b27 是另一个两单元 scope-sensitivity family，
用于量化首轮 readout scope偏差，不代替上述科学 gate。

### 6.2 blocks 13--27 的 post-block causal sweep

Q2/Q3 对 blocks `[13,14,...,27]` 全部运行 post-block patch，不用 D1 表示曲线代替因果定位。
每层固定：

- `full_to_full_identity`；
- `null_to_null_identity`；
- `same_request_full_to_null`；
- `cross_request_same_layer`。

absolute full state 与 `null + (full - null)` 数学上相同，只保留前者，不登记重复实验。每个模型
对 request `i`、block `k` 定义 `e_ik = margin_i(same_k)-margin_i(null)`，并使用 fold 0 的
注册序列 `E_k = mean_fold0(e_ik)` 选择
`j = argmin_{k=14..27}(E_k - E_{k-1})`，tie 取较小 block；选择记录和输入 SHA 在读取
fold 1 结果前原子冻结。fold 1 只确认固定的 `j-1 → j` 相邻负转折，不能改选第二个 layer。
若 fold 0 的最小相邻步不为负，则不产生 `j`，该模型停止局部分支归因；若 fold 1 不复现固定
负转折，层定位标为 unresolved，selected-block 结果只能探索性报告。固定 breadth 结果仍全部保留。
Q2 block 13/27 的既有审计 bundle在 intervention语义完全一致时直接复用，不重复计算；Q3 因
all-native-position scope改变，不用首轮 first-position bundle替代新结果。

### 6.3 固定节点与 composition-safe patch

每个模型只对其预注册选择记录中的 block `j` 报告七个主节点，并用 `j-1` post-block 结果作为
相邻转折边界：

1. block input residual；
2. input RMSNorm output；
3. attention increment（`o_proj` output）；
4. post-attention residual；
5. post-attention RMSNorm output；
6. MLP increment（`down_proj` output）；
7. block output residual。

SwiGLU product到 MLP increment的重组、block-output 与 `u+m`、以及 final-norm input 与 block-27
output只作路径等价检查，不作为独立机制发现或 family unit。节点效应统一称
“null-context sufficiency intervention”，不解释为可加和贡献或 Shapley decomposition。

若 `j=27`，额外描述 final RMSNorm input/output，但不增加 D2 family。post-attention residual 不能
只 patch `post_attention_layernorm` 的输入，因为 DecoderLayer 的 Python 局部 residual仍会保留
recipient 值。其正式实现固定为：令 recipient block input 为 `r_N`、目标 post-attention state
为 `u_F`，在 self-attention output处写入 `a* = u_F - r_N`，并断言随后实际 state 等于 `u_F`。

每个 node 的 patch/control 固定为：

- `full_to_full_identity` 与 `null_to_null_identity`；
- `same_request_full_to_null`；
- `cross_request_same_layer`，沿用冻结 deterministic donor mapping；
- `matched_wrong_user_history_to_null`：recipient query/candidate保持不变，只换 qrels-blind wrong-user
  history state；
- `donor_direction_at_recipient_rms`；
- `recipient_direction_at_donor_rms`；
- `random_direction_at_recipient_rms`，方向由固定 seed生成。

前两个尺度 control 分离方向与 norm，random-direction 只作非特异负控；cross-request只作 donor
stress，matched wrong-user才检查 history specificity。随机方向由
`(node,block,request_id,candidate_id,position,seed)` 决定，与 batch/shard/resume无关；head tensor
按逐 head RMS，普通 node按完整 trailing dimension RMS。所有 patch 在 persistent per-model worker中
以固定小 cell batch执行，模型只
加载一次，但每个 cell 仍有独立 run contract、resume shard 和输出 SHA。

对 donor absolute state `F`、recipient state `N`，令 `dir(x)=x/(RMS(x)+1e-12)`。三个 control
absolute states固定为 `D@R=dir(F)*RMS(N)`、`R@D=dir(N)*RMS(F)`、
`Z@R=dir(z)*RMS(N)`；`z`是上述 identity-keyed Gaussian方向。零 RMS cell机械失败且不换样。
这里的“direction”只表示 activation vector方向，不声称它已经是 preference direction。

wrong-user mapping在 model/qrels前一次性物化并由 manifest绑定 SHA。target是 internal-dev request，
donor pool只来自 train；target query/candidate始终保留。target可见数固定为
`H=min(6,len(original_history))`，`H=0` 冻结为不合格。每个 donor只取其时间顺序最后 `H` 个事件，
并依次要求：different user、`H` 个事件全部早于 recipient `ts`、其 item IDs与 recipient全部
candidate及全部 original-history item IDs不相交。只在满足这些硬约束且恰有 `H` 个可见事件的
donor中，最小化 Qwen冻结 tokenizer对 `serialize_history` 的 token-length绝对差；tie按
`SHA256("deep-dive-wrong-user-v1", recipient_request_id, donor_request_id)` 最小值。没有合格 donor
的 request冻结为不合格。正式 score bundle仍保持全 request/candidate coverage：不合格 request的
该 condition写回 frozen null score；specificity inference只使用 manifest冻结的 mapped-eligible
surface和计数，不能事后改变。

### 6.4 精确 contrasts 与判定

对 selected block `j`、node `n`，均在 fold 1对 request-level margin先作差再 bootstrap，固定：

- same sufficiency：`S_n = mean[margin(same_n)-margin(null)]`；
- donor stress：`S_n-C_n`，其中 `C_n=mean[margin(cross_n)-margin(null)]`；
- history specificity：`S_n-W_n`，其中 `W_n=mean[margin(wrong_n)-margin(null)]`；
- adjacent node：`S_{n+1}-S_n`（按七节点顺序）；
- norm contrast：`S_n-D@R_n`，即 donor direction/donor RMS 对 donor direction/recipient RMS；
- direction contrast：`S_n-R@D_n`，即 donor direction/donor RMS 对 recipient direction/donor RMS；
- specific-direction stress：`S_n-Z@R_n`，即原 donor state 对随机 direction/recipient RMS。

all-layer unit就是 `E_k`，相邻层 unit就是 `E_k-E_{k-1}`。所有 family仍做双侧检验；机制
“通过”还必须同时满足预注册期望符号、fold 0/1同符号和 BH `q<0.05`。selected harmful
bottleneck 的期望符号固定为 `S_n<0`、`S_n-C_n<0`、`S_n-W_n<0`；三种 scale/direction
contrast若用来支持 harmful donor信息，期望也固定 `<0`。反向显著结果原样报告为相反机制证据，
不改写假设。`patch-null` 是主要效应；`patch-full` 只作 reproduction/descriptive，不进入上述 gate。
target margin没有 SESOI，禁止用“相当”“等价”或不显著来通过 gate。

- attention branch 被视为候选瓶颈，仅当 attention output patch 满足上述 same、stress、specificity、
  scale/direction 的 point-sign、两 fold方向和 BH门；
- MLP 同理；
- 若单独 attention/MLP 均不足、post-block 才出现反转，结论保持为 residual composition；
- final RMSNorm/input-output 的差异只能说明尺度/方向重映射，不能独立称 preference readout。

## 7. D3：attention head、edge 与 Q/K/V

### 7.1 固定观测

blocks `[13,20,27]`，所有 16 query heads 与 8 KV groups 全部报告。正式路径用 project-owned
`AttentionInterface` wrapper 接收已 RoPE 的 Q/K/V 与真实 additive mask，并继续委托原
SDPA/backend 产生未干预输出；只对注册 query rows 在线聚合，不构造/保存完整 attention matrix：

- history-summary query 对 query/history span，以及 readout query 对 query/history/candidate span
  的 attention mass；
- 每个 span 的 value contribution，以及按真实 `o_proj` column slice所得 per-head vector norm与
  cosine；中间 head不定义 signed scalar contribution，只有完整 causal score delta带符号；
- Q/K pre/post norm 与 RoPE 前后 full-null delta；
- 两个 query heads共享一个 KV head 的固定 GQA group 摘要。

### 7.2 固定因果干预

- history→readout logits mask：只改注册 readout query row、所有 heads jointly，将 history-key
  additive entries在 softmax前设为 exclusion并重新归一化；其他 query row不变；
- history→readout value-edge zero：在每个 readout head输出中减去 baseline未重归一化的
  `sum_history softmax_weight * V`；不把全局 V tensor置零，其他 query row不变；
- full K/V history span→等长 content-neutral recipient；neutral span严格沿用 D5冻结 token recipe；
- cross-request 仅 patch 单个注册 `history_summary_end` K/V token；不同长度的完整 span 不做直接
  transplantation；
- 机械 controls固定为 `zero_additive_delta`，以及先实施 mask、再把注册 readout query的 head output
  恢复为 baseline原值的 `mask_then_restore_output`；两者都必须过 score identity；
- 全 head joint 干预进入完整 internal-dev确认性 family；全部 8 个 GQA groups原样报告全量观测，
  逐 group因果只在 manifest冻结的稳定哈希样本上作 descriptive localization，不做不完整人口 NDCG。

query→history formation edge 与 history→readout transport edge分开，不把外部 relevance filtering
等同于内部 attention routing。

## 8. D4：MLP feature groups 与 residual geometry

blocks `[13,20,27]` 的 3,072 维 SwiGLU product按固定 SHA256(dim index, seed `20260718`)分成
16 个等容量 group。全人口在线报告 activation summary；逐 group patch只在 manifest冻结的稳定
哈希样本上作 descriptive localization：

- full/null activation RMS、Hoyer sparsity、cosine 与因果 score delta；中间层不使用未经定义的
  signed readout projection；
- same-request group patch、cross-request group patch、group-only permutation结构负控；
- 数值 identity 必须同时 permutation SwiGLU dimensions并对 `down_proj` input columns做精确逆置；
- attention increment、MLP increment 与 residual 的夹角和范数合成。

group 结果属于 exploratory localization；只有 full-branch patch 与独立 readout/behavior证据一致时，
才能把 MLP 作为主要机制。禁止按 group 结果追加 neuron 或扩大/缩小 group。

D3/D4 的 `[13,20,27]` 是独立固定 breadth，不依赖 D2选出的 `j`，过各自机械 identity后必须运行。
若 `j` 不在三点中，可另生成一个 `j` 的 adaptive descriptive slot，但它不进入确认 family、不能
替代固定 anchors，也不能声称固定 anchors解释了 `j`。

## 9. D5：位置、RoPE 与固定长度输入控制

Q2/Q3 先从 instrumented prompt 生成 query/history/candidate semantic span map。contextual controls
使用同一 token 长度与默认 position IDs，固定：

1. unmodified identity；
2. history content neutralization：只替换 history span token IDs，attention mask 与 position IDs 不变；
3. history attention-null：token IDs/position IDs 不变，只屏蔽 history key span。

content-neutral的精确 recipe固定为：对完整 internal-dev 8,000 requests及全部 candidates，先由
冻结 prompt builder在最终 context里定位被保留 `history_text` 的精确 substring，只替换该 substring
编码所得全部 token IDs为 `<|endoftext|>=151643`；prefix/query/candidate/suffix及 history span外的
分隔符保持原 token，attention mask、长度和 position IDs逐元素不变。找不到 span、无可见 history
或 span在截断后为空的 request在 model/qrels前冻结为不合格，该 condition写回 baseline score；
eligibility count/request SHA在 manifest冻结。D3 neutral K/V使用同一 token recipe，不另造 neutral。

RoPE 因果条件固定 blocks `[13,20,27]`。AttentionInterface实际收到 post-RoPE Q/K，因此正式
变换固定为 `q'=R(p')R(p)^T q`（K同），token IDs、mask 与其他层 position不变。`H` 固定为截断后
实际保留的 history-content token数；history K scope是全部保留 history span，anchor缺失的 eligibility
在模型/qrels前冻结且不替换 token：

1. `readout_q_distance_compression`：只将 readout Q 的相位位置减 history span length；
2. `history_k_distance_compression`：只将 history K 的相位位置加 history span length；
3. `paired_qk_distance_compression`：Q只减 `floor(H/2)`、K只加 `ceil(H/2)`，总相对压缩仍为 `H`。

每个 mode 增加等幅 `distance_expansion`；科学 contrast 固定为 compression-minus-expansion，active
与 no-op identity同时描述。这样 common-offset只承担机械恒等门，不被误用为一般扰动负控。

允许 phase position 与自然 token index不同，因为这是 layer-local RoPE机制干预，不把它伪装成
合法自然序列 position IDs。另设两类机械 identity：显式传入 Qwen 默认共享 `arange(S)`（包括
left padding，不能改成 attention-mask cumsum）；以及所有 token position统一加 `+17` 的 common
offset RoPE invariance。任何 condition若改变 query/candidate token identity、candidate slate 或
非注册 layer/head，直接机械失败。

## 10. D6：native readout 与 Q0/Q1 扩展

### 10.1 Q2/Q3

- Q2：lowercase yes/no token IDs 固定为 `9693/2152`，native score 精确分解为
  `h_finalnorm · (E_yes - E_no)`；报告 final RMSNorm 前后 hidden direction/norm、tied rows与
  yes/no logit common-offset control；tied rows只经 output-only shadow lm-head/logit intervention，
  不能原地改参数并同时改变 prompt input embedding；
- Q3 target 固定为 `[Yes=9454, im_end=151645]` 与 `[No=2753, im_end=151645]`。native score由
  三个不同 hidden states上的四个 log-prob terms组成：共享 prompt state预测 Yes/No，Yes context
  与 No context分别预测 terminator；完整两 token平均 log-likelihood差仍是唯一 native score。
  首轮 block-27 prompt-state patch只影响首 token的限制单列；
- frozen null-marker counterfactual、history-dependent common offset 与 candidate-relative residual分解；
  不另造含糊的“query-only”自然语言 prompt。

Q2 causal cells固定 final RMSNorm input/output两个 nodes；Q3固定共享 prompt、`P+Yes`、`P+No` 与
三者 joint四个 cells。null recipient就是 frozen null-marker request；same donor是完全相同
request/candidate/native-position的 full state；cross donor沿用首轮冻结 SHA-ring successor并以
recipient candidate ordinal modulo donor slate取 donor candidate。identity固定 null→null与
full→full。正式人口是 label-free
dev全部请求/候选，scorer不按 strict/target选样。comparison固定 same-minus-null、same-minus-full、
same-minus-cross。每请求 slate的精确代数分解固定为
`common_i = mean_j(score_ij)`、`relative_ij = score_ij-common_i`；null-marker counterfactual不属于
该代数恒等式。

### 10.2 Q0/Q1 广度扩展

- Q0/Q1 固定运行全层 residual trajectory、blocks `[13,20,27]` 的 branch aggregate 与 final
  RMSNorm/readout；定义与 Q2/Q3 一致，不能因结果修改层或 node；
- Q1 保留完整候选 slate 和原 KV-cache scoring，覆盖每个 response token，不改成 pointwise proxy；
- Q0 specialized pretraining boundary单列，不与 General Qwen 做参数 matched claim。

## 11. D7：loss 分量、AdamW effective update 与 LoRA path

### 11.1 Q2

- 复用首轮固定的三个 train surfaces、96 requests/surface、observed/label-shuffle 与
  base/step-500/final states；
- 分别反向 `pairwise_ranknet`、`listwise_softmax` 和固定 0.5/0.5 combined loss；
- 参数族覆盖 embedding/readout、Q/K/V/O、RMSNorm、MLP gate/up/down，并按所有 28 blocks聚合；
- frozen final state 已完成 `967/967` steps 且 terminal LR 为零，只做 terminal-LR audit，不把零
  delta称为机制结果。正式 nonterminal replay绑定
  `artifacts/motivation_v1_2/resume_canary/q2_step500_seed20260714` 的 exact step-500 state；恢复
  AdamW moments、scheduler 与 clipping，分别对完全相同的 surface microbatch 做 step-501
  one-step replay；每次从同一 `(theta,m,v,scheduler,RNG)`恢复；
- 报告 raw gradient、clipped gradient、Adam preconditioned direction 与实际 delta-theta 的 norm、
  cosine和 parameter-family share。cosine pair中 RankNet-vs-ListNet是上述确认性 endpoint，其余 pair
  只描述。parameter-family必须是 mutually-exclusive partition，share固定为该 family squared-L2
  除全参数 squared-L2；tied embedding/readout只计一次。Adam preconditioned direction不含 weight
  decay，最终 delta另分报 moment update与 weight-decay项。96 requests/surface按稳定哈希顺序冻结为6个16-request blocks；
  Q2每个 block保持 `batch_requests=1 × accumulation=16`。每个 block独立从相同 step-500
  snapshot恢复并执行
  `backward → scaler.unscale → global clip → AdamW.step(current LR) → scaler.update → scheduler.step`。
  RankNet、ListNet与 combined loss从同一 snapshot独立 replay；只有 raw gradient满足机械恒等
  `g_combined=0.5*g_ranknet+0.5*g_listnet`，clipping/moments/v/weight-decay后的 delta不可称可加
  component share。

### 11.2 Q3

- base/step-500/final、三个 surfaces 与 label shuffle 不变；nonterminal replay绑定
  `artifacts/motivation_v1_2/resume_canary/q3_step500_seed20260714`；
- 同样按稳定哈希顺序冻结6个16-request blocks，每个 block保持原生
  `batch_requests=2 × accumulation=8`，并从同一 step-500 snapshot独立恢复；
- 对每层 q/v LoRA A、B 分别报告梯度与有效低秩更新 `B@A`；
- 固定 `alpha/r=2`，函数变化精确写为
  `delta_DeltaW=2*((B+dB)(A+dA)-BA)`，分报 A-only、B-only、joint与二阶 interaction；
- 八个 coordinate方向全部描述；机制结论使用 `DeltaW=2BA` 的降序 SVD gauge-invariant方向，重复或
  近零 singular values只解释子空间；
- 区分零 B 初始化导致的 base A-gradient=0 与训练后真正的方向变化。

A-only/B-only replay中未激活 factor保持 `grad=None`，且不得接受 optimizer moment或 weight decay；
joint才同时更新。step-500与final逐层/q-v对 `DeltaW`作SVD；near-zero固定为
`sigma <= max(sigma1*1e-6,1e-8)`，相邻 singular value relative gap `<1e-4`视为 degenerate，只比较
对应子空间而非单向量。manifest必须绑定 step-500 trainer/progress SHA、scheduler state、当前 LR、
model identity及稳定 parameter-order digest；Q2 step-501也执行同样绑定。

LoRA机械控制包括固定正交 gauge变换 `A'=RA,B'=BR^T` 输出 identity、merged/unmerged identity、
SVD重组 identity，以及 base `B=0` 时 A-only函数更新为零。

optimizer probe 只用 train qrels，不运行 dev evaluator，也不把一步更新当作性能结果。

## 12. 四卡执行波次

| 波次 | GPU 0 | GPU 1 | GPU 2 | GPU 3 | 启动门 |
|---|---|---|---|---|---|
| W0 | Q2 D0 smoke | Q3 D0 smoke | Q0 D0 smoke | Q1 D0 smoke | manifest frozen |
| W1 | Q2 train all-layer | Q2 dev all-layer | Q3 train all-layer | Q3 dev all-layer | D0 identity PASS |
| W2a | Q2 b13--27 sweep | Q2 fixed breadth queue | Q3 position scientific gate A | Q3 position scientific gate B | native-position identity PASS |
| W2b | Q2 fold-0 select/freeze | Q2 fold-1 confirmation | Q3 b13--27 fold-0 A | Q3 b13--27 fold-0 B | Q3 scientific gate PASS |
| W2c | Q2 selected branch A | Q2 selected branch B | Q3 fold-1 confirmation A | Q3 fold-1 confirmation B | per-model selection record frozen |
| W2d | Q2 branch continuation | Q2 fixed breadth continuation | Q3 selected branch A | Q3 selected branch B | fold-1 confirmation materialized |
| W3 | Q2 attention heads/edges | Q2 MLP groups | Q3 attention heads/edges A | Q3 attention/MLP queue B | backend-wrapper identity PASS |
| W4 | Q2 position/RoPE | Q2 native readout | Q3 position/RoPE | Q3 native readout | explicit-position identity PASS |
| W5 | Q0 residual/readout | Q1 residual/readout | Q2 component/effective update | Q3 LoRA path | preceding formal completeness |

同一波次的任务独立写目录；GPU 空闲时只允许启动下一张已过门的注册任务，不允许临时增加条件。
因既有实测 Q3 patch约为 Q2 的 2.48 倍，causal queue固定采用一张卡处理 Q2、三张卡处理 Q3，
或由 manifest中的 deterministic weighted queue产生等价分配；不能按中间结果手工换 cell。每个
原子单元固定为 `(model,block,node,patch_kind,q3_answer_path,request_shard)`，worker在 13,500 秒
边界安全退出。

W2是硬拆分协议：全层 score先只物化fold 0；evaluator只读fold-0 qrels并输出序列；选择器原子写入并
哈希 `j` 记录后，才允许 scorer/evaluator materialize/reveal fold 1；fold-1确认完成后才生成
selected-block七节点 contract。`j`冻结前不得运行任何 selected-branch identity或结果 cell。

## 13. 交付与停止点

科学阶段门先于“全部跑完”：

- Q3 all-native-position gate若不保留 b13/b27 的注册反转，停止 Q3 branch localization并报告
  positional-scope artifact；
- 某模型 fold 0没有负相邻步，或 fold 1不复现固定转折，则停止该模型 selected-block确认归因；
- same不优于 matched-wrong-history，不能称 history/preference-specific；same不优于 cross时只能说
  donor stress不分离；
- direction/scale gates不通过，不能形成 donor-direction或 norm机制结论；但 D3/D4固定
  `[13,20,27]` breadth仍在各自identity通过后运行。只有 adaptive `j` head/group slot由 D2门控；
  attention/MLP aggregate都不通过时，selected-block解释停在 residual composition；
- 若只有 post-block/final norm变化，结论停在 residual/norm interaction；
- RoPE只有在 active NDCG区间不落入 `±0.005` 等价带且 compression区别于等幅 expansion时才可
  支持 position explanation；否则保持 unresolved或在完整等价条件下 weakened；
- readout same-request与 cross不分离且 NDCG落入等价带时才可削弱 readout bottleneck；代数恒等式
  本身不构成机制支持；
- optimizer/LoRA若没有预注册 cosine/share SESOI，只保持 descriptive/unresolved，不以非显著性
  反证目标或低秩路径；
- 任何门停止后不追加层、head、group、seed、模型或数据集，独立的 D1/D5/D6/D7广度任务仍按注册
  完成，不用某一纵向门替代用户要求的横向覆盖。

本阶段最终交付需要同时具备：

- frozen deep-dive manifest、实现 identity 和四卡 lineage；
- Q2/Q3 全层曲线及 attention/MLP/residual/norm/readout 分支结果；
- head/edge、MLP group、position/RoPE、Q0/Q1、optimizer/LoRA 的已注册结果或有绑定 failure
  record 的明确 mechanical non-result；
- 所有正式 score bundle 的 pre-qrels audit、共享 evaluator metrics 与 append-only dev ledger；
- 一份更新后的 Transformer component evidence matrix，逐组件标记 supported、weakened、
  unresolved、untested/mechanical-failure；
- 一份更新后的架构机会排序，明确新证据是否改变 factorized signed preference path 的优先级；
- 边界审计、全量测试、冻结资产哈希和 source-test-closed 证明。

到此停止并等待用户是否授权实现新 transfer 架构。不得用本阶段内部诊断结果覆盖首轮报告，
不得把 outcome-selected head/neuron 或 diagnostic control 包装成论文方法。
