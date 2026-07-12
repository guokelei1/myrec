# 2026-07-12 — C46 post-terminal generality audit

This audit used only the already-open C46-A cohort and blind standardized
records. It is formulation evidence, not a C46 rescue, a new candidate, or a
dev/test result. No alternative formula was selected from the outcomes below.

## Cross-domain recurrence surface

A label-free scan of the unified records found:

| dataset/split | requests | repeat-present | multi-repeat |
|---|---:|---:|---:|
| KuaiSearch train | 163,717 | 23.4404% | 12.2205% |
| Amazon-C4 train-blind | 11,199 | 0.5090% | 0.0089% |
| Amazon-C4 dev-blind | 1,876 | 0.4264% | 0.0000% |

The same exact-recurrence architecture would therefore act on almost fifty
times more requests in KuaiSearch than Amazon-C4, while multi-repeat
competition is effectively absent in Amazon-C4. A recurrence-only successor
would be a KuaiSearch-specific mechanism search rather than a general PPS
architecture. C23/C24 already closed the two plausible recurrence-internal
objects (post-anchor evolution and candidate competition), so no C47
recurrence-redistribution gate is authorized.

## Fixed predictive-residual diagnostics

The three frozen C46 true-pair checkpoints generated prefix predictions. For
each event after the first, `r_t = normalize(W item_t) -
normalize(T(item_<t))`. Coverage was 498/600 requests.

1. Mean residual direction scored `0.286017` NDCG@10, below semantic mean
   `0.301470`. True-minus-wrong was `+0.019047`, but its interval crossed zero.
   Clicked true-minus-wrong was positive with CI `[0.000769, 0.011820]`.
2. Surprise-weighted semantic value used
   `(1-cos(pred_t,item_t))*item_t`. It improved to `0.294761`, but still trailed
   semantic mean by `-0.006709`. Its clicked direction and clicked
   true-minus-wrong intervals were both positive.
3. One symmetric error-memory diagnostic used
   `q' = q + mean_t[(e_t r_t^T + r_t e_t^T)q]`. It improved its paired projected
   query base by `+0.002168` NDCG@10 with all three hash folds positive, but the
   CI was `[-0.004838, 0.009265]`. True-minus-wrong was only `+0.000305` with a
   zero-crossing interval and one negative fold.

## Cross-domain ridge-memory formulation diagnostics

A later fixed diagnostic tested the closed-form history projector

`P_H = H^T (H H^T + I)^-1 H`, `q' = q + P_H q`.

This is not a novel operator: Cubit independently registers KRR as a
Transformer token mixer. It is retained only as a strong nearest control and
as evidence about the information object.

- On C46 KuaiSearch strict-nonrepeat A, the plain projector improved its paired
  frozen-BGE query score by `+0.006836`; all three folds were positive, while
  the CI narrowly crossed zero. True-minus-wrong was `+0.008249`, CI
  `[0.000392, 0.016422]`.
- On the already-open C42 Amazon-C4 cohort, the identical formula improved its
  query base by `+0.043399`, CI `[0.034248, 0.052886]`, and true-minus-wrong was
  `+0.038740`, CI `[0.030125, 0.047607]`; every fold was positive.

One prespecified attempt to set KRR observation precision from C46 prefix
prediction surprise failed: weighted KRR trailed its same-representation
unweighted control by `-0.003894`, CI `[-0.007898, -0.000095]`, and lost in all
three seeds. Surprise precision is closed; its weight may not be inverted or
retuned on C46-A.

The remaining parameter-free fidelity law multiplied the Cubit-style mean
write by the candidate's own posterior support,
`rho_c = c^T P_H c`, giving `score = base + rho_c * c^T P_H q`. This conservative
write produced:

| cohort | vs query base | vs plain ridge | true minus wrong |
|---|---:|---:|---:|
| KuaiSearch C46-A | `+0.008291`, CI `[0.000981,0.016248]` | `+0.001455`, CI crosses zero | `+0.008534`, CI `[0.000121,0.017326]` |
| Amazon-C4 C42-A | `+0.063569`, CI `[0.051323,0.076234]` | `+0.020171`, CI `[0.013359,0.027163]` | `+0.062711`, CI `[0.051534,0.074694]` |

These are formulation diagnostics on exposed cohorts, not a promotion. They
justify only a fresh C47 signal gate in which posterior support must beat
Cubit/plain KRR, ordinary attention, and a free candidate scalar gate. If its
margin over plain KRR is not stable on fresh KuaiSearch as well as Amazon-C4,
the whole posterior-support family closes.

These results separate two properties: prediction error contains some
user-specific information, but its direction is not a stable relevance
representation. It may serve only as a future fidelity signal after a fresh
gate proves incremental utility over semantic history; it cannot yet be the
ranking value or architecture premise.

## Reduction and direction decision

The error-memory update is a close instance of prediction-error associative
memory: DeltaNet uses a delta rule for targeted memory updates, Gated DeltaNet
adds adaptive erasure, and Titans uses surprise-driven neural memory. Generic
surprise event weighting is also close to CARD's prediction-error reduction
attention already registered by C01. These forms are mandatory controls, not
paper novelty by renaming.

Decision:

- stop recurrence-only design because its operational surface is sharply
  dataset-dependent;
- do not promote any of the three opened-cohort diagnostics;
- require the next real candidate to use the identical mechanism and mask-only
  interface on both KuaiSearch strict-nonrepeat and Amazon-C4;
- require it to beat ordinary semantic history, a DeltaNet/fast-weight
  reduction, and its matched Transformer control before any dev access;
- retain exact recurrence as a protected baseline contract on KuaiSearch, not
  as the claimed cross-domain primitive.

Primary neighbours:

- DeltaNet: https://arxiv.org/abs/2406.06484
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Titans: https://arxiv.org/abs/2501.00663
- CARD: https://arxiv.org/abs/2601.15673
- Cubit: https://arxiv.org/abs/2605.06501
- Sparse GP attention: https://openreview.net/forum?id=jPVAFXHlbL
