# C39 halfspace-certified value terminal outcome

C39 terminated at train-internal A1. The authoritative report is
`reports/pps_c39_train_gate.json`, SHA-256
`8dd30fddb73e5298a697ff7896139554e7af0fac4296a5de86c170b56c78c5ec`.
The proposal and execution lock SHA-256 values are respectively
`0f2c4273af77bc02a5767ff6e57f2aebd6a0e203952c8fde156f4f3bd38fbb99`
and `893f2d045496eac64194c6dfb134f2bb2dc5534e41c86bc72df2dcd1b099048d`.

The frozen ordering was respected: C0/C1, proposal lock, label-free feature
encoding, G0, execution lock, and three physical-GPU fits completed before A
labels opened. All five modes had 197,376 parameters and paired initialization.
All 31 label-free A0 checks passed. The primary projection certificate held,
unsupported edges were exactly zero, almost every negative raw edge was changed,
and the primary changed 99.25% of complete orders and top-10 sets relative to
the base. Exact no-history, no-query, and repeat fallbacks held. C39 therefore
failed on utility, not on an inactive or incorrectly implemented mechanism.

On 1,200 untouched Amazon-C4 train-internal A requests, seed-averaged NDCG@10
was 0.215267 for the frozen base and 0.345152 for the eventwise-halfspace
primary. The paired primary-minus-base gain was `+0.129885`, with 95% CI
`[0.105142, 0.154551]`; every seed and request-hash fold was positive. This is
evidence that the common trainable query/history Transformer stack is useful,
not yet evidence for the halfspace primitive.

Every equal-capacity architectural control nominally exceeded the primary:
eventwise raw reached 0.345716 (primary minus control `-0.000564`), global-only
0.346076 (`-0.000924`), post-pool halfspace 0.345922 (`-0.000770`), and the
score-ray-only reduction 0.345817 (`-0.000665`). All four confidence intervals
crossed zero, all four frozen effect-size gates failed, and none had all three
per-seed differences nonnegative. Thus eventwise projection, projection
location, candidate-local vector content, and score-neutral vector content paid
no architecture rent beyond the shared backbone.

The evidence-fidelity checks also failed. True minus same-length-bin wrong-user
history was only `+0.000018`, CI `[-0.002298, 0.002288]`. The candidate-specific
clicked-direction contrast was `-0.001233`, CI `[-0.003374, 0.000893]`. The
large gain over the frozen base therefore cannot be attributed to identifying
the correct user's candidate-relevant history on this cohort.

C39 has status `failed_A1_terminal`. The remaining 399-request reserve and all
upstream dev/test labels, features, and scores remain unopened. No projection
strength, threshold, loss, seed, cohort, or encoder rescue is authorized.

The problem formulation has become more dataset-independent—the primitive has
no dataset/category/query-type branch and was tested on an Amazon cohort never
used by the KuaiSearch lineage—but the architecture has not become better. The
surviving boundary is narrower: trainable unprojected query/history interaction
is a strong backbone, while hand-specified tangent and halfspace geometries are
not load-bearing. A successor must first make true history distinguishable from
matched wrong history at the internal attention/interaction interface and must
beat raw/global matched controls; another post-attention geometric projection
is not justified.
