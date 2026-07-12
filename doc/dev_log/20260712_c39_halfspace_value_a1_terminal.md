# C39 halfspace-certified value A1 terminal

C39 cleanly separated a useful common Transformer backbone from its proposed
value law. All 31 label-free A0 checks passed and the active primary improved
over frozen BGE by `+0.129885` NDCG@10 on a fresh Amazon-C4 train-internal
cohort. However, raw-event, global-only, post-pool, and ray-only equal-capacity
controls all nominally scored higher. True history was indistinguishable from a
same-length-bin wrong-user history, and the candidate-specific clicked-direction
contrast was not positive.

The terminal interpretation is not that personalized history lacks value: C38
already established a cross-domain true-over-wrong signal. It is that C39's
halfspace projection does not preserve or recover that identity-specific value.
The large C39 base gain belongs to the shared trainable history Transformer, not
to the proposed eventwise value geometry.

This reduces research-overfit risk rather than validating an architecture.
C39 contains no Amazon-specific branch, its A cohort was untouched before the
lock, and matched controls were trained before labels opened. Nevertheless, a
positive claim remains unauthorized because it has only one Amazon
train-internal result and the defining mechanism failed. The next primitive
must target true-versus-wrong evidence discrimination inside attention or token
interaction, with a pre-outcome corruption falsifier; no tangent/halfspace
projection continuation or opened-cohort tuning is allowed.

Authoritative report: `reports/pps_c39_train_gate.json`, SHA-256
`8dd30fddb73e5298a697ff7896139554e7af0fac4296a5de86c170b56c78c5ec`.
