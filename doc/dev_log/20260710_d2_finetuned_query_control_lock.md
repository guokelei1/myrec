# D2 Fine-tuned Query Control Lock

Locked at 2026-07-10T17:54:02+08:00, before D2 token materialization,
training, scoring, or dev evaluation.

- Protocol: `doc/19_finetuned_nonpersonalized_control_protocol.md`
- Protocol SHA-256: `b87278c4f9906435358ff766db399916a9b0541bc0d32e0471431ab95049f575`
- Config: `configs/analysis/finetuned_nonpersonalized_control.yaml`
- Config SHA-256: `89cd643133385d54a02454158b834d722488a2ba5b167d5172ead6e17bd03885`
- Fixed variants: D2t fine-tuned text; D2p train-only text/popularity mix.
- Fixed seeds: 20260708, 20260709, 20260710.
- Authorized dev evaluations: six.
- Test use: prohibited.

This is a disclosed post-D1 robustness control. D1 metrics motivated testing a
trainable text encoder, but no D2 result may alter its model family, alpha grid,
epoch rule, retry rule, or interpretation branches.

Train-only calibration subsequently froze epoch 3 and D2p alpha 0.6 before any
D2 dev scoring. Final config:

- `configs/analysis/finetuned_nonpersonalized_control_final.yaml`
- SHA-256: `fc05bdd490fee7167bb80aa2ca31f2f5cf17566b2ed73761dca806009f121204`
