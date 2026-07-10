# D2h Static Waterline Lock

D2h was added after observing the fixed D2 results and before any D2h
calibration, score generation, or dev evaluation. The extension is required for
baseline fairness: D2t is stronger than B2z, so the static query/history
waterline must be reissued with D2t before system design.

The global alpha grid, tie-break, three true-history runs, three wrong-history
runs, comparisons, and interpretation branches are frozen in doc 20. D2h does
not retrain a model and test remains prohibited.

- Protocol SHA-256: `a062bca37ec2cb56260804882728e0ff7965b184b07ba6fdf7c965b9e29bd937`
- Config SHA-256: `b34b722f8172dd07f4e5e03b71950d91ac6e9de54cc7acbcb29ac49c4161692b`

Train-only calibration selected alpha 0.1. Before any D2h dev scoring, the
final config was frozen at
`configs/analysis/d2h_static_history_control_final.yaml`, SHA-256
`9b7d32314d4b951211c3bb96bca2d8464e1fb989058690a5d4316b43de4000d4`.
