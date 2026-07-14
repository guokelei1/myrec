# PPS ProdSearch baseline

This directory vendors the upstream ProdSearch implementation used as a
query-aware personalized-search control. It is a control candidate for the
new direction, not a proposed system.

Upstream repository: `https://github.com/kepingbi/ProdSearch`

Pinned commit: `449335ba652fe7c877a008e154157d7b2a4b0e76`

License: Apache-2.0 (`LICENSE.md`)

Use the standardized record interface, the boundary card in
`experiments/pps_baseline_cards.md`, and the shared evaluator. Generated
training data, checkpoints, logs, score dumps, and artifacts stay under
`data/`, `runs/`, `models/`, or `artifacts/`.

The adapter is an official-code-to-KuaiSearch interface adaptation. It must be
reported with its input boundary and candidate-coverage limitations; it is not
an externally aligned reproduction unless a future protocol establishes that
alignment.
