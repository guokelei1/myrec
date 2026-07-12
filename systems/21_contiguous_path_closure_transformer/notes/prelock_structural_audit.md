# C21 pre-lock structural audit

Status: **label-free; observed before any compact fit label or ranking outcome**.

The immutable C21 selection has SHA-256
`6903d0363d3e95357909c4a73d3ae703e62fd8276d91aac894da9307642384a8`.
It contains exactly 9,000 `train_fit` and 3,000 `internal_probe` requests,
their union is exactly C06 fit, and both have zero overlap with C06 non-fit
roles.

Only packed offsets/masks were inspected to establish execution feasibility:

| role | history >= 2 | median clipped history | p90 clipped history | median candidates | p95 candidates | max candidates |
|---|---:|---:|---:|---:|---:|---:|
| train_fit | 6,950 / 9,000 (77.22%) | 4 | 18 | 24 | 159 | 1,043 |
| internal_probe | 2,336 / 3,000 (77.87%) | 4 | 18 | 24 | 154.05 | 890 |

Thus the primary has a valid path on most requests, while single-event rows
remain honest zero-write cases.  Dynamic batching can retain full candidate
sets under the frozen 3,072 padded-candidate-row budget.  The hash-fixed
wrong-history map is a 3,000-request derangement with exact clipped-history-
length preservation and no self donor.

No label-shaped array, C06 delayed role, dev/test record, qrel, metric or model
outcome was read for this audit.
