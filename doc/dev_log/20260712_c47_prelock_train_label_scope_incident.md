# 2026-07-12 — C47 prelock train-label scope incident

During a C47 cohort-availability audit, the operator correctly used only
label-free histories/candidates to identify 2,370 strict-nonrepeat requests
remaining in the union of C26 internal-A, delayed-B, and escrow after excluding
known opened outcomes. The audit then incorrectly opened the packed
`train/candidate_labels.npy` array before a C47 proposal/selection lock and
counted how many of those requests contained a positive.

Only one aggregate was emitted (`2370/2370`); no request ID, candidate label,
ranking score, metric, subset comparison, or model output was inspected. Dev,
test, and qrels were not accessed. Nevertheless, this is prelock label access
and the full 2,370-request pool is conservatively outcome-exposed for C47.

The sorted packed-index list is reproducible from existing selections and has
SHA-256 `7f2d3c8c8456f8ced06575303356c6cd1eb8b00a2af22c0c6c014fff72ce83aa`
under compact JSON serialization. Its range is `[94, 96911]`.

Decision:

- all 2,370 requests are forbidden as C47 A/delayed/escrow outcomes;
- they may be reused only as fit data because their train labels are now open;
- the label array must not be read by C47 selection code;
- fresh C47 Kuai A must be selected only from label-closed C34-A/C36-A after
  removing every opened outcome and all incident indices;
- this incident does not affect Amazon-C4 or any dev/test protocol.
