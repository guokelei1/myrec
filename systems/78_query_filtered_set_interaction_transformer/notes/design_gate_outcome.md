# C78 data-free design-gate outcome

Decision: `close_c78_before_repository_data`.

All mechanics, frozen anchors, gradients, fallbacks, candidate permutation,
and event-set permutation checks passed.  Set modes had event-permutation score
error below `1e-6`; the positional control was observably non-invariant.  Every
mode had 219,648 trainable parameters and all 15 fits decreased loss.

The primary was strong and stable.  Across the three seeds, clean and shuffled
supported accuracy were identical at `0.9986/0.9804/1.0000`; shuffle-margin
retention was `1.00000009/0.99999989/1.00000000`.  Wrong history reduced
supported accuracy to `0/0.0035/0`, query mask returned to the base, and
repeat/no-history accuracy was one.  The primary beat the positional control
by `+0.712--+0.734` worst-stratum accuracy, ungated set interaction by
`+0.980--+1.000`, and pairwise set admission by `+0.261--+0.599`.

The binding control was `triadic_set`, which combines C77's frozen
query-authenticated C-H triangle with C78's event-set positions.  It matched or
slightly exceeded primary: primary-minus-triadic worst-stratum accuracy was
`-0.0014/-0.0196/0.0000`.  Therefore set symmetry is useful, but the proposed
query-filter graph is not the strongest realization and does not pay unique
rent.

C78 closes without repository data.  No position, filter, threshold, control,
or real-data rescue is permitted.  Under the terminal budget, the winning
pre-outcome `triadic_set` control becomes the final C80 candidate; this is a
new locked candidate, not a favorable reinterpretation of C78 primary.

Authoritative report: `reports/pps_c78_design_gate.json`.
