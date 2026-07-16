# Motivation V1.1 robustness and population extension

This directory contains the pre-registered follow-up to Motivation V1.
Motivation V1 remains immutable under `doc/40_*.md` and the reports it cites.
V1.1 may add evidence, but it may not rewrite, replace, or selectively omit a
V1 result.

V1.1 is ordered. Axis A keeps the V1 KuaiSearch population fixed and tests
training sufficiency (more epochs). Axis B then keeps the selected epoch fixed
and replaces only the earlier training window with a larger strictly
preceding source-train window. A run that changes both is exploratory and is
not binding evidence. JDsearch is the downstream second-population fallback
because its source boundary passed the functional product-search audit; its
anonymized term IDs do not support a plaintext semantic claim. KuaiSAR and
Amazon-C4 remain role-limited as described in `doc/10_direction_decision.md`.

The frozen protocol is `protocol.md`. Raw records, checkpoints, scores, and
logs are not tracked here. The V1.1 report must be a new file under `reports/`
and must cite every seed and every pre-declared checkpoint outcome.
