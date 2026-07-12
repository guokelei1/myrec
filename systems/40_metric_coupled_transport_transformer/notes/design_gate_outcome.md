# C40 metric-coupled transport terminal design-gate outcome

C40 closed at data-free D1. The authoritative report is
`reports/pps_c40_design_gate.json`, SHA-256
`ac845ea629adaf9f142c5a18d19fdf15410da1399895309fc1a9d066acda9406`.
The design-lock SHA-256 is
`c5fd328aef77457ba44967bd64339fb724b927731226d7b9497f71a025fca900`.

The first process stopped before D0 because deterministic CUDA required
`CUBLAS_WORKSPACE_CONFIG`. It wrote no report and exposed no outcome. The same
locked command restarted with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no source,
generator, threshold, seed, or training setting changed.

All eleven D0 checks passed on physical GPU 3. Every mode had 1,024 parameters
and paired initialization, both low-rank factors received gradients, outputs
were finite and deterministic, candidate permutation error was zero, and all
registered fallbacks were exactly zero.

The planted task confirmed conditional capacity. Base NDCG@10 was 0.442527.
Across seeds 20261901/02/03, the primary reached 0.837572, 0.832310, and
0.844982; clean-minus-wrong margins were 0.458776, 0.446333, and 0.465955. It
beat the single-wide control in every seed and all fallback, corruption, and
activity checks passed.

C40 nevertheless failed architecture rent. In seed 20261902 the primary beat
the equal-parameter shifted loop by only `0.007568`, below the frozen `0.01`.
More importantly, `selection_only` scored 0.881485, 0.866857, and 0.877672,
higher than the primary by 0.043914, 0.034548, and 0.032690 in every seed.
Learning a closed value/readout metric was unnecessary even when the teacher
used that loop; preserving raw semantic values was the stronger rule.

Status is `failed_D1_terminal`. C40 read no repository dataset, qrels, dev, or
test input and receives no real-data authorization. It may not be rescued by
changing the teacher, margin, epochs, rank, or seed. A successor may test
learned routing with immutable LM-semantic values/readout, but identity values
alone have direct Transformer precedents and are not a novelty claim.
