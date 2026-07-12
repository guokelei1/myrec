# Mid-layer tangent residual diagnostic rejection

After C32 closed, a post-terminal diagnostic tested whether the validated
tangent-write law should move inside the frozen BGE Transformer.  Query token
states and authenticated item CLS states were materialized exactly after layer
2 of the four-layer BGE.  Re-running layers 3--4 from the cached states
reproduced every final query embedding with maximum absolute error 0.  A
rank-16 adapter then injected the authenticated tangent into query CLS at that
midpoint; the remaining frozen attention/FFN blocks processed the personalized
residual.

Three seeds trained on the unchanged 5,840 active fit requests.  On already-open
C32-A their gains over D2p were +0.003527, +0.005097, and +0.004566.  The
ensemble gain was +0.004396 with 95% CI [+0.000532,+0.008101], and all folds
were positive under both the diagnostic partition and C32's formal partition.
However, the paired gain over the simpler final-representation C32 was only
+0.000128 with CI [-0.002985,+0.003158].  The extra internal Transformer
execution therefore paid no architecture rent.  Clicked-minus-unclicked
correction was also significantly negative in every seed.

The diagnostic exposed why the single formal C32 negative fold must not be
overinterpreted: the exact same C32 per-request scores are positive in all
three folds under seed 20260921, although they fail one fold under the frozen
20260901 partition.  The formal failure remains authoritative; changing the
partition cannot rescue it.

Nearest-neighbor review also found BeliefFormer (withdrawn ICLR 2026
submission), which projects attention residuals orthogonally inside Transformer
layers: https://openreview.net/forum?id=Ard2QzPAUK.  C32 differs through strict
causal external-memory admission and a single shared ranking query, but a
generic internal orthogonal residual is not itself a defensible novelty claim.

Decision: reject a mid-layer C33.  The next useful test is a new, independently
selected confirmatory cohort with paired, capacity-matched tangent and
unprojected transports from the outset.  It confirms or rejects the C32
architecture rather than adding an unsupported block-level module.
