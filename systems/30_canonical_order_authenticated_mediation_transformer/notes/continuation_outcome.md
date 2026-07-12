# C30 continuation outcome

Status: terminal at A1.  Delayed-B, escrow, dev, and test remain closed.

Canonical item-ID serialization repaired the only C29 A0 failure without
changing weights: deterministic and candidate-permutation max differences were
exactly zero in all three seeds, and all 12 continuation A0 checks passed.  The
ensemble changed 38.33% of complete orders and 8.83% of top-10 sets; wrong
history retained its strong structural effect.

A labels were then opened for the first time.  Utility failed all six frozen
checks.  Primary NDCG@10 was 0.57193694 versus D2p 0.57217738, a difference
-0.00024043 with 95% CI [-0.00258852, 0.00175733].  Per-seed differences were
-0.00136090, +0.00102654, and -0.00038694.  Hash-fold differences were
-0.00146308, -0.00036332, and +0.00098966.  True-minus-wrong equaled the same
negative point estimate because authenticated wrong history reduced exactly to
base.  Clicked-minus-unclicked correction was -0.00036203 with a zero-crossing
interval.

Bounded conclusion: strict causal authentication makes history provenance and
corruption effects reliable, and canonical execution fixes equivariance, but
an independently trained scalar factual-minus-null LM head still does not learn
the correct candidate ranking direction.  C31 must change the candidate-relative
readout/training interface; it may not tune authentication, select seed 20260832,
or open C29 delayed-B while being designed.

Authoritative report SHA-256:
`83a9b757a661e52e79eac2c9a57547ee9309e5a772ef4731a37abf3a032e1280`.
