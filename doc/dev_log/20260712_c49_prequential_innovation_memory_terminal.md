# 2026-07-12 — C49 prequential innovation memory terminal

C49 was the first post-C47 candidate to change the learned value
representation instead of another KRR confidence scalar.  Six fixed GPU runs
showed that a two-layer causal Transformer learned strict-prefix item
transitions and that KRR reads of its innovations were highly load-bearing.
This removed inactivity as an explanation.

The direction failed.  On Kuai the innovation memory had a positive but
unstable +0.008191 point estimate over base and tied raw KRR/DeltaNet; its
clicked and true/wrong intervals crossed zero.  On Amazon it fell to 0.221624,
well below base 0.253202 and raw KRR 0.274713.  Wrong history nominally beat
true by 0.022824 and clicked specificity was significantly negative.  The
beta=1 DeltaNet control was even worse, so prediction-error fast weights do
not repair the direction.

This closes replacement of semantic KRR values by prequential errors.  The
only remaining bounded use of the result is a new formulation question:
whether raw semantic memory can remain structurally protected while a
behavioral innovation is confined to its orthogonal complement.  That must be
tested on already-open C49/C47 cohorts before any ranking fine-tuning or fresh
role; simple raw-plus-innovation is a mandatory control.
