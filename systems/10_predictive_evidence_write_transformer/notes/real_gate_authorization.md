# Real gate authorization

Decision: **not authorized**.

The frozen synthetic gate failed its primary transfer, nearest-neighbour,
non-collapse, and corruption-retention conjuncts.  Consequently C10 must not
materialize a real cohort, read train-internal labels, allocate a real-data GPU
run, or access dev/test/qrels.  This file deliberately contains no real-data
paths or runnable configuration.
