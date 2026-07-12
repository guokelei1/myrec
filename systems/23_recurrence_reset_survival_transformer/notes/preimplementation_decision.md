# C23 pre-implementation decision

Decision: **proceed to one locked train-only hard-anchor gate**.

The decision uses only label-free packed structure.  Across all 25,122 repeat
requests there are 68,121 repeated candidates.  For 51,972 of them (76.29%) at
least one later history event follows the last exact occurrence.  Suffix length
has median 2, mean 5.23 and 90th percentile 14; 11.28% of repeated candidates
occur multiple times.  The post-cut pool contains 2,758 untouched repeat
requests, sufficient for a 1,200/600/958 delayed role split.

This proves only that RRST has a nonempty intervention surface.  It does not
show label alignment.  The following outcomes terminate C23-A:

- no material order changes relative to item-only;
- no positive paired utility over item-only;
- tie/loss against any matched learned control;
- suffix shuffle retains the learned advantage;
- pre-anchor changes affect scores;
- query masking, non-repeat or no-history produces a learned write.
