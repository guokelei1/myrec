# C43 authenticated-history post-terminal diagnostic

After C43 had already terminated, a read-only diagnostic reused its exact
three frozen `multihead_coupled` checkpoints on the same opened 1,200-request
C43-A cohort. No weights, thresholds, architecture, cohort, or labels were
changed. Dev/test and qrels remained closed.

The only intervention replaced raw request history with C37's strictly
prequential user-memory authentication: an event was retained only when it
had appeared for the recipient user at a strictly earlier timestamp. All
requests sharing a timestamp read memory before that timestamp group was
committed. Wrong-user donor events were tested against the recipient user's
memory, not the donor's.

Authentication retained 6,310 true events across 687/1,200 requests
(`0.4043` mean retained fraction). It retained no wrong-user donor event, so
the authenticated-wrong condition fell back exactly to the nonpersonalized
base. Seed-averaged NDCG@10 was:

- base / authenticated wrong: `0.595612`;
- authenticated true: `0.597458`;
- authenticated true minus base/wrong: `+0.001846`, 95% bootstrap CI
  `[-0.000804, +0.004459]`.

Each individual checkpoint had a positive true-minus-wrong mean
(`+0.001388`, `+0.001175`, `+0.001192`), but the paired request interval
crossed zero. This is diagnostic evidence, not a rescue or a new gate result.

Interpretation: evidence hygiene restores the intended direction, so the
representation layer matters. It does not recover a statistically reliable
or practically sufficient effect, and therefore does not validate C43. The
next proposal should make strictly prequential behavior events first-class
contextual tokens and learn their relation to query and candidate inside the
Transformer. Continuing to tune pooling, QKV coupling, transport geometry,
or history preprocessing alone is not justified by this result.
