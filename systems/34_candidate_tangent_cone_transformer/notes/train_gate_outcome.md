# C34 candidate tangent-cone terminal outcome

C34 terminated at label-free A0.  The authoritative report SHA-256 is
`bcd8068412c511977bc67044dba0c5c4eaf29ef307834db058f8d28eca73d96d`.

G0 passed all five strict-past authentication checks.  Nine fixed GPU fits
(three modes by three seeds) completed with paired initialization, equal 16,384
parameters, finite gradients, exact repeat/no-history/no-auth/query fallbacks,
and zero determinism/permutation error.  The primary was load-bearing: it
changed 39.67% of complete rankings and 7.17% of top-10 sets versus D2p, and
about 41%/10% versus each matched control.  Wrong history changed 39.0%--39.8%
of orders and 6.83%--7.0% of top-10 sets.

Only one of 28 A0 checks failed, but it is the defining falsifier.  Across
15,424 active candidate rows, only 5/2/5 rows across seeds had exact zero cone
support: 0.0324%/0.0130%/0.0324%, below the frozen 0.1% minimum.  Although
99.71% of requests had candidate-distinct displacements, the positive
half-space almost always admitted at least one of several events.  Thus
`ReLU(cos)>0` is not a meaningful fail-closed evidence law under event
multiplicity; it behaves nearly always-on.

No A label opened.  Delayed-B, escrow, dev, and test remain untouched.  C34
cannot be rescued by changing the cone angle, zero-support threshold,
temperature, history length, or cohort.  A successor must compare each event's
candidate support against the competing candidate set before admission, rather
than tuning the same absolute half-space.
