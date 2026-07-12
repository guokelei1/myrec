# C44 partial-evidence logit flow terminal result

C44 tested whether each history event should allocate mass across candidates
plus null and write that mass directly into centered ranking logits. It was
designed to avoid C35's vector-write alignment problem and C03's expensive
multi-plan OT.

The locked data-free gate showed that the operator is structurally sound and
learnable. However, primary, forced assignment, partial vector write, and
global pooling all achieved perfect clean NDCG in every seed. Null mass was
also slightly below its frozen rejection threshold. Thus neither null nor
direct logit-space writing paid unique rent.

The report retains a status-polarity implementation bug: negative access
declarations made the aggregate D0 boolean false even though every scientific
structural check passed. No rerun is needed because D1 failed independently on
all three matched-control margins.

Decision: close C44 before repository data. The next architecture should not
continue changing only the normalization/output form of the same semantic
edge matrix. C43 and C44 together indicate that the missing object is likely
the evidence representation itself, not pooling, QKV tying, null mass, or
logit versus vector write.
