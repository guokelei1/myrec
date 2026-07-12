# 2026-07-12 — C69 semantic-null behavior relation terminal

C69 asked whether an open-catalog item relation survives after ordinary LM
semantic similarity is neutralized. A shared two-item Transformer learned an
exactly anchored interaction from adjacent fit-history events; its binding
control used random rather than semantic-matched cross-history negatives.

The implementation behaved as intended, but the scientific gate failed in
both domains. KuaiSearch tied the random-negative control and trailed ordinary
semantic attention. Amazon collapsed far below both controls and assigned a
significantly negative direction to clicked candidates. True history did not
beat wrong history with a positive interval in either domain.

This is a useful generality result rather than evidence for dataset tuning:
the graph, optimizer, negative rule, thresholds, and aggregation were frozen
identically across domains, and labels were unavailable during fit and
scoring. It closes the idea that generic adjacent behavioral compatibility is
the missing PPS value representation. In particular, harder semantic
negatives cannot be treated as an architecture contribution or rescued after
the cross-domain failure.

The next design must not add another item-only sequence objective. It must
make current-query-conditioned ranking direction load-bearing inside the
Transformer and show the same mechanism on both domains, or narrow the claim
to evidence-safe abstention. Dataset-specific event/query fields that do not
exist on both tracks cannot be the proposed primitive.
