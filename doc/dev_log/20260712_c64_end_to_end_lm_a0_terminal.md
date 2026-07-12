# C64 end-to-end LM representation probe: A0 terminal

C64 tested the major remaining representation question instead of inventing
another frozen-state operator.  BGE's final two Transformer layers were trained
end to end with a C53-style joint query/history/candidate ranker.  Equal-
capacity adaptive query-candidate and frozen-history modes were trained on the
same sampled candidates and exposed-fit split.

All nine fits completed.  Unlike C61--C63, the model was clearly rank-active,
so frozen representations were part of the earlier inactivity problem.
However, correct versus wrong history was not consistently active in Top-10:
two seeds changed only 7 and 5 of 1,200 sets.  Full validation scoring also
exposed bf16 candidate-order numerical dependence, but repairing it would not
change the binding failure.  A0 therefore stopped before validation labels.

The updated boundary is narrower and useful: greater representation capacity
can move rankings, but it still preferentially learns a generic query-candidate
reranker rather than a stable personalized evidence path.  Do not sweep LM
depth, LoRA/full fine-tuning, epoch, sampling, precision, or seed on this role.
Any successor must make user-history dependence a structural training object,
not merely supply more adaptable layers.
