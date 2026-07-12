# C49 — Prequential Innovation Memory Transformer

C49 changes the history value representation rather than adding another
confidence scalar to frozen semantic KRR.  A causal Transformer predicts each
history event from its strict prefix; the semantic prediction error becomes a
value stored under the event's semantic key.  A differentiable ridge solve
reads those innovation values for the current query.

The initial gate is an exposed train-internal learnability test on C47-A.  It
cannot support a fresh or paper claim.  Raw-value KRR/Cubit, innovation
softmax, a DeltaNet update, shuffled innovation values, query base, and wrong
history are binding controls.
