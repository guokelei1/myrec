# C43 Cross-Domain Metric-Coupled Transformer

C43 is the one-shot KuaiSearch transfer falsifier for the metric-coupled
multi-head Transformer that survived C41/C42 on Amazon-C4. It is not an
Amazon rescue and does not tune on C42.

The C40/C42 operator, four heads, rank, temperature, scales, optimizer, loss,
epoch count, fit count, and all outcome thresholds are frozen. The only model
shape change is the mechanically required LM hidden width from 384
(`bge-small-en`) to 512 (`bge-small-zh`). C43-A is exactly the union of C37
delayed-B and escrow, both feature/score/label unopened when C37 closed.

Dev/test records and qrels are forbidden. A positive result establishes a
cross-dataset architecture foundation, not a novelty claim; a negative result
closes metric coupling without a KuaiSearch rescue.
