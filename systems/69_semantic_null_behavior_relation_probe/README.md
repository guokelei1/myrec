# C69 Semantic-Null Behavior Relation Probe

C69 is a two-domain signal prerequisite, not the proposed architecture. It
tests whether an open-catalog Transformer can learn item-to-item behavioral
compatibility beyond ordinary language similarity when its negative pairs are
matched on semantic geometry.

The probe trains only on C47 fit histories, scores the already-open C47-A roles
without labels, and opens those existing labels only after all score/integrity
checks pass. Fresh reserve, dev, test, and qrels remain closed.
