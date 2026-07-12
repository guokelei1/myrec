# C32 pre-lock implementation review

Decision: accept for one untouched train-internal falsifier.

C32 is a one-operator continuation, not a hyperparameter rescue.  Compared with
C31, only `p -> (I-qq^T)p` is inserted before query normalization.  The unit
sphere tangent identity is directly tested; all candidates share the same
transported query and the only trainable tensors remain the two low-rank
matrices (16,384 parameters).  The frozen BGE Transformer remains load-bearing.

The selection chain is auditable: C31-A is open but excluded; C32-A inherits
C31 delayed-B, whose feature/score/label flags are false in both C31 G0 and
terminal reports.  C32 delayed-B inherits C31 escrow.  Selection is label-free,
roles are disjoint, and donors forbid same-user and recipient-candidate overlap.

The chief risk is that tangent projection is generic geometry rather than novel
by itself.  No novelty is claimed for orthogonal projection.  The falsifiable
architecture contribution is its use as the only admissible write from
strictly authenticated behavior into one shared LM query state, with exact
fallbacks and an unprojected matched reduction.  A1 must pass before controls or
the next cohort can open.
