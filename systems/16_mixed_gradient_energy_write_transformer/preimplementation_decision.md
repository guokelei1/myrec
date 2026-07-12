# C16 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

The mixed-gradient/energy candidate family fails the design gate before outcome
access:

1. the gradient of a bilinear/log-sum-exp candidate--history energy is exactly
   weight-tied cross-attention and the modern-Hopfield retrieval update;
2. a nonlinear conservative update remains a scalar-energy gradient, the
   mechanism family already represented by Energy Transformer and
   Hopfield--Fenchel--Young networks;
3. candidate-axis normalization supplies competition but is the Slot Attention
   allocation primitive, while an ET energy restricted to candidate--history
   edges is already the direct bipartite construction;
4. contracting a mixed Hessian with a candidate-independent event direction is
   exactly the gradient of a contracted scalar potential;
5. allowing candidate-dependent contractions either preserves integrability
   and hence some scalar-potential representation, or breaks integrability and
   forfeits the energy-descent claim; and
6. `softmax(s)-uniform` remains a scalar energy/regularizer choice inside the
   HFY envelope and is exactly `softmax(s)-softmax(0)`, covered by the signed-
   map construction of Differential Transformer and directly targeted by
   ZeroS.

The family therefore has no architecture fingerprint that is simultaneously
conservative, non-reducible, and distinct from established attention/energy
mechanisms.  Combining a known energy gradient, a known competition axis, and a
known uniform subtraction is composition, not one new falsifiable primitive.

No source model, runner, config, lock manifest, synthetic probe, real gate, GPU
run, checkpoint, data read, dev evaluation, test evaluation, qrels access, or
label access is authorized or present.  This file is the binding terminal
decision for C16.

A successor would first need a motivation-derived candidate/event vector law
that has a concrete witness separating it from tied retrieval, scalar-energy
descent, slot competition, centred/differential attention, and a generic
pairwise MLP.  Renaming a mixed derivative or changing a normalization axis is
not sufficient.
