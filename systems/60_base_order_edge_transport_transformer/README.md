# C60 — Base-order edge-transport Transformer

C60 tests a different residual-write contract after C59: frozen Transformer
history evidence no longer contributes a candidate score.  It only opens
one-sided transport edges between adjacent slots of the strong-base ordering;
each edge can transfer at most that base margin.  The update is conservative,
local at every rank, and exactly zero without evidence.

The first formulation gate reuses C59's already exposed 1,200-request role and
therefore cannot support a fresh result.  A positive result only authorizes a
new-role trainable Transformer; a negative result closes this fixed edge law.
