# C74 — Semantic-Conservative Query-Relay Transformer

C74 tests a two-hop Transformer attention primitive in which trainable maps
may route evidence and learn chronology, but history values and candidate
readout remain in the shared LM's original semantic coordinates.

The locked first stage is data-free.  A pass authorizes only a separately
frozen pretrained-LM probe; it does not authorize dev, test, qrels, or a paper
claim.
