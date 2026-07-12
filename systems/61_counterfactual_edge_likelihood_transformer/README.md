# C61 — Counterfactual edge-likelihood Transformer

C61 keeps C60's safe base-margin transport but replaces its fixed cosine
evidence with a trainable Transformer likelihood ratio for one question:
does the user's history overturn this adjacent base edge?  The same edge
network is evaluated with factual and NULL history, antisymmetrized under
candidate swap, and subtracted before it can open a bounded transport edge.

Training uses the already exposed C26 fit labels.  The 1,200-request C26
internal-A role is frozen as fresh evaluation and remains label-closed through
G0, contextual-token materialization, training, and label-free A0.
