# C17 mechanism fingerprint

## Intended primitive

For candidate state `x_i^l` and history events `h_j`, C17 proposed a persistent
ledger `L_ij^l` rather than an immediately pooled history vector:

```text
x_i^(l+1) = TransformerBlock_l(x_i^l, ...)
L_ij^(l+1) = LedgerBlock_l(L_ij^l, x_i^l, h_j, ...)
Delta s_i = Readout_i({L_ij^L}_j) - mean_k Readout_k({L_kj^L}_j).
```

The motivation-derived hope was that event identity would remain visible until
the final candidate-relative score, preventing the candidate-common saturation
seen in C02/C05 and the vanishing final flow seen in C06.

## Required witness

Implementation would have required one construction satisfying all of these:

1. the ledger state cannot be reinterpreted as a vector state on a
   candidate--event edge;
2. its layer update is not generic edge-conditioned message passing or
   triangular edge attention;
3. it changes the ranker's forward function, rather than only decomposing an
   existing forward into token attributions;
4. the score readout is not an elementwise/eventwise gate on ordinary
   attention or attribution values;
5. its candidate-relative readout is not the divergence of an antisymmetric
   flow, nor an ordinary pairwise score head; and
6. histories with the same ordinary pooled values but different ledgers lead
   to a difference that a matched edge-state Transformer cannot reproduce.

No natural branch satisfies all six.  The last requirement is impossible for
a universal learned edge-state control, while imposing exact chain-rule tying
to escape that control removes functional freedom and turns the ledger into an
explanation of the original forward.

## Boundary

No-history identity, exact-recurrence monotonicity, candidate centring, and
corruption tests remain valid safety contracts.  They do not make the ledger a
new primitive because they can be attached unchanged to each reduced form.
