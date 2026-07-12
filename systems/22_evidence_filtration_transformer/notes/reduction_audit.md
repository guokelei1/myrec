# C22 pre-implementation reduction audit

Decision: **provisionally pass for a minimal synthetic falsifier; global novelty
not established**.

## Rejected alternatives

1. Dynamic filtration bases were rejected because they are candidate-conditioned
   adapters/hypernetworks and return to C02/C15.
2. Final recurrence-cone projection was rejected because C18 already implements
   that safety locus and showed it cannot manufacture transfer.
3. Independent recurrence/semantic scorers were rejected as a multi-channel
   ensemble/router.
4. Equality-biased attention alone was rejected as relational attention or an
   induction-style retrieval head.
5. Shared LayerNorm was rejected because its denominator violates the claimed
   zero-Jacobian direction even when linear weights are triangular.

## Surviving distinction

The surviving hypothesis is a Transformer-wide **filtration-preserving causal
graph**: exact recurrence enters a reliable quotient, every hidden operation
preserves the evidence order, and semantic computation can read but never
overwrite that quotient.  This is functionally different from dense mixing,
parallel streams and a final safety projection at matched states.

The close StairFormer neighbour prevents a name-only novelty claim.  The only
reason to implement is falsifiability: if filtration does not improve the
repeat/non-repeat worst-stratum objective over all three controls, it has paid
no empirical rent and closes immediately.
