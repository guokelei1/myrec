# Mechanism fingerprint

| Field | C23 RRST |
|---|---|
| operator | candidate identity induces a last-occurrence reset boundary; a causal Transformer evolves the reset token only through the post-anchor suffix |
| intervention | history-to-candidate attention graph and positional coordinate, not an output router |
| state | candidate-local reset state plus suffix event tokens and query-candidate read token |
| training signal | listwise click loss on full candidate sets, starting from the registered item-only ordering |
| exact zero | no exact anchor in stage A gives zero learned write and exact D2p output |
| protected input | registered C5-R3 item-only score on repeat-present requests |
| degeneration | remove reset mask → ordinary target-aware history Transformer; remove order → suffix set encoder; remove query → query-independent recurrence calibrator |
| unique falsifier | post-anchor shuffle removes utility while pre-anchor replacement is exactly invisible |

The fingerprint is not a DeltaNet recurrence.  No fast-weight matrix is
updated over the complete history, and no event performs a generic
`M <- M + beta(v-M)` write.  Identity changes which edges exist in the
Transformer graph.
