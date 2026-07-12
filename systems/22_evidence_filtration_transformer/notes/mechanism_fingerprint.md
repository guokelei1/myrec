# C22 mechanism fingerprint

## Load-bearing fingerprint

`reliability-labelled input coordinates -> prefix-preserving normalization ->
one-way block-triangular attention/FFN at every layer -> monotone recurrence
readout + bounded centred transfer readout`.

Deleting any one of the following changes the hypothesis:

1. **filtration preservation**: dense mixing lets speculative history overwrite
   recurrence coordinates;
2. **one-way coupling**: block-diagonal streams prevent transfer computation
   from reading the protected recurrence state;
3. **relation placement**: moving equality into the transfer block reduces the
   reliable prefix to an empty architectural label;
4. **prefix normalization**: ordinary RMSNorm reintroduces transfer-to-prefix
   dependence through the shared norm denominator.

## Non-reduction witnesses

- A final score projection cannot reproduce an intermediate computation in
  which layer `l+1` transfer features depend on a layer-`l` recurrence feature
  while the reciprocal Jacobian is identically zero at every layer.
- Two independent parallel Transformers cannot reproduce that one-way
  conditioning without adding a cross-stream map, which is precisely the
  filtration edge.
- A dense Transformer can represent the same function by choosing structured
  weights, but it does not impose the zero-Jacobian invariant.  Dense mixing is
  therefore the principal matched supermodel control, not a proof of novelty.
- A candidate-conditioned change of basis would reduce to a hyperadapter/C02;
  C22 forbids it.  Filtration coordinates and masks are fixed across requests;
  only the identity relation input is request dependent.

## Mandatory Jacobian contract

At every layer and token:

```text
d(anchor_out) / d(recur_in, transfer_in) = 0
d(recur_out)  / d(transfer_in)           = 0
d(transfer_out) / d(recur_in)            may be nonzero.
```

The test suite must verify these with autograd, including normalization.  A
model that satisfies only parameter masks but violates the functional Jacobian
contract is not C22.
