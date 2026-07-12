# C11 reduction and matched-control audit

Status: pre-outcome.

## Pooled C10

`pooled_c10` averages event contexts before decoding candidate tokens, computes
one `log p(x|q,pool(H))-log p(x|q)` vector, and repeats it through the identical
late integrator.  Parameter count, initialization, optimizer, token embeddings,
decoder, integrator, write, and head are exact matches.  The difference is
solely whether candidate-token evidence exists separately for every event.

C11 is not algebraically this control.  The checked witness holds mean event
gain fixed while changing its distribution across events; `tanh` is applied
before event integration, so eventwise innovations differ.  Learned positional
integration can additionally distinguish chronology.  Reduction can still
occur empirically if the late integrator learns uniform pooling; that is a
collapse diagnostic, not an identity.

## Ordinary centred attention

`centered_attention` uses the standard scaled dot product between base candidate
state and contextual event values, applies a masked event softmax, and candidate
centres the resulting event contributions.  To rule out a capacity objection,
those contributions pass through the same late integrator and write.  It is
therefore a capacity-strengthened ordinary attention control.

Predictive gain cannot in general reduce to that dot product: two event states
may have the same candidate-state dot product but different vocabulary
normalizers or different logits on a candidate token, hence different
`log p_e(x)-log p_0(x)`.  Conversely, attention may react to a value direction
without improving candidate-token prediction.  The frozen comparison decides
which inductive bias is learnable.

## Scalar final-logit delta

`scalar_logit` averages the same eventwise log-ratio over event and token,
candidate centres it, bounds it, and adds it after the hidden-state head.  It is
the explicitly allowed nearest-neighbour control, not a proposed architecture.
C11 retains token directions and event chronology before the head; equal scalar
sums can yield different vector writes.

## Same-capacity eventwise hidden control

`eventwise_hidden` replaces predictive likelihood ratio with
`<E[x_i,t], F(q,h_e)-F(q)>/sqrt(d)` while keeping the complete event matrix and
every other operation identical.  It tests whether any eventwise dual-stream
interaction suffices.  The primary and this control each have 29,872 trainable
parameters, as do pooled-C10 and centred attention; unit tests assert equality.

## Stop/collapse conditions

- Failure against any matched control closes the predictive-gain attribution.
- A base above the frozen 0.80 ceiling invalidates the generator before a
  positive mechanism claim.
- Low event-gain variance or fewer than 10% changed transfer rankings closes the
  conditional-interaction claim.
- Repeat degradation, non-positive clean transfer, or corruption retention over
  its frozen limit closes C11 without a real gate.
- No extra epoch, threshold change, new event feature, or dataset-specific branch
  may repair the same outcome.
