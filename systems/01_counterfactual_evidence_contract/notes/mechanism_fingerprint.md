# C01 Mechanism Fingerprint

Status: frozen before any C01 outcome.

## State and operator

For request query `q`, candidate `c`, and ordered history events
`H=(e_1,...,e_L)`, let frozen local text states be `x_q`, `x_c`, and `x_i`.
Candidate/event metadata supply event type, reverse-age position, deepest
category, exact-item relation `r_i`, and category relation.  The trainable
sequence is

```text
z = [W_q x_q + s_q,
     W_c x_c + s_c,
     W_e x_1 + m_1(c), ..., W_e x_L + m_L(c)]
h = T_theta(z, mask)
```

where `T_theta` is a two-layer Transformer encoder.  The contextual event state
`h_i` directly produces an energy `a_i=w_a^T h_i` and value
`v_i=tanh(w_v^T h_i)`.

For each observational training instance `o`, the same `T_theta`, `w_a`, and
`w_v` encode four counterfactual twins `k`:

- a batch-rotated different-user history;
- a deterministic non-identity event-order permutation;
- a masked-query token;
- a coarse-only twin in which item text/identity is removed while category
  information remains.

On a clicked candidate without exact recurrence, define the robust sequence
margin

```text
A(o)   = tau_lse * logsumexp_i(a_i(o) / tau_lse)
M(o)   = A(o) - max_k A(k)
L_cf   = relu(mu_cf - M(o)).
```

The ranking loss and `L_cf` jointly train the same internal states.  This is a
multiple-instance contract: it does not assert that every historical event is
useful.

## Train-only quantile and inference certificate

After fitting the encoder, its weights and energy head are frozen.  On a
chronologically later **train-only calibration slice**, collect non-exact event
energies from all four counterfactual twins.  With false-admission target
`alpha_cf=0.10`, use the finite-sample upper quantile

```text
Q_cf = sorted(a_cf)[ceil((n + 1) * (1 - alpha_cf)) - 1].
```

The inference gate is

```text
g_i = 1                                      if r_i = exact,
      sigmoid((a_i - Q_cf) / tau_gate)       otherwise.
```

For diagnostics, the hard admission is `a_i > Q_cf`.  Only true `(q,c,H)` is
needed at inference.  Twins are training/diagnostic interventions, never online
features.

## Contract score and logit

The exact atom has lower-bound value
`3 * event_weight / sqrt(reverse_age)`, matching the audited C5-R3 component.
It is injected before the same contract aggregation, not used as an independent
candidate scorer.  The contracted score is

```text
E(c,H) = sum_i r_i * exact_floor_i
       + lambda_tr * sum_i (1-r_i) * g_i * softplus(v_i).
```

Within a history-present request, the final logit is the preregistered static
anchor form

```text
s(c) = beta * z(D2p(c)) + (1-beta) * z(E(c,H)),  beta=0.30,
```

unless no event is admitted for any candidate, in which case the evidence
contract is empty and scores fall back to D2p.  For `history_present=false`,
`s(c)=D2p(c)` exactly.  The Transformer is load-bearing because all non-exact
personalized ranking information and its admission decision pass through its
contextual event states; the D2p anchor alone cannot produce that residual.

## Collapse controls

- Counterfactual quantile calibration fixes the null false-admission target and
  prevents the threshold from drifting to admit all twins.
- Robust max-over-twins margin prevents success against one easy corruption
  from masking failure against another.
- Exact atoms remain admitted even if transfer is rejected everywhere.
- Frozen admission-rate, variance, and true-vs-twin gap checks detect all-zero,
  all-one, and near-constant certificates.
- The value/readout stage is trained only after the energy encoder and `Q_cf`
  are frozen, preventing post-calibration threshold invalidation.

## Degenerate and matched forms

- **Plain target-attention control:** identical input projection, Transformer,
  heads, exact relation access, train split, optimizer steps, and parameter
  count; it replaces the quantile contract with ordinary softmax weights over
  event energies and receives no twin loss.  Unused calibration scalars are
  retained as trainable dummy parameters to make exact parameter counts equal.
- **DIN degeneration:** remove query-conditioned event-event contextualization,
  replace `T_theta` states by independent candidate-event local activation, and
  normalize/use every event without a counterfactual null.
- **TEM/RTM degeneration:** remove `Q_cf` and twin margins and read the query or
  sequence state directly as the score.
- **CARD degeneration:** replace multi-twin energy calibration by
  leave-one-event-out future prediction-error reduction.
- **CFT degeneration:** replace the event certificate by one paired
  history/no-history output-logit difference.

## Complexity and inference inputs

- Frozen text backbone: BAAI/bge-small-zh-v1.5 states reused locally.
- Trainable core: `d_model=96`, 2 Transformer layers, 4 heads, FFN 192, history
  length at most 20; the exact parameter count is recorded by the run.
- Time per candidate: `O((L+2)^2 d)` for `L<=20`.
- Inference inputs: query, candidate, true strictly-prior history, masks, and the
  frozen D2p anchor.  No donor history, qrels, external API, or online LLM.

Fingerprint verdict before outcome: **mechanistically distinct, novelty still
uncertain at field-wide scope**.  The closest discovered works are CARD and CFT;
their non-equivalence and matched degenerations are recorded in
`nearest_neighbors.md`.
