# C49 reduction audit

- With `p_t=0`, C49 is raw-value KRR/Cubit; that is a binding same-checkpoint
  control.
- Replacing the normal equation by normalized positive weights is ordinary
  innovation-value attention; that is a binding control.
- An online error-corrected fast-weight write is DeltaNet; the exact beta=1
  delta read is a binding control, and a later trainable gate would require
  Gated DeltaNet.
- A causal predictor plus error-valued memory is close to DeltaNet, Titans,
  test-time memory, and classical least-squares associative memory.  Neither
  the predictor, KRR, nor error values are individually novel.

The only potentially distinct mechanism is the closed-form, request-local
least-squares read of *prequential semantic innovations* under separate
semantic keys.  This is a composition-level claim with high novelty risk.  It
must pay direct empirical rent over every reduction before fresh data.

Verdict: `distinct-composition-with-high-uncertainty`.
