# Novelty, epsilon, and wrong-user risk audit

## Prior-art boundary

Normalization tailored to permutation-invariant/equivariant set networks and
equivariant residual placement is already explicit in
[Set Norm and Equivariant Skip Connections](https://proceedings.mlr.press/v162/zhang22ac.html).
Activation whitening, inverse-square-root covariance transforms, and iterative
approximations are established in
[Iterative Normalization](https://arxiv.org/abs/1904.03441).  The latter also
documents that small-eigenvalue directions and noisy statistics make the extent
of whitening a stability/generalization trade-off.

The specific placement at a personalized-ranker history write is an application
choice.  Candidate centring is a valuable structural contract, but “set norm or
whiten a residual before adding it” is not by itself a sufficiently new
candidate-conditioned evidence primitive for the proposed-system standard.

## Epsilon dilemma

For singular value `s`, whitening applies multiplicative gain

```text
a_epsilon(s) = 1 / sqrt(s^2/C + epsilon).
```

This gain is strictly larger for weaker modes.  Therefore whitening always
amplifies small modes **relative to** strong modes.

- `epsilon -> 0`: every non-zero output singular value tends to `sqrt(C)`.
  Arbitrarily weak numerical/noise modes receive the same pre-bound magnitude
  as the strongest signal mode.  At exact zero the output is zero, but the
  limit depends on approach direction; without epsilon the operator is
  discontinuous/undefined at collapse.
- large `epsilon`: the transform becomes `X/sqrt(epsilon)`, a fixed scalar
  rescale with no whitening novelty.
- intermediate `epsilon`: defines an absolute hidden-scale threshold.  Picking
  it from outcome or dataset statistics is architecture tuning; learning it is
  effectively a global reliability/scale gate, which C13 forbids.

The final Frobenius bound prevents explosion but does not repair direction.  It
only redistributes a fixed energy budget over whatever modes whitening retained.

## Wrong-user and corruption risk

A clean residual may be low-rank: one strong candidate contrast plus small
noise.  A wrong-user or shuffled history can produce several weak, incoherent
contrast modes.  Whitening flattens both spectra.  Under the common final norm
bound, high-rank wrong-user noise may receive energy in more directions than the
low-rank clean signal.  Because no reliability gate or base-score comparison is
allowed, C13 cannot tell these cases apart.

This is not merely a theoretical corner case.  The motivation says history
evidence is uneven and corruption controls are mandatory.  Turning a `1.83e-5`
write into a fixed-size write assumes that its direction is useful—the very fact
that zero order changed does not establish.  The operator can solve amplitude
collapse only by discarding amplitude as evidence, which is unsafe when a tiny
amplitude may correctly express uncertainty.

## Scale/initialization audit

Scalar RMS is precisely an adaptive scale correction and must be classified as
a normalization/initialization intervention, not architecture innovation.
Whitening is invariant to common positive scaling when epsilon is negligible,
so it changes conditioning rather than merely initialization.  But that
conditioning is generic and pays no evidence-fidelity rent.  A fixed scalar
matched to the primary's median clean write norm, and a trainable global scalar
with the same parameter count, are mandatory controls.

If either matches the primary within the minimal claimable effect, the entire
claim reduces to scale calibration.  If only aggressive whitening wins while
wrong-user noise is amplified, the mechanism fails even if clean synthetic
NDCG rises.
