# Minimal falsifier if C13 is ever reopened

Status: design only; no execution is authorized.

## S0: algebraic operator probe

Use fixed FP64 residual matrices before training:

1. candidate-common `R=1m^T` must map bitwise to zero;
2. rank-one and `C=2` inputs must match an explicitly computed scalar rescale;
3. the rank-two witness in `algebraic_and_constructive_audit.md` must differ
   from every scalar rescale and from post-head RMS;
4. candidate permutations must commute exactly within tolerance;
5. global write norm and zero sum must hold for random, duplicate, rank-deficient,
   all-zero, and maximum-magnitude inputs;
6. scale `X` by `10^-12 ... 10^6`; log output norm, Jacobian norm, and mode
   ratios for the one frozen epsilon.

Stop if the registered implementation behaves as a scalar in all rank profiles
actually produced by the Transformer.

## S1: synthetic signal-versus-noise probe

Freeze one ordinary cross-attention generator/checkpoint and expose identical
raw residuals to every control.  Generate four regimes without dataset-specific
model branches:

- clean low-rank candidate contrast aligned with a relevant candidate;
- pure candidate-common translation;
- weak aligned signal plus orthogonal small noise;
- wrong-user/adversarial high-rank incoherent noise matched in raw Frobenius norm.

Include exact-repeat, non-repeat, and no-history requests.  The primary must:

- beat ordinary centred attention, per-row LayerNorm, post-head RMS, scalar set
  RMS, and fixed/learned scalar rescale by a frozen minimal effect;
- protect the internal item-only repeat path;
- improve the clean clicked-minus-unclicked margin while wrong-user, shuffle,
  query-mask, and adversarial noise cannot reproduce it;
- retain the advantage over an epsilon ×0.1/×10 robustness audit without
  selecting epsilon from outcomes;
- change candidate order because of aligned signal, not merely maximize change
  rate;
- satisfy no-history bitwise identity, exact common-mode zero, and the global
  write bound.

Any wrong-user gain, high-rank-noise amplification, or tie with a scalar control
closes the candidate.

## A0: real label-free safety gate

Only after S0/S1, train on fit-only labels and freeze the checkpoint before
opening any internal-A outcomes.  A new cohort and candidate/base hashes are
required; dev/test/qrels remain blocked.  A0 performs no relevance evaluation:

1. assert base score/config/candidate hashes and no-history pointwise parity;
2. run max batch, all-no-history, all-repeat, all-non-repeat, duplicate
   candidates, empty corruption mask, checkpoint reload, and serialization;
3. verify candidate permutation, zero sum, bound, finite SVD/backward, and
   deterministic repeated execution;
4. compare clean versus matched wrong-user/shuffled/query-masked raw and
   normalized write norm, effective rank, Jacobian norm, and order-change rate;
5. require whitening not to increase the wrong/clean amplification ratio or the
   wrong/clean order-change ratio;
6. require near-zero real residuals to remain below a frozen safety ceiling,
   rather than becoming bound-sized writes;
7. confirm by static dataflow audit that the normalizer never reads base scores,
   labels, dataset IDs, or query classes.

A0 stops before labels if normalization equalizes clean and wrong-user energy,
if near-zero writes jump discontinuously, if rank is almost always one (scalar
reduction), or if numerical/determinism contracts fail.

## Why A0 cannot validate the positive claim

Order-change and write norm are label-free safety diagnostics, not evidence that
the new order is better.  A later separately frozen train-internal label gate
would still have to beat all controls.  C13 is rejected before reaching A0
because the paper audit already exposes the normalization/noise dilemma and weak
novelty.
