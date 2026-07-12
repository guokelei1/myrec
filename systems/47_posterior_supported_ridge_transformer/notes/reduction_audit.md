# C47 algebraic reduction audit

## Exact reductions

Let `P=H^T(HH^T+lambda I)^-1H`, `d_c=c^TPq`, and
`rho_c=c^TPc`. C47's diagnostic coordinate is `rho_c d_c`.

1. Setting `rho_c=1` reduces exactly to plain KRR/Cubit-style mixing.
2. Replacing `P` by a normalized rank-one history mean reduces to a scalar
   target-attention/profile gate.
3. Treating `rho_c` as a free learned number reduces to a candidate-local
   diagonal gate. This is a binding control, not a claimed distinction by name.
4. A linear-kernel Gaussian process exposes the same posterior mean geometry
   and predictive support. GP uncertainty is not novel.

## Narrow surviving distinction

C47 ties the contraction and the mean write to the *same* empirical history
operator. A free gate may approximate or dominate it, but is not algebraically
forced to obey:

- `Hc=0 => rho_c d_c=0`;
- `0<=rho_c<1`;
- history-basis invariance (`H` may be left-multiplied by any orthogonal event
  reparameterization without changing `P`);
- duplicate aligned evidence increases support while never amplifying beyond
  the plain ridge write.

These properties establish a falsifiable inductive bias, not an expressivity
separation from a general Transformer or MLP. If a matched free gate or Cubit
matches C47, the primitive has paid no empirical rent and closes.

## Novelty verdict

`distinct-with-high-uncertainty`. The posterior-supported contraction appears
narrower than the reviewed KRR/GP token mixers, but its ingredients are known
and its result is a candidate-local scalar contraction. Paper-level novelty is
not authorized before matched-control and broader literature review.
