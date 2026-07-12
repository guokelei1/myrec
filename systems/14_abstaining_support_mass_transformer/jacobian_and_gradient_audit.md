# Jacobian factorization and gradient audit

## Factorized coordinates

Let allocation logits be `a`, `p=softmax(a)`, support logit be `g`, and
`rho=sigmoid(g)`.  For `w=rho p`,

```text
partial w / partial g = rho(1-rho) p,
partial w / partial a = rho [Diag(p) - p p^T].
```

The first derivative is radial: it changes total real mass.  The second is
tangent to the simplex because `1^T partial w/partial a=0`: it reallocates mass
without changing the total.  This is a useful diagnostic coordinate system, but
not a new Jacobian.

## Exact null-softmax coordinate map

Set

```text
l_NULL = 0,
l_j    = g + a_j - logsumexp(a).
```

Ordinary softmax of `l` produces `(1-rho,rho p)`.  Differentiating through this
map yields exactly the two Jacobians above.  Conversely, for arbitrary null
softmax logits,

```text
g = logsumexp(l_real) - l_NULL,
p = softmax(l_real).
```

Thus a “Jacobian factorization test” cannot distinguish C14 from a matched
NULL-attention layer: forward values, first derivatives, and all higher
derivatives through the smooth coordinate map agree.

Separate neural parameter heads for `g` and `a` change optimization coordinates,
not the function class.  A matched NULL control can use the same two heads and
the mapping above, making parameter-level gradients identical too.

## Abstention and starvation

As `rho->0`, both derivatives vanish:

```text
||partial w/partial g|| = O(rho),
||partial w/partial a|| = O(rho).
```

Allocation is also unidentifiable at exact zero.  A hard zero from ReLU,
hard-sigmoid, sparsemax, or entmax typically has a zero/subgradient region;
straight-through estimation would be an additional training trick.

If the residual is `lambda W_O o`, all attention-parameter gradients are also
multiplied by LayerScale `lambda`.  A small non-zero initialization avoids the
**exact** zero-gradient fault of `lambda=0`, but it does not cure sigmoid
saturation and can compound small `lambda*rho` gradients.  This is a sensible
identity-safe initialization control, not evidence-mechanism novelty.

## Required gradient canaries if revisited

- two optimizer steps must move support, allocation, `V`, `W_O`, and LayerScale;
- report gradient norms separately for radial and tangent coordinates;
- compare those norms pointwise with the transformed NULL-softmax control;
- test `rho` on a logarithmic ladder down to numerical zero;
- show that small non-zero LayerScale improves stability without being the sole
  source of any metric gain.

The exact matched control is predicted to pass every canary identically, which
is itself the pre-implementation stop condition.
