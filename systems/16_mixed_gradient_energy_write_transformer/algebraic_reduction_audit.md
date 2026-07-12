# Algebraic reduction audit

## 1. Linear candidate gradients are tied cross-attention

Let

```text
q_i = B z_i,
k_j = C h_j,
s_ij = q_i^T k_j,
p_ij = softmax_j(s_ij / tau).
```

For the smooth retrieval potential

```text
Phi_i(z_i;H) = tau log sum_j exp(s_ij / tau),
```

the candidate gradient is exactly

```text
grad_{z_i} Phi_i = B^T sum_j p_ij k_j
                   = sum_j p_ij (B^T k_j).
```

This is cross-attention with query `B z_i`, keys `k_j`, and values
`v_j=B^T k_j`.  The value/output map is tied to the derivative of the score
map.  More generally, for a bilinear pair energy
`epsilon(z_i,h_j)=z_i^T A h_j`, its value direction is `A h_j`; no additional
event information appears beyond an attention-weighted retrieval.  Choosing
the sign of `Phi`, adding a step size, or appending a quadratic state term only
changes descent direction or residual damping, not the primitive.

This is the continuous modern-Hopfield update written in Transformer notation.
Untying a free `W_V` makes the module ordinary cross-attention rather than a new
energy-derived write.

## 2. Nonlinear conservative values remain scalar-energy retrieval

Replace the bilinear score by an arbitrary differentiable pair energy
`epsilon_theta(z_i,h_j,q)` and define

```text
Phi_i = tau log sum_j exp(epsilon_theta(z_i,h_j,q) / tau).
```

Then

```text
grad_{z_i} Phi_i
  = sum_j p_ij grad_{z_i} epsilon_theta(z_i,h_j,q).
```

Putting a nonlinear value inside the event sum therefore does not escape the
energy family when the value is constrained to be the score gradient.  It is an
energy-minimizing attention/associative-memory update.  Energy Transformer
already engineers token energies and follows their gradients; Hopfield--
Fenchel--Young networks generalize the retrieval energy and regularized
transformation.  An unrestricted neural scalar `epsilon_theta` adds generic
potential capacity but no PPS-specific structural law.

## 3. A mixed Hessian has only two outcomes

Consider the proposed contraction

```text
F_a(z,h) = sum_b [partial^2 Psi(z,h) / partial z_a partial h_b] u_b(h),
```

where the contracted event direction `u(h)` is independent of `z`.  Define the
scalar

```text
chi(z,h) = sum_b [partial Psi(z,h) / partial h_b] u_b(h).
```

Direct differentiation gives

```text
F(z,h) = grad_z chi(z,h).
```

Thus the apparently second-order vector write is exactly a first-order
gradient of a contracted scalar potential.  Summing this identity over events
or applying request-level coefficients independent of `z` does not change it.

If `u=u(z,h)`, differentiating `chi` also produces

```text
sum_b [partial Psi / partial h_b] grad_z u_b,
```

which the bare mixed-Hessian contraction omits.  Its candidate-state Jacobian
is then not generally symmetric.  On a simply connected state domain, the
integrability condition for `F` to be conservative is

```text
partial F_a / partial z_c = partial F_c / partial z_a  for every a,c.
```

If this condition holds, the Poincare lemma gives some scalar potential and the
proposal returns to scalar-energy descent.  If it does not hold, the update is
a non-conservative pairwise vector field and cannot claim monotone energy
descent.  Symmetrizing the Jacobian to restore integrability again restores the
scalar-potential reduction.

## 4. Candidate-axis competition is an axis change, not a new energy law

A joint candidate/event potential can use

```text
Phi(Z,H) = tau sum_j log sum_i exp(epsilon(z_i,h_j) / tau).
```

Its candidate gradient includes `p(i|j)`, so candidates compete to explain each
event.  This is precisely the allocation-side idea of Slot Attention: inputs
normalize attention over slots, after which slots aggregate their assigned
inputs.  With candidates as slots and events as inputs, changing the softmax
axis does not establish a new ranking primitive.

A scalar Energy Transformer potential restricted to edges between two token
partitions yields the same bipartite candidate--history coupling.  “Bipartite
ET” here names that direct restriction; it is not asserted to be a separate
paper or architecture title.

## 5. Softmax-uniform centring is already signed attention

For `H` events,

```text
p_j - 1/H = softmax(s)_j - softmax(0)_j.
```

Therefore centred attention is an exact fixed-second-map instance of
Differential Transformer.  The same field is the gradient of the scalar
difference

```text
tau log sum_j exp(s_j/tau) - (1/H) sum_j s_j.
```

ZeroS directly removes the uniform zero-order softmax component and reweights
the remaining zero-sum residual.  HFY energies also already expose the choice
of regularizer/transformation.  Calling the centred term an “energy contrast”
does not distinguish it from these mechanisms.

## Binding consequence

Every conservative C16 branch reduces to a known scalar-energy or tied-
attention update.  The only algebraic escape abandons conservativity and becomes
a generic pairwise vector write, which has no surviving structural insight.
This is a pre-outcome novelty failure, so implementation and experimentation
are not authorized.
