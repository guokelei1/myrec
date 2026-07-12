# C20 proposal

## Observation carried from C19

C19 could learn recurrence and a successor task, but its oriented scalar
cofactor beat the best structured control in only one seed and retained almost
all corrupted gain in another.  Endpoint affinities answer “do these tokens
match?”; they do not answer the stronger question “is this particular
query-to-candidate change explainable by the user's observed directed
changes?”.

## Single primitive

A shared lower Transformer produces query state `q`, candidate states `c_i`,
and ordered history states `h_j`.  In one learned relation space define

```text
r_i = W_r(c_i - q)
d_j = W_d(h_j+1 - h_j)
D   = [d_1 ... d_H-1].
```

For every candidate, HTCT executes a fixed number of projected-gradient steps
for the ridge nonnegative least-squares problem

```text
alpha_i* = argmin_(alpha >= 0)
           1/2 ||r_i - D alpha||^2 + lambda/2 ||alpha||^2
p_i      = D alpha_i*.
```

`p_i` is the portion of the query-to-candidate displacement reconstructible as
a positive composition of observed chronological transitions.  The candidate
token is changed only through this vector:

```text
u_i  = W_p p_i - mean_candidates(W_p p)
c'_i = c_i + LayerScale * u_i
score_i = shared_rank_head(upper_Transformer(c'_i)).
```

There is no evidence bonus at the score interface.  Empty or single-event
history has no transition columns and selects the untouched base score bitwise.
Exact identity is an observed input coordinate in the base Transformer path;
it is not synthesized by the cone solver.

## Why nonnegative composition is the hypothesis

An unconstrained subspace can reconstruct both `r` and `-r` from one observed
transition by changing the coefficient sign.  A convex cone cannot: observing
`a→b` supports a positive composition containing that direction, not its
reverse.  With nonorthogonal transitions, the NNLS coefficients are jointly
coupled by `D^T D`; they are not independent ReLU similarities or a softmax
over events.

## Falsification

The primitive is rejected if it cannot simultaneously:

1. keep a history-blind candidate set exchangeable at supported labels;
2. recover positive compositions while rejecting reverse directions;
3. beat an unconstrained span, one-step ReLU, simplex retrieval and matched
   pooled-transition MLP under identical Transformer capacity;
4. preserve repeat and bit-exact no-history behavior;
5. lose its clean margin under wrong, shuffled, query-masked, coarse-only and
   reversed histories; and
6. remain candidate-permutation equivariant with finite gradients through the
   unrolled solver.

Synthetic passage would establish only a trainable, non-degenerate inductive
bias.  It would not establish real PPS utility or global novelty.
