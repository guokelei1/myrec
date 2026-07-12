# C07 Proposal: Pairwise Dead-zone Signed-Kernel Transformer

Date: 2026-07-11
Stage: formulation + structural CPU prototype
Authorization after this document: candidate-local synthetic CPU probe only

## 1. One-sentence design hypothesis

**Observation.** History can alter ranking only through candidate-relative
evidence, and weak cross-item differences should be allowed to produce exactly
no history update while strong exact or jointly supported evidence should
redistribute score mass between candidates.

**Architecture consequence.** Replace the candidate-to-history softmax in an
end-to-end ranking Transformer with one primitive: pairwise dead-zone signed
normalization over candidates for every history event.

**Cheap falsification.** On a label-free synthetic contract with at least three
candidates, the primitive must beat both its linear-centered degeneration and a
scalar-gated centered-softmax control on supported non-repeat patterns, while
remaining exactly inactive for no-history/common-mode/dead-zone patterns; if it
does not, C07 stops before any real-data or GPU work.

This is a hypothesis, not an established empirical insight.

## 2. Architecture and information flow

For request query representation \(q\), history event representations
\(h_j\), and jointly ranked candidate representations \(c_i\), a shared
Transformer first contextualizes the sequence

\[
  [q, h_1,\ldots,h_H,c_1,\ldots,c_C].
\]

Its structural attention mask permits query-to-query, history-to-query/history,
and candidate-to-query/candidate paths, but blocks ordinary
history-to-candidate attention.  Thus the new kernel is the sole path by which
history can alter candidate states.  The same Transformer weights process all
token types; this is not an embedding-plus-MLP scorer or an offline feature.

One head forms a query/history/candidate logit

\[
  s_{ij} = {\langle W_c c_i \odot \tanh(W_q q), W_h h_j\rangle\over\sqrt d}
           + \beta\,e_{ij}\,
             \operatorname{softplus}
             \left({\langle W_c c_i,\tanh(W_q q)\rangle\over\sqrt d}\right),
\]

where \(e_{ij}\) is an available exact-item identity-match feature and
\(\beta>0\).  The positive query/candidate compatibility factor prevents the
identity path from becoming a query-blind additive shortcut.  A production LM
may realize the identity feature through its
item-token channel; the explicit tensor in the prototype makes the contract
auditable.  Cross-item evidence has no identity term and must come from the
learned tri-linear interaction.

The exact-match term therefore does not bypass the query, and the complete
logit remains the input to candidate-set competition.  A query-masked
corruption remains a mandatory negative control in the synthetic gate.

## 3. The primitive

For each history event \(j\), compute all ordered candidate margins

\[
  d_{ikj}=s_{ij}-s_{kj}.
\]

Apply an odd soft-threshold with fixed positive dead-zone width \(\tau\):

\[
  \rho_\tau(x)=\operatorname{sign}(x)\max(|x|-\tau,0)
              =[x-\tau]_+-[-x-\tau]_+.
\]

The candidate's signed balance for that history event is

\[
  u_{ij}={1\over C-1}\sum_{k\ne i}\rho_\tau(d_{ikj}).
\]

Finally use a null-mass-stabilized absolute normalization over all valid
candidate/history cells:

\[
  a_{ij}={u_{ij}\over \kappa+\sum_{r,t}|u_{rt}|},\qquad \kappa>0,
\]

and inject history values into candidate states:

\[
  \Delta c_i=W_o\sum_j a_{ij}W_vh_j,
  \quad \tilde c_i=\operatorname{LN}(c_i+\Delta c_i),
  \quad \ell_i=w^\top\tilde c_i.
\]

The Transformer/LM state \(\tilde c_i\), not a fixed-score router, is the
ranking core.  Multiple heads use the same equation with head-local projections;
the minimal prototype uses one signed kernel after a conventional shared
Transformer layer.

## 4. Structural properties

### Candidate conservation

Oddness gives, separately for every history event,

\[
  \sum_i u_{ij}=0\quad\text{and}\quad\sum_i a_{ij}=0.
\]

Every positive history allocation is paid for by a negative allocation to
competitors.  A history value cannot enter as the same additive vector for all
candidates through this branch.

### Common-mode invariance

For any event-wise scalar \(b_j\), replacing \(s_{ij}\) by \(s_{ij}+b_j\)
leaves all \(d_{ikj}\), hence all outputs, unchanged.  This directly removes
the rank-inert common translation identified in the design brief.

### Structural abstention

If \(\max_i s_{ij}-\min_i s_{ij}\le\tau\), then all pairwise margins for event
\(j\) lie in the dead zone and \(a_{ij}=0\) for every candidate.  This is an
open set of unequal logits, not the measure-zero equality case of centered
softmax.  If every event abstains, history contributes exactly zero.

### No-history identity

Masked events are removed before pairwise aggregation.  With no valid history,
\(\Delta c_i=0\) exactly and the final logits equal the query/candidate base
logits pointwise, not merely in rank.

### Bounded magnitude

\[
  \sum_{i,j}|a_{ij}| < 1
\]

because \(\kappa>0\).  This limits the signed branch before its learned value
and output projections; it is not a claim of whole-network Lipschitzness.

### Set equivariance

Permuting candidate tokens and the corresponding identity rows permutes the
outputs.  Permuting history event/value/mask tuples leaves candidate logits
unchanged.  Temporal order, when needed, is carried inside each history
representation, not inferred from input row order by the kernel.

## 5. Mandatory mathematical reduction audit

The first attempted construction was simply a centered signed normalizer,
\(a_i\propto s_i-\bar s\).  It was rejected before implementation because it
is centered linear attention.

The surviving construction has a sharp reduction boundary:

1. **If \(\tau=0\), it collapses.** Since \(\rho_0(x)=x\),

   \[
     u_{ij}={C\over C-1}(s_{ij}-\bar s_{\cdot j}),
   \]

   exactly a centered first-order attention logit followed by L1 scaling.  This
   is a mandatory degeneration control, not C07.

2. **With two candidates it is not identifiable.** Every zero-sum two-vector
   is one-dimensional.  Novelty and mechanism probes therefore require
   \(C\ge3\).

3. **For \(C\ge3,\tau>0\), the pairwise map is not an elementwise centered
   activation.** Suppose there were a scalar \(g\) such that

   \[
     u_i=g(s_i)-C^{-1}\sum_r g(s_r)
   \]

   for all score vectors.  Away from threshold kinks, for \(k\ne i\), the
   left cross-derivative is

   \[
     {\partial u_i\over\partial s_k}
       =-{\rho_\tau'(s_i-s_k)\over C-1},
   \]

   which depends on \(s_i-s_k\); the proposed right side would be
   \(-g'(s_k)/C\), independent of \(s_i\).  Equality for all inputs forces
   \(\rho'\) to be constant, hence an affine \(\rho\), contradicting the
   positive dead zone.

4. **It is not ordinary fixed-temperature centered softmax.** Centered softmax
   is nonzero for every unequal finite-logit vector, while C07 is identically
   zero on the open region \(\max s-\min s\le\tau\).  A scalar dead-zone gate
   can reproduce that zero region but cannot change the active attention
   direction.  For \(s_\alpha=(2\alpha,0,-\alpha)\) and fixed \(\tau>0\), the
   C07 direction tends to \((5,-1,-4)\) as \(\alpha\to\infty\); any finite,
   fixed-temperature centered-softmax direction tends to \((2,-1,-1)\).
   These are not collinear.  Likewise, for
   \(s=(2,0,-1),\tau=0.5\), C07 has
   \(u=(2,-0.5,-1.5)\), which is not collinear with the centered-linear
   direction \((5/3,-1/3,-4/3)\).

   A request-conditioned temperature or unrestricted vector gate is a more
   expressive control and is not excluded by this proof.  The synthetic gate
   therefore includes a request-conditioned scalar amplitude/temperature
   control; if it matches C07, the mechanism stops.

5. **This is still a high-risk distinction.** A sufficiently expressive
   vector-valued gate or downstream Transformer may approximate the same
   function with ordinary attention and MLP layers.  The claim is only an
   inductive-bias/normalization claim, never an expressivity impossibility claim
   against generic gating or full Transformers.

## 6. What this is not

- not graph/Hodge/divergence reasoning;
- not transport, Sinkhorn, or a doubly stochastic assignment;
- not a target-attention rename: ordinary target attention normalizes history
  independently for each target, whereas C07 couples all candidates for each
  history event and permits negative conserved weights;
- not a fixed-score expert router or static mixture;
- not a dataset/category/query-type rule;
- not a hyperadapter, prefix, prompt-only scorer, or offline LLM feature;
- not a claim that unsupported semantic transfer has been solved.

## 7. Complexity and implementation risk

The literal prototype materializes \(C\times C\times H\) margins, requiring
\(O(HC^2)\) work and memory per head.  This is acceptable only for a tiny CPU
falsifier.  Because soft-thresholded pairwise sums can be computed from sorted
scores and prefix sums, an eventual optimized kernel could reduce arithmetic to
\(O(HC\log C)\); no such kernel or efficiency claim exists yet.

The hard dead zone also gives zero gradient inside inactive pairs.  That is the
mechanism's abstention property and its largest optimization risk.  The
synthetic gate must measure active-pair and gradient coverage.  A post-outcome
switch to a smooth nonzero surrogate would change the primitive and require a
new lock.

Other risks are threshold-scale sensitivity, candidate-count dependence,
signed-value cancellation across history events, an exact-match shortcut that
never learns supported transfer, and close prior art on every ingredient.  The
direction advances only if the non-factorizable matched control fails where C07
succeeds.

## 8. Current decision

The algebra survives the mandatory centered-attention reduction review for
\(C\ge3\) and \(\tau>0\), and the structural CPU prototype passes its tests.
That is sufficient to justify **one candidate-local, CPU-only synthetic probe**
under `pre_outcome_gate.md`.  It does not authorize standardized records,
cohorts, labels, a dev evaluator call, GPU allocation, or full training.
