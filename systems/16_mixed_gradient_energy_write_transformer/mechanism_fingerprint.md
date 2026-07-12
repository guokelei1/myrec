# Mechanism fingerprint

## Intended intervention

Let `z_i` be the end-to-end Transformer state for candidate `i`, and let `h_j`
be the state for history event `j`.  C16 would replace or augment an ordinary
attention write with a vector field derived from a scalar interaction energy:

```text
E(Z,H,q) = aggregate_ij epsilon_theta(z_i,h_j,q),
Delta z_i = -eta_i grad_{z_i} E.
```

The hoped-for primitive is that an energy gradient supplies a direction that is
candidate-specific, evidence-sensitive, and structurally safer than an
unrestricted pair MLP.  Five natural branches were considered before any code:

| branch | proposed differentiator | pre-implementation result |
|---|---|---|
| linear candidate gradient | values are derivatives of bilinear scores | weight-tied cross-attention / modern Hopfield |
| nonlinear conservative write | learned scalar pair energy before aggregation | Energy Transformer / HFY energy retrieval |
| mixed-Hessian write | contract `d^2 Psi / dz dh` with an event direction | scalar-potential gradient if conservative; no energy claim otherwise |
| candidate-axis energy | events allocate mass competitively across candidates | Slot Attention allocation / bipartite restriction of ET |
| centred energy | remove the uniform softmax component | HFY energy choice / Differential Transformer fixed-uniform case / ZeroS |

## Required witness

For C16 to be a new architecture primitive, it would have to exhibit a vector
field `F_i(Z,H)` satisfying all of the following before outcome evaluation:

1. it is not pointwise equal to tied cross-attention or a modern-Hopfield update;
2. its conservative law is not merely `F_i=-grad_{z_i} Phi` for a renamed
   scalar neural energy already covered by energy-based Transformers;
3. candidate competition is not obtained solely by changing the softmax axis;
4. its mixed-Hessian construction remains conservative without reducing to the
   gradient of a contracted scalar potential;
5. its signed/zero-sum term is not `softmax(s)-softmax(0)` or a learned
   difference of existing attention maps; and
6. its advantage cannot be matched by an equal-capacity generic pairwise vector
   network after removing the energy interpretation.

The mixed-Hessian branch cannot satisfy items 2 and 4 simultaneously.  The
remaining branches fail items 1, 3, or 5 by exact construction.  Consequently
there is no surviving C16 fingerprint to freeze.

## Safety contracts do not create novelty

No-history exactness, a zero-valued NULL state, bounded non-zero LayerScale,
candidate-mean centring, deterministic scoring, and corruption specificity
would remain mandatory for any successor.  They constrain a write but do not
distinguish C16 from its nearest neighbours, so they cannot rescue this family.

No model, GPU, synthetic record, real record, or label was used to reach this
decision.
