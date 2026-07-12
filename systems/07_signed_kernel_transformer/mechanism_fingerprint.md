# C07 Mechanism Fingerprint

## Canonical name

**PDSK** — Pairwise Dead-zone Signed Kernel.

## Minimal distinguishing equation

For \(C\ge3\), \(\tau>0\):

\[
  a_{ij}=
  {\frac{1}{C-1}\sum_{k\ne i}
    \left([s_{ij}-s_{kj}-\tau]_+-[s_{kj}-s_{ij}-\tau]_+\right)
  \over
  \kappa+\sum_{r,t}\left|
    \frac{1}{C-1}\sum_{k\ne r}
    \left([s_{rt}-s_{kt}-\tau]_+-[s_{kt}-s_{rt}-\tau]_+\right)
  \right|}.
\]

Everything else in the proposal is replaceable scaffolding.  Removing the
candidate-pair soft-threshold or the signed conservation removes the C07
fingerprint.

## Information fingerprint

| Field | C07 value |
|---|---|
| Transformer insertion point | candidate-to-history attention normalization |
| Raw interaction | query-modulated candidate/history tri-linear logit + query/candidate-modulated exact-ID feature |
| Competition axis | candidates compete pairwise for each history event |
| Nonlinearity | odd soft-threshold on candidate margins |
| Sign | positive and negative attention weights |
| Conservation | per-event candidate sum exactly zero |
| Abstention | exact zero on an open pairwise dead zone |
| Magnitude control | null-mass absolute normalization; total signed mass below one |
| History-to-score path | internal Transformer state update, then shared LM/ranking head |
| No-history behavior | pointwise logit equality with query/candidate base |
| Dataset branching | none |
| Minimum identifiable candidate count | three |

## Algebraic collision tests

The design must be renamed or stopped if any implementation fails one of these:

1. `tau == 0` or learned \(\tau\) numerically collapses to zero: fingerprint
   becomes centered linear attention.
2. Removing pairwise comparisons leaves the same function: fingerprint was
   decorative.
3. For random active triples, the C07 balance is always collinear with centered
   logits or centered-softmax weights: it is a scalar gate in disguise.
4. Weights are constrained nonnegative or sum to one: it is sparse/ordinary
   attention, not PDSK.
5. Candidate weights are computed independently: it is target attention.
6. Abstention requires selecting a fixed base expert after scores are formed:
   it is a router, not kernel-level abstention.
7. The signed update can contain a candidate-common component: the conservation
   contract was lost.

## Declared degeneracies

| Setting | Result | Interpretation |
|---|---|---|
| \(\tau=0\) | scaled centered logits | mandatory linear control |
| \(C=2\) | one-dimensional zero-sum vector | not mechanism-identifiable |
| all ranges \(\le\tau\) | exact zero | intended abstention |
| \(\tau\to\infty\) | exact zero | dead model |
| \(\kappa\to\infty\) | update tends to zero | dead model |
| \(\kappa\to0\) | unit L1 mass when active | unstable/no abstention reserve |
| no valid history | exact zero | required fallback |
| per-event common logit shift | no change | required invariance |

## Closest forbidden reparameterizations

- `center(scores) / L1(center(scores))`;
- `gate(range(scores)) * center(softmax(scores))`;
- two softmax maps subtracted with a scalar coefficient;
- softmax over history independently for each candidate plus a null token;
- a fixed base score plus a separately routed history score.

These are controls, not acceptable alternate descriptions of C07.
