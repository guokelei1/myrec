# C68 proposal — population-relative interaction free energy

Status: pre-outcome, data-free architecture falsifier.  No repository record or
label is authorized.

## Observation → primitive

C42 showed that query-conditioned history transport can be useful and
true/wrong specific on Amazon-C4.  The identical C43 operator retained only a
small generic ranking gain on KuaiSearch: wrong-user history and shifted metric
loops reproduced it.  Standard history attention asks which event in the
provided history is most compatible; it never asks whether the same support is
ordinary across other users.  It can therefore turn any user's history into a
generic query expansion.

C68 introduces one new information object: **population-relative triadic
evidence**.  One shared bounded Transformer energy `F_theta(q,c,e)` processes query,
candidate, and event tokens.  The same parameters and token types process a
user history `H` and an exchangeable source-only population reference `R`; no
branch/source embedding tells the Transformer which set an event came from.

For a nonempty event set `S`, define its conditional free energy

```text
A_S(q,c) = tau * log mean_{e in S} exp(F_theta(q,c,e) / tau).
```

With a shared NULL-candidate token `0`, the sole personalized coordinate is

```text
d(q,c;H,R)
  = [A_H(q,c) - A_R(q,c)]
    - [A_H(q,0) - A_R(q,0)].

score_c = strong_base_c + center_candidates(d_c).
```

The first bracket removes query/candidate effects that are ordinary in the
population; the second removes candidate-free user/population affinity.  The
correction survives only when candidate identity changes how the user's event
distribution differs from the reference distribution.

## Structural identifiability

The operator has four exact contracts.

1. Adding any query/candidate-only term `a(q,c)` to every event energy cancels
   between `H` and `R`.
2. If `F(q,c,e)=a(q,c)+b(q,e)`, the complete correction is zero: a generic
   query-history carrier and a generic query-candidate ranker cannot express
   the primitive.
3. If the shared event encoder collapses every event to one constant state, or
   if `H` and `R` are the same multiset, the correction is zero exactly.
4. No history, no query, and registered repeat requests return their protected
   bases exactly.  Candidate permutation can only permute outputs.

An arbitrary user-only network is not attached after `d`; the ranking head
cannot recover a generic query/candidate function from a nonzero history
carrier.  A fixed-carrier attack and a reference-only attack are binding G0
checks, not assumptions.

## Event reference lifecycle

The data-free gate receives explicit synthetic reference events.  A real
implementation, if separately authorized, must build one label-blind reservoir
from source-only training histories before any outcome role is opened.  Events
are encoded by the same LM as user events, deterministic request hashes select
reference subsets, and the frozen reservoir identity becomes part of the
checkpoint manifest.  There is no query-type, category, dataset, candidate
count, or score-threshold branch.  The same code and masks must run on
KuaiSearch and Amazon-C4.

## Binding controls

Every mode owns the same triplet Transformer, NULL token, projections, rank
head, parameter count, initialization, optimizer, batches, and steps.  The
scalar energy is fixed to `2*tanh(raw/2)` in every mode so an unbounded
first-moment control cannot emulate tail selection only by inflating its scale.

- `interaction_free_energy` — the C68 four-way log-partition primary;
- `mean_interaction` — the exact high-temperature/first-moment reduction;
- `single_null_interaction` — replaces the population set by one shared NULL
  event, the C65/C04-style counterfactual boundary;
- `user_only_free_energy` — uses only user-history conditional free energy;
- `pooled_joint_transformer` — supplies the mean history event to the same
  triplet Transformer, the ordinary pooled-state boundary.

The first real gate would additionally bind C42-style attention, C25 anchored
Möbius interaction, RESUS-style global/residual preference, and an equal-compute
ordinary reference-token attention layer.

## Data-free falsifier

Each synthetic task contains a strong query/candidate base, a task-specific
population nuisance shared by all users, a sparse user-specific preference
present in only part of the factual history, and population reference events
whose user deviations average away.  The positive candidate is selected by
the base plus the user-specific preference.  Wrong history swaps only the
user preference.  Unsupported requests draw factual and reference events from
the same distribution.

C68 advances only if all three seeds:

- satisfy exact algebra/fallback/permutation/determinism contracts;
- beat every reduction by the locked accuracy margin;
- lose substantial accuracy under wrong history;
- keep unsupported, equal-set, fixed-carrier, and reference-only corrections
  below their locked bounds;
- keep all parameter groups active and all fits finite.

A failure closes C68 before repository data.  Generator, reference count,
temperature, dimension, sparse-event count, loss, steps, seed, and thresholds
cannot be rescued after the first outcome.
