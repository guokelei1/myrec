# Locked proposal: Reversible Write–Probe–Undo (RWPU)

Lock time: 2026-07-11 (Asia/Shanghai). This proposal was formulated without
reading any other candidate workspace. It uses only the repository-wide rules
in `AGENTS.md` and docs 07, 11, and 12.

## Insight template

**Observation.** A history update should affect ranking only when it fails to
commute with the current query/candidate's internal probe; a candidate-common or
unsupported write should disappear under exact undo.

**Architecture consequence.** Insert one reversible write–probe–undo primitive
inside the Transformer's candidate-token FFN state: history composes a nonlinear
volume-preserving map, the query/candidate applies a probe map, both are undone,
and only the centered closed-loop displacement enters subsequent Transformer
blocks.

**Falsification.** If the full loop does not beat a parameter-matched ordinary
terminal-state memory on a learned conditional synthetic task, or if wrong,
shuffled, query-masked, or disjoint evidence reproduces its effect, the primitive
is an unnecessarily elaborate memory read and C08 stops before real data.

## One primitive

Let the memory state be two streams, `z=(x,y)`, each in `R^d`. A history event
encoding produces two normalized evidence axes `p_t,q_t` and strength `s_t`.
One history write is the conditional additive coupling

```text
x' = x + s_t g_h,1 p_t tanh(<q_t,y>  + b_h,1)
y' = y + s_t g_h,2 q_t tanh(<p_t,x'> + b_h,2).
```

It is exactly inverted by

```text
y  = y' - s_t g_h,2 q_t tanh(<p_t,x'> + b_h,2)
x  = x' - s_t g_h,1 p_t tanh(<q_t,y>  + b_h,1).
```

The two triangular shears each have unit Jacobian determinant, hence their
composition is volume-preserving. This is the conservation law used here; no
claim of Euclidean-norm preservation is made.

For chronological history `h_1,...,h_T`, define

```text
W_H = U_hT ... U_h2 U_h1.
```

The shared Transformer encodes a query/candidate pair into probe axes and a
probe strength, defining the same kind of invertible coupling `P_qc` with
probe-role gains. The only history residual is

```text
r(q,c,H) = (P_qc^-1 W_H^-1 P_qc W_H - I) z0.
```

Across the fixed candidate set, the parameter-free invariant

```text
r_centered(c) = r(c) - mean_j r(j)
```

removes a candidate-common displacement before it reaches the ranking head. A
linear map injects `r_centered(c)` into the candidate token between Transformer
blocks. Remaining Transformer blocks and a shared head produce the final score.
This centering is part of the primitive's ranking invariant, not a second
expert, gate, or score router.

## End-to-end information flow

```text
query tokens ─┐
              ├─ shared lower Transformer ─ candidate probe P_qc ─┐
candidate ────┘                                                    │
                                                                   ├─ RWPU residual
history events ─ shared lower Transformer ─ writes U_h1...U_hT ───┘
                                                                   │
candidate token + centered residual ─ upper Transformer ─ score ──┘
```

The same item-ID embedding is present in history and candidate representations,
so exact recurrence has a direct shared-axis path. Query-conditioned text
representations can align non-identical items in the learned evidence space.
There is no category rule and no dataset branch. In the ideal disjoint-support
case, a history write and candidate probe commute exactly, so the loop residual
is zero. Cross-item personalization therefore requires learned overlap among
query, history, and candidate axes rather than merely a nonempty history.

## Exact contracts

1. **No history.** `H=empty` makes `W_H=I`, hence `r=0`. The implementation
   explicitly overwrites inverse-roundoff with an exact zero under the history
   availability mask. The entire ranker is then bitwise identical to its own
   query-only backbone.
2. **Candidate permutation.** Each probe uses shared parameters; subtracting a
   set mean is permutation-equivariant. Permuting candidates only permutes
   scores.
3. **Common mode.** Adding one identical memory displacement to every candidate
   changes neither the centered hidden update nor score order.
4. **Undo.** Each write is inverted with the same axes, strength, and parameters,
   and history is unwritten in reverse chronological order.
5. **Conservation.** Each coupling has determinant one; a test checks both exact
   reconstruction and the Jacobian determinant.

## Why this is not merely an attention rename

The primitive has no softmax, no normalized row weights, and no weighted sum of
history values. History changes an internal nonlinear FFN state transition. The
candidate read is the failure of two invertible state transformations to commute.
An ordinary endpoint-memory ablation uses the same encoders, axes, strengths,
readout, and parameter count but reads only `W_H z0`.

There is a constructive separation from that matched endpoint state: one valid
history transformation can fix `z0` exactly (`W_H z0=z0`) while acting
nontrivially on the state reached after `P_qc`; the ordinary endpoint is then
identical to empty history while RWPU is nonzero. The CPU test freezes this
witness.

This is deliberately a limited claim. A sufficiently general Transformer over
the full history, or an operator-valued memory large enough to retain the whole
map `W_H`, can emulate the finite computation. C08 claims an architectural
inductive bias and a separation from the matched terminal-vector control, not a
universal function-class impossibility theorem.

## Backbone and fallback integration

The prototype uses a lower and upper `TransformerEncoderLayer`, with RWPU
inserted between them. A real probe may start only from the exact query-only
Transformer checkpoint/function used for the frozen D2p anchor. To satisfy the
4,110-record fallback contract, those base weights must remain frozen during
the design gate; only the internal RWPU parameters may train. If an integration
cannot make no-history scores/ranks exactly equal to that anchor, C08 stops
rather than adding a score-level fallback.

## Complexity and non-claims

The low-rank coupling costs `O(C H d)` per request for `C` candidates, `H`
history events, and evidence width `d`; it replays history in reverse for each
candidate. There is no online LLM call in the prototype. Latency, quality, and
memory advantages are not claimed before measurement.

The proposal does not use a graph construction, externally generated adapters,
prompt/prefix deltas, score experts, a fixed router, category-specific logic, or
dataset-specific rules. It makes no explanation, novelty, real-data quality, or
efficiency claim at the current gate.
