# C09 Proposal: Conjunctive Margin Attention

## One-sentence insight

**Observation.** History should change a candidate ordering only when two
structurally different, information-restricted paths through the same
Transformer support the same history-induced candidate margin.

**Architecture consequence.** Use one primitive—Conjunctive Margin Attention
(CMA)—that opens candidate-to-candidate contrast attention only on ordered
pairs whose query-first and candidate-first residual margins agree.

**Cheap falsification.** Stop if one-view corruption retains the correction, a
matched scalar/diagonal gate or single-view attention matches CMA, or either
restricted mediator leaks the evidence from which it is required to be blind.

This is a hypothesis derived from unequal history-evidence fidelity, not a
claim that agreement itself is new or already useful.

## 1. Shared record and tokens

For one request, the input is

```text
x = (optional query q, optional history H=(h_1,...,h_T),
     fixed candidate set C={c_1,...,c_N}, evidence masks).
```

One shared token Transformer `E_theta` encodes query, history-item, and
candidate text/identity tokens.  One shared history Transformer contextualizes
`H`.  There is no dataset, category, query-type, or surface branch.  Query and
history availability masks are the only structural conditions.

## 2. Two restricted views, one set of parameters

The views differ only in an attention mask / choice of the first mediator
query.  They do not have separate encoders, adapters, heads, prompts, prefixes,
or expert weights.

### Q-first path: history selection is candidate-blind

```text
a^Q = MHA_theta(E_theta(q), E_theta(H), E_theta(H)).
u_i = w^T T_theta([RANK, E_theta(q), E_theta(c_i), a^Q]).
```

`a^Q` can use query and history but cannot observe any candidate.  The same
mediator is presented to every candidate.  It asks whether the historical
evidence is relevant to the current query before seeing which candidate might
benefit.

### C-first path: history selection is query-blind

```text
a_i^C = MHA_theta(E_theta(c_i), E_theta(H), E_theta(H)).
v_i = w^T T_theta([RANK, E_theta(q), E_theta(c_i), a_i^C]).
```

`a_i^C` can use candidate and history but cannot observe the query.  It asks
whether history provides candidate-specific support before current-query
context is reintroduced by the shared rank Transformer.

The same `MHA_theta`, `T_theta`, token encoder, and linear head `w` compute both
paths.  “Independent” is not claimed: the paths share weights and may have
correlated errors.  They are only **structurally information-restricted**.

### Base path

```text
b_i = w^T T_theta([RANK, E_theta(q), E_theta(c_i), a_empty]),
```

where `a_empty` is a learned no-history mediator.  This is the query/candidate
Transformer ranker, not an external fixed-score router.

## 3. The CMA primitive

Compute view residuals relative to the *same* base, not raw relevance scores:

```text
r_i^Q = u_i-b_i,                       r_i^C = v_i-b_i
m_ij^Q = r_i^Q-r_j^Q,                  m_ij^C = r_i^C-r_j^C.
```

The ordered-pair permission and update are

```text
K_ij = (m_ij^Q)_+ (m_ij^C)_+
       / ((m_ij^Q)_+ + (m_ij^C)_+),    K_ii=0

A_ij = K_ij / (1 + sum_l K_il)
Delta z_i = sum_j A_ij W_V(z_i^0-z_j^0)
s_i = b_i + M_q M_h w_o^T Delta z_i.
```

`M_q` and `M_h` are query/history-present masks.  The parallel sum makes
`K_ij` exactly zero unless both paths support `i` over `j`, and no larger than
the weaker support.  Negative agreement is represented by the reverse ordered
pair `(j,i)`.  `z_i^0` is the base rank-token state.

CMA is inside the ranking core: its listwise state is read by the final shared
Transformer head.  It does not choose among precomputed ranker scores.  Its
candidate-pair matrix and off-diagonal value mixing are what distinguish it
from an ordinary scalar gate; the strict qualification is in
`reduction_audit.md`.

## 4. Invariants and exact contracts

1. **No history:** `M_h=0` uses `torch.where` to return `s=b` bit exactly.
2. **Query masked:** `M_q=0` returns the masked-query base exactly; query-masked
   history cannot create a correction.
3. **Disagreement:** if every pair disagrees, `K=0` and `s=b` exactly.
4. **Common-mode:** adding any candidate-constant offset to either view's
   residual logits leaves every margin and final output unchanged.
5. **Candidate permutation:** permuting candidates permutes `s`; no source-order
   or item-ID ordering is used.
6. **Singleton:** `N=1` has no margin and therefore no correction.
7. **Information barriers:** changing candidates cannot change `a^Q`; changing
   the query cannot change any `a_i^C`.

The prototype and hand-computed tests exercise every item.

## 5. Train-only objective

If a later synthetic gate authorizes training, use labels only from authorized
training records:

```text
L = L_rank(s,y) + L_rank(b,y)
    + 0.5 L_rank(u,y) + 0.5 L_rank(v,y).
```

The auxiliary view losses are needed because exact disagreement blocking
intentionally gives zero CMA gradient to a view on a blocked pair.  There is no
KL/agreement loss: directly rewarding agreement could let the shared paths
collude and would make the permission certificate self-fulfilling.  Dev/test
qrels are never inputs to training or scoring.

The coefficients above are frozen for the first synthetic falsifier; no sweep
is permitted before that result.  A later data implementation must use the
shared evaluator and its registered tuning budget.

## 6. Nearest controls and ablations

All controls retain the same base Transformer, token access, candidate set, and
training labels.

1. query/candidate-only base `b`;
2. Q-first-only attention (`K_ij=(m_ij^Q)_+`);
3. C-first-only attention (`K_ij=(m_ij^C)_+`), the closest DIN-like control;
4. arithmetic mean and product-of-probabilities view fusion (diagnostics only,
   never called C09);
5. global scalar personalization gate with matched extra feed-forward capacity;
6. per-candidate diagonal residual gate with matched capacity;
7. ordinary learned candidate self-attention receiving both view states but no
   conjunctive margin mask;
8. CMA with one view paired to a shuffled request;
9. parameter-matched base with an extra Transformer/FF block;
10. remove `W_V(z_i-z_j)` and use a constant value; if this matches CMA, the
    mechanism has collapsed to confidence gating and C09 stops.

## 7. Complexity

The reference computation uses three shared rank passes (base, Q-first,
C-first), one shared history encoding, and an `O(N^2 d)` CMA matrix.  Parameters
added by CMA are `d^2+d`; the two views add no view-specific parameters.  A
quality claim must be compared with a base enlarged by the same parameter and
compute budget.  Latency is a material risk for large candidate sets.

## 8. Predicted failure modes

- **Path collusion:** shared weights can make the restricted margins agree for
  the same wrong reason.  Wrong-user, shuffled-event, and shuffled-view controls
  must break the gain.
- **No non-repeat signal:** reliable exact recurrence may not yield useful
  cross-item agreement; then the non-repeat gate fails and C09 stops.
- **Dead disagreement gradients:** exact blocking makes optimization brittle;
  auxiliary view losses may still be insufficient.
- **Attention relabeling:** a standard learned candidate-attention control may
  match CMA, leaving no innovation surplus.
- **Base mismatch:** the prototype proves `s=b` with no history, but it does not
  prove that a future `b` is rank-equivalent to the frozen D2p control.  That is
  a mandatory pre-evaluator check, not an assumption.
- **Identical candidate states:** agreed margins cannot help when all base
  contrast values are equal.
- **Quadratic online cost:** candidate-pair attention may fail the eventual
  cost/quality comparison even if it improves quality.

## 9. Current claim boundary

The current artifact establishes only algebraic and software contracts on
synthetic tensors.  It has not accessed a cohort, label, qrels, dev metric,
GPU, or test split.  It does not establish evidence fidelity, personalization
gain, D2p equivalence, or publication-level novelty.
