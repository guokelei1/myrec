# C05 proposal: Candidate-Contrastive Evidence Budget Transformer

Status: **pre-outcome successor design; amended after independent pre-run review**
Candidate ID: `c05`
Current authorization: G0/G1/G2a on physical GPU 0; no dev/full/test

## 1. Observation -> consequence -> falsification

### Observation

History evidence has unequal fidelity.  Exact item recurrence is the only
stable positive history component established by C5-R3.  Coarse category
transfer is not independently useful, and the first four architecture probes
show that certificate separation, operator magnitude, null mass, or a
history/null logit difference can all exist without improving ranking.

The narrow unresolved question is therefore not whether a model can react to
history.  It is whether non-repeat history contains **candidate-discriminative
ranking evidence** after conditioning on the query and the actual candidate
set.

### Architecture consequence

Modify exactly one Transformer personalization sublayer.  Replace ordinary
positive softmax attention from each candidate to history with a
candidate-contrastive signed evidence budget.  For request query state `q`,
candidate state `c_i`, and strictly-prior history event state `h_j`:

```text
a_ij = <tanh(W_q q + W_c c_i), W_h h_j> / sqrt(d)
       + softplus(beta_exact) * exact_ij

d_ij = a_ij - mean_{k in valid candidates}(a_kj)

u_ij = relu(d_ij - tau) - relu(-d_ij - tau)

w_ij = u_ij / (1 + sum_l |u_il|)

Delta c_i = rho_max * tanh(rho) * W_o sum_j w_ij W_v h_j
c'_i = c_i + 1[history present] * Delta c_i
```

`d_ij` removes event relevance common to the whole candidate set.  The signed
dead zone lets weak evidence abstain without a certificate head.  The
denominator makes `sum_j |w_ij| < 1`, so history has a finite internal update
budget.  `rho=0` at initialization.  With no history, every `w_ij` and
`Delta c_i` is exactly zero.  Exact recurrence is not a separate scorer: item
identity is a trusted atom in the same evidence logit.

The update is inserted into the candidate residual stream before the final
Transformer FFN/ranking head.  Thus the final architecture remains one
Transformer ranker; CCEB is not a router over fixed scores.

### Falsification sequence

The original proposal attempted to use CCEB itself as the signal-existence
probe.  The pre-run review rejected that logic: a failed centering/dead-zone
operator would not prove that history signal is absent.  The frozen sequence is
now:

1. G2a uses ordinary target attention, non-repeat-only data, exact off, and no
   corruption training to test whether the frozen representation contains any
   shallow transferable signal over clean D2p;
2. G2b audits held-out hard twins without training on those constructions;
3. only survivors enter CCEB mechanism attribution;
4. exact/action/recency protection and the full Transformer are last.

Before any full system is implemented, the staged probe must show:

1. the registered D2p coordinate and candidate ordering pass exact adapter
   parity before an optimizer exists;
2. no-history scores are exactly the registered base and all-no-history full
   loss batches are finite;
3. on the frozen train-internal non-repeat cohort, true history gives a stable
   positive ranking gain over the scope-correct D2p coordinate;
4. wrong-user, shuffled-event, query-masked, and coarse-only histories do not
   reproduce the true-history **ranking gain** (attention mass is not enough);
5. repeat-present ranking is non-inferior to the registered item-only control;
6. after G2a/G2b pass, ordinary target attention, Denoising-Attention-style filtering without
   candidate contrast, and a candidate-contrastive update without the budget
   cannot reproduce the same gain under matched capacity and steps.

G2a failure stops the current frozen-representation transfer claim before CCEB
is trained.  A later-stage failure stops only its corresponding mechanism
claim.  None authorizes a deeper encoder, learned router, or threshold tuning to
rescue the story.

## 2. Why this is the minimum next experiment

C01--C04 each attempted to solve evidence fidelity and representation learning
at the same time.  C05 separates the questions:

1. **Base fidelity:** is the non-personalized coordinate exact and stable?
2. **Signal existence:** can ordinary target attention improve
   non-repeat ranking at all?
3. **Evidence authenticity:** does the gain disappear under held-out hard
   identity/query/event replacements?
4. **Mechanism value:** is candidate contrast plus a real score-space bound better
   than the nearest attention degenerations?
5. **System value:** only after 1--4 pass, internalize the base in a local LM
   and train the full end-to-end ranker.

The probe uses frozen Transformer text states and the registered D2p score only
as a falsifier coordinate.  It must never be presented as the final LLM4Rec
system or a paper result.

## 3. Predicted failure modes

- Non-repeat history may contain no stable signal; CCEB then stays near zero or
  fails to beat D2p.
- Candidate centering may remove useful absolute preference together with
  generic noise.
- Exact recurrence may dominate learning and hide a non-repeat failure.
- The operator may reduce empirically to Denoising Attention plus listwise
  context; novelty is therefore `uncertain` until matched degenerations run.
- Candidate-set dependence may be brittle under pool changes; candidate
  permutation equivariance alone is insufficient, so nested-pool/distractor
  stability is mandatory before CCEB advances.

The current code's L1 normalization bounds attention mass only.  It is not yet
a hidden-state or final-score trust region, and the proposal no longer claims
otherwise.  CCEB must be revised after G2a/G2b before mechanism training.

## 4. Explicit non-goals

- No claim that semantic/category transfer is already established.
- No claim that event order is useful until a shuffle-sensitive ranking gain is
  observed.
- No online LLM calls, generated profiles, dataset branches, or fixed-score
  router.
- No full training, dev evaluator, multi-seed, test, Amazon, or JD execution in
  the current authorization state.
