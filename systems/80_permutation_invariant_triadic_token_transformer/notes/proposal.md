# C80 proposal — Permutation-Invariant Query-Authenticated Triadic Token Transformer

Status: pre-outcome final architecture proposal.  C80 is update 3/3 after C76;
there will be no C81.

## Evidence chain

1. Amazon full-token HSO established a practical, user-specific source on a
   previously unopened reserve (`true-null +0.02530`, `true-wrong +0.03594`).
2. Frozen edge ablation showed direct bidirectional C-H plus Q-H contextual
   interaction is load-bearing; pooled states and query-only relays are not.
3. C76 showed that factual/cut trajectories still admit candidate-only
   shortcuts through adaptive history modulation.
4. C77 showed that a frozen semantic admission graph blocks that shortcut, but
   positional history created unsupported order sensitivity and its triadic
   graph tied a simpler filter.
5. C78 made history events exchangeable.  Its `triadic_set` control reached
   clean/shuffle supported accuracy `1.0`, wrong accuracy `0`, and exact shuffle
   retention in all three seeds; it matched or beat every C78 primary.

## Final primitive

A frozen pretrained LM embedding coordinate gives normalized WordPiece anchors
`a_t`.  For candidate token `c`, history token `h`, and current query tokens Q,

```text
g(c,h|Q) = max_q [<a_c,a_q>]_+ [<a_h,a_q>]_+ [<a_c,a_h>]_+.
```

Fixed top-budget admission retains candidate and per-event history tokens with
the strongest positive triangle support.  Ranking gradients cannot change the
anchor or admission.  Query/readout tokens and admitted C/H tokens form a dense
bidirectional interaction graph.  History events share within-item position
IDs and have no event-index embedding, making complete-event permutation a
structural symmetry while preserving WordPiece order inside each item.

One adaptive BGE Transformer is evaluated on the authenticated graph with all
edges and with H cross-edges cut.  A frozen BGE query-candidate coordinate,
initialized from the same already-trained full-token HSO checkpoint, supplies
the protected base:

```text
delta_i = rho*tanh(logit_full_i - logit_H-cut_i)
score_i = frozen_base_i + center_candidates(delta_i).
```

The subtraction is a registered safety interface; the innovation claim, if
any, is the **frozen query-authenticated, event-set-equivariant raw-token
attention graph**.  No-history/query-mask returns the base; exact recurrence
retains item-only.  No dataset/category/query-type branch exists.

## Fresh real gate

Fit uses the 5,966 strict-nonrepeat Amazon requests already used by token HSO.
Evaluation uses every remaining C38-unused request that is strict-nonrepeat and
whose user is absent from fit: 365 requests / 365 users.  None was tokenized,
scored, or label-opened by HSO or C38--C78.  Selection, wrong donors,
tokenization, graphs, modes, seeds, and thresholds freeze before fit labels are
reused.  Fresh labels open only after all scores pass mechanics.

Three seeds train C80 and four equal-capacity controls:

1. `triadic_set` — final primary;
2. `query_filtered_set` — C78 primary;
3. `pairwise_set` — no query authentication;
4. `triadic_positional` — removes event-set symmetry;
5. `ungated_full` — ordinary full token graph with the same residual interface.

The frozen ordinary full-token HSO checkpoint is an external strong control.
C80 passes only if primary beats protected null/base, wrong history, every
trained control, and ordinary full-token scoring with positive user-cluster
intervals, all-seed signs, and at least `+0.002` over base.  Candidate/event
permutation, deterministic, frozen-anchor/base hashes, no-history, and label
staging are binding.

Failure closes architecture search.  No token budget, anchor centering,
similarity, layer, LR, epoch, seed, cohort, threshold, or dev rescue follows.

## Novelty ceiling

Triadic attention, Set Transformers, pretrained routing, cross-encoders, and
counterfactual score differences are established individually.  C80 is not
declared globally novel before outcome.  A positive real result plus the
registered reductions could support a narrow architecture contribution; a tie
would leave only a strong ordinary full-token baseline and the retrospective
lesson.
