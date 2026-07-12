# C36 pre-outcome implementation review

The implementation changes the internal history-attention residual state, not
an external baseline mixture. The frozen BGE LM remains load-bearing; its query,
history, and candidate states share one trained low-rank adapter. All five modes
have exactly 16,384 trainable parameters and share initialization, data,
optimizer, and loss.

The primary's defining invariants are directly computed from hidden states in
A0 rather than inferred from output metrics. Candidate permutation is tested
after canonical ordering. Repeat/no-history/no-auth/query-absent fallbacks are
exact. Candidate hashes are checked before every stage. Method code contains no
dev/test qrels or metric path and aggregation alone may open internal-A train
labels after A0.

Known risks are intentionally binding:

- pre-normalization barycenter conservation need not preserve NDCG;
- the global anchor may itself be wrong for some requests;
- the norm bound may be mechanically valid but too weak or too strong to earn
  utility over global-only/unbounded controls;
- candidate-set-relative centering can be sensitive to candidate composition.

These are reasons for the frozen matched controls, not reasons to add a learned
router or tune the trust coefficient after outcome. The label-free formulation
audit establishes only identifiability and load-bearing activity. It makes no
ranking-quality claim.
