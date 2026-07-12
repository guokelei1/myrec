# C74 pretrained-LM probe review

Decision: `authorize_exposed_fit_token_lm_probe_after_G0_and_execution_lock`.

Evidence supporting authorization:

- C74's locked data-free gate passed every condition in all three seeds;
- the real graph preserves the same raw-carrier/coupled/pooled/factual modes;
- query WordPieces stay token-resolved, so this is not another pooled BGE
  formula;
- the BGE Transformer, routing attention, and ranking energy train jointly;
- the role is exposed fit only and cannot support a final paper claim;
- validation labels, fresh roles, dev, test, and qrels have explicit stage
  barriers.

Known risk: C64 showed that late-layer LM adaptation can become strongly
rank-active without making history identity load-bearing.  C74 must therefore
pass true/wrong Top-10 activity before any validation label opens.  Passing
mechanics is not evidence of utility.
