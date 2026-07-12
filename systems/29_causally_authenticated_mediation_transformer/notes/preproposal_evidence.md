# C29 preproposal evidence boundary

C28 passed every label-free structural check but failed all internal-A utility
checks.  Its corrections had near-zero cross-seed correlation and a
three-order-of-magnitude scale spread.  Post-terminal tests on the already-open
C28 internal-A established the following bounded facts:

- proper pair loss alone gave primary-minus-D2p
  -0.000527/-0.000614/+0.000020;
- fixed contextual BGE query-aspect residual had clicked direction +0.000010
  with a zero-crossing interval; purchase-only was +0.000006;
- a causal item-cooccurrence graph covered only one of 1,346 clicked rows;
- brand/seller/category exact attributes and product-quantized semantic codes
  had no positive-CI direction;
- causal same-query clicked-item memory covered no candidate row;
- full pretrained factual-minus-null BGE gave positive D2p deltas in three
  seeds, but wrong history was better in two seeds;
- adding true/wrong/null ranking contracts made all three D2p deltas positive
  (+0.002015/+0.000700/+0.000665) but true-minus-wrong remained near zero;
- strict prior-user authentication separated true from wrong history by
  +0.3836 [0.3530, 0.4136], with true higher on 342 requests and wrong higher
  on one;
- applying that mask to the full LM produced mean D2p gain approximately
  +0.00099 over three seeds, but only 2/3 seeds were positive.  Frozen and
  convex readout controls did not repair stability.

These are design-formulation diagnostics, not C29 validation and not paper
results.  C29 may use only the authentication law and pretrained mediation
path.  It may not tune score caps, select the two favorable seeds, filter the
evaluation cohort to authentication-present requests, or reuse C28 internal-A
as an outcome.
