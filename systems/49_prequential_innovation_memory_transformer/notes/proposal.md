# C49 proposal — prequential innovation memory

## Primitive

C47/C48 show that raw semantic KRR contains useful cross-domain signal, while
two scalar confidence laws derived from the same normal equation do not pay
rent.  C46 separately shows that a causal content Transformer learns real user
transitions, but its pooled behavioral state does not beat semantic history.

C49 changes the value stored in history memory.  For semantic event state
`k_t` and a strict-prefix Transformer prediction `p_t`, define the behavioral
innovation `r_t = k_t - p_t`.  The current query reads

`u_q = R^T (K K^T + I)^-1 K q`,

and candidate `c` receives `c^T u_q`.  Keys retain LM-semantic addressability;
values remove the component already predicted from generic sequential
semantics.  The predictor and memory solve are inside the Transformer ranking
core, not offline features supplied to an MLP.

## Falsification boundary

The first gate trains the same two-layer, width-128 causal predictor for 500
label-free history-transition steps on the 6,000 C47 fit requests in each
domain, then scores only already-open C47-A.  Three fixed seeds per domain use
identical hyperparameters.  It must beat, on both domains, the same-checkpoint
raw-value KRR, innovation softmax, beta=1 DeltaNet, cyclically shuffled
innovation values, query base, and wrong history with the frozen intervals,
seed signs, and fold signs.  Failure closes C49 without fresh reserve.

If this exposed gate passes, it only authorizes a separately frozen fresh
architecture confirmation.  It is not itself a proposed-system result.
