# C19 synthetic gate protocol

Status: **frozen before implementation outcome**.  The final source, tests,
config and this protocol will be hash-locked before the one learned run.

## G0 structural gate

CPU tests must establish:

1. hand-computed diagonal/forward/reverse/cofactor values;
2. reversal leaves diagonal invariant and negates the oriented component;
3. self traces have exactly zero skew component;
4. no-history scores are bitwise query-only in every mode;
5. candidate permutation equivariance to `1e-5`;
6. candidate-common affinity traces produce zero centred residual;
7. all five trainable modes have identical parameter names/count/initial state;
8. zeroing the hidden-state write map removes all score effects despite nonzero
   temporal evidence;
9. finite nonzero gradients reach token encoder, affinity projections and the
   hidden-state write map;
10. the runner exposes no repository-data/evaluator path.

## G1 one-shot synthetic GPU gate

- environment `/data/gkl/conda_envs/myrec-c19`;
- physical GPU 3, visible as `cuda:0`;
- seeds `20260721`, `20260722`, `20260723`;
- 4,096 clean train and 1,536 independent eval requests per seed;
- fixed 25% no-history, 37.5% repeat and 37.5% supported-successor mixture;
- eight candidates, eight ordered events, 16 raw dimensions;
- modes: `oriented`, `diagonal`, `forward`, `symmetric`, `free_signed`;
- identical 500 steps, batch schedule, AdamW settings, parameter count and
  initialization; no sweep, early stop or retry;
- evaluation corruptions: wrong history, event shuffle, query mask,
  coarse-semantic removal and full reversal;
- repository records/labels, dev/test and evaluator calls: zero.

All three seeds must pass every condition:

1. repeat accuracy at least `0.98`;
2. supported accuracy at least `0.85` and at least `0.30` above base;
3. supported accuracy exceeds the best of diagonal/forward/symmetric by at
   least `0.03`;
4. worst-subset accuracy exceeds those same structured controls by `0.03`;
5. supported accuracy is no more than `0.01` below `free_signed`, while clean
   target margin exceeds forward by at least `0.15`;
6. clean target-margin gain over base is positive and wrong/shuffle/query-mask/
   coarse each retain at most `0.25`;
7. reversing history gives oriented-component correlation at most `-0.80` and
   retains at most `0.10` of clean target-margin gain;
8. at least 30% of history-present requests have score-delta range `>=0.05`;
9. no-history is bitwise base, candidate permutation error `<=1e-5`, rescore is
   deterministic, all values finite, and parameter/init audits match.

Failure closes C19 before real data.  Passage only permits design of a separate
train-internal real gate; it does not authorize its execution, dev or test.
