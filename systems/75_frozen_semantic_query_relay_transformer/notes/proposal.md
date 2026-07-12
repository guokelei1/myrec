# C75 proposal — Frozen Semantic Query-Relay Transformer

Status: pre-outcome design.  C75 has not read a validation label, fresh role,
dev, test, or qrel.

## Observation → architecture consequence → falsification

**Observation.** C74's semantic-conservative two-hop relay passed its fresh
data-free gate by large margins and became strongly Top-10- and wrong-history-
load-bearing with pretrained BGE states.  Yet adapting BGE's last two layers
moved the supposed semantic carrier substantially: final-to-initial cosine was
only `0.786--0.852` for history and `0.792--0.846` for candidates.  Two of
three seeds then failed the all-mode training trend.  A carrier cannot serve as
the stable reference coordinate while ranking loss is free to rewrite it.

**Architecture consequence.** C75 makes the pretrained LM an immutable part of
the ranking architecture:

```text
q_t, h_j, c_i = stopgrad(LM_0(text)); LM_0 is always eval

a_tj = softmax(<R_1(q_t),R_1(h_j)>/tau + chronology_j)
q^H_t = normalize(q_t + sum_j a_tj h_j)

b^H_it = softmax(<R_2(c_i),R_2(q^H_t)>/tau)
b^0_it = softmax(<R_2(c_i),R_2(q_t)>/tau)
delta_i = center[3*tanh(sum_t b^H_it<c_i,q^H_t>
                              - sum_t b^0_it<c_i,q_t>)].
```

Only `R_1`, `R_2`, and chronology train.  The same frozen Transformer token
states remain load-bearing at history value, query carrier, candidate value,
and energy readout.  This is not offline LM features passed to an MLP: the LM
states enter two trainable attention stages and the end-to-end ranking energy,
and no non-Transformer scorer exists.

No history/query returns D2p exactly; exact repeat returns item-only exactly.
The graph has no dataset/category/query-type branch.

**Falsification.** Before training, G0 must prove the serialized backbone state
is bit-exact before/after disposable route optimization, backbone parameters
never receive gradients, route/chronology gradients are active, and all
fallback/permutation/wrong-history contracts pass.  Formal A0 then requires:

1. every seed/mode lowers loss on the *same* hash-frozen 256-request anchor
   from initialization to final checkpoint by at least 0.5%;
2. primary changes base and matched wrong-history Top-10 decisions in every
   seed;
3. primary is rank-distinct from equal-parameter coupled-value, pre-relay
   pooled, and factual-only reductions;
4. backbone hashes remain unchanged and validation labels remain closed.

Only an A0 pass may open the 1,200 exposed-fit validation labels.  A1 binds
primary over base, wrong history, and all reductions with paired intervals,
seed signs, and fixed folds.  Failure closes C75; no layer unfreezing, learning
rate, epoch, seed, cohort, scale, or threshold rescue is allowed.

## Why this is a new architecture constraint, not C74 rescue

C74 is terminal and its validation labels stay closed.  C75 has a new ID,
source tree, model class, optimizer parameter set, G0, fixed-anchor gate,
execution lock, checkpoints, and seeds.  Freezing the carrier changes the
function class and training dynamics: no ranking gradient can alter LM token
meaning.  C74's checkpoints and scores are not reused.

## Predicted failure modes

- Frozen LM semantics may not encode behavioral preference directions.
- Route-only capacity may be insufficient to beat the strong base.
- Token-resolved relay may again tie its pooled reduction.
- Wrong histories may remain semantically coherent and useful.
- Fixed anchor loss may improve while held-out utility remains null.
