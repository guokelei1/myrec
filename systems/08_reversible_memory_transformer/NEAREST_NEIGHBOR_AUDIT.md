# Nearest-neighbor and reducibility audit

Audit date: 2026-07-11. Scope: primary papers found by a targeted search for
reversible recurrent/Transformer memory, orthogonal recurrent updates, recurrent
Transformer memory, fast weights, DeltaNet, and products of reversible state
transitions. This is a collision screen, not a patent search or a novelty proof.

## Closest architecture families

| Neighbor | What it already establishes | Collision decision for C08 |
|---|---|---|
| [RevNet](https://arxiv.org/abs/1707.04585), [Reversible RNN](https://proceedings.neurips.cc/paper_files/paper/2018/hash/4ff6fa96179cdc2838e8d8ce64cd10a7-Abstract.html), and [Reformer](https://arxiv.org/abs/2001.04451) | Invertible activations/transitions can reconstruct earlier states and reduce activation storage. | Reversibility alone is not novel. In those closest uses, inverse execution serves training-memory reconstruction; C08's score depends on an inverse trajectory in the forward ranking computation. |
| [R3Mem](https://aclanthology.org/2025.findings-acl.235/) | A reversible Transformer can compress context forward and reconstruct it backward for long-context retention/retrieval. | Very close in using the inverse at task time. R3Mem reconstructs context with reversible adapter streams; C08 measures a query/candidate-conditioned noncommutation residual inside ranking and has exact empty-history cancellation. A learned matched control is still mandatory. |
| [scoRNN](https://proceedings.mlr.press/v80/helfrich18a.html) and other orthogonal/unitary RNNs | Norm-preserving recurrent matrices improve long-memory optimization. | Conservation or invertibility by itself is not C08's claim. C08 uses nonlinear unit-determinant coupling and the full write/probe/undo composition. |
| [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) | Recurrent memory tokens pass compressed information between Transformer segments. | RMT is a forward terminal-memory read. It is a required conceptual endpoint-memory neighbor, not the same forward scoring map. |
| [Mamba](https://arxiv.org/abs/2312.00752) | Input-dependent recurrent state transitions selectively propagate or forget tokens. | An input-conditioned transition is not enough. C08 has exact inverse composition and a candidate probe; if its advantage is reproduced by a selective forward recurrence, the claim shrinks or stops. |
| [Linear Transformers as Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) and [DeltaNet](https://arxiv.org/abs/2406.06484) | Attention-like reads are recurrent fast-weight memories; delta updates improve associative recall. | This is the strongest warning against calling any recurrent memory read new. C08 must beat a matched ordinary endpoint memory and generic history attention, not only a static mixture. |
| [DeltaProduct](https://arxiv.org/abs/2502.10297) | Products of generalized Householder transformations give expressive, stable recurrent state transitions. | Strongest structural neighbor. Products of reversible transitions are already known. C08 survives the paper audit only because its load-bearing object is the nonlinear candidate-conditioned `P^-1 W^-1 P W` read, not the product `W` alone. Failure to beat DeltaProduct-like/endpoint controls closes C08. |

No directly matching product-ranking method using the exact nonlinear
write–probe–undo commutator was found in this bounded search. That absence is not
evidence of novelty and must not be written as such.

## Reduction ladder

### 1. Scalar gate or fixed-score router

RWPU does not choose among precomputed query/history scores. Its output is a
vector-valued internal state displacement before the upper Transformer. There is
no scalar expert weight. Renaming its history/probe strengths as a gate loses the
inverse-order interaction and cannot reproduce the endpoint-collision witness.

Decision: **not a trivial scalar-gate reduction**.

### 2. Ordinary terminal-vector memory

The exact matched control forms the same forward state `W_H z0`, uses the same
history/candidate encoders and parameters, and reads its projection on candidate
axes. The frozen witness chooses a valid write that fixes `z0` but changes a
probe-perturbed state. Therefore

```text
W_H z0 = z0                  (ordinary endpoint equals empty history)
(P^-1 W_H^-1 P W_H - I)z0 != 0.
```

No function that receives only that terminal vector and the candidate can
distinguish the two cases. This is a constructive, same-state-width separation.

Decision: **nontrivial relative to the matched terminal-vector control**.

### 3. Additive attention / first-order similarity memory

For weak transformations, the loop's leading term is a commutator and is
bilinear in history/probe strengths and axis overlap. That term can resemble a
similarity-weighted history read. This is a genuine reducibility risk, not a
selling point. The learned synthetic gate requires surplus over same-backbone
history attention and ordinary memory; otherwise C08 stops.

Decision: **not cleared by algebra alone; outcome-gated**.

### 4. Operator-valued recurrent memory or generic Transformer

If an ordinary memory is allowed to store the entire nonlinear map `W_H`, or a
generic Transformer retains and processes every history event, it can emulate
the finite loop. The current work has no lower bound against those universal or
larger-state controls.

Decision: **no absolute irreducibility claim**. The defensible claim is a novel
candidate-conditioned state-update bias only if matched learned controls lose.

## Audit verdict

The initial idea “use reversible rotations as memory” is closed as a trivial
collision with orthogonal RNN/DeltaProduct families. The surviving nonlinear
RWPU primitive has a valid endpoint-state separation and is worth one bounded
learned synthetic falsifier. It is **not yet worth a real-data probe**. Synthetic
failure closes the candidate without reinterpretation.
