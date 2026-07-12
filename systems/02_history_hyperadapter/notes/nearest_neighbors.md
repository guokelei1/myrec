# CHHT nearest-neighbor audit

Audit date: 2026-07-11, before any C02 model/dev outcome.  Sources below are
primary papers or official project pages.  The verdict is operator-level and
does not claim that CHHT is globally novel.

## Search question and pivot

The first candidate operator was a history-conditioned diagonal update,
`U diag(alpha(q,c,H)) V^T`.  That operator is **reducible** to input-sensitive
LoRA families: DISeL directly generates input-dependent diagonal rank gates,
Ouroboros modulates LoRA singular directions from the current hidden state,
and Gated LoRA selects ranks as a function of the input.  It was discarded
before proposal lock and before any C02 model outcome.

The locked operator instead composes an off-diagonal skew matrix from
query-candidate-event triples and applies its Cayley rotation within a shared
low-rank FFN subspace.  This is a bounded, ephemeral ranking-time update; its
diagonal is structurally zero and it mixes rank directions rather than scaling
independent LoRA atoms.

## Closest operator families

| Neighbor | Closest mechanism | Reducibility test against CHHT |
|---|---|---|
| [Profile-to-PEFT](https://arxiv.org/abs/2510.16282) | A hypernetwork maps a user/profile representation to full LoRA `A,B` parameters. | The profile generator can specialize a persistent/profile-level adapter, but it does not specify an event-composed, per-candidate skew/Cayley update.  CHHT changes when only the candidate changes and stores no user adapter.  Replacing CHHT's kernel with profile-generated `A,B` is the profile-adapter degeneration. |
| [DISeL](https://arxiv.org/abs/2605.19028) | The input generates diagonal gates in `A diag(g(x)) B`, selecting/scaling rank-one LoRA components. | The discarded diagonal proposal is reducible.  CHHT is not diagonal rank gating: its generator produces a zero-diagonal skew matrix and Cayley mixes directions nonlinearly.  A diagonal-core ablation is the explicit DISeL-style degeneration. |
| [Ouroboros](https://arxiv.org/abs/2604.02051) | A hidden-state controller dynamically modulates LoRA directions expressed on SVD bases. | Like DISeL, its closest form modulates directions rather than constructing candidate/event pairwise off-diagonal rotation.  Removing CHHT's event composition and retaining per-rank modulation makes it reducible. |
| [Gated LoRA](https://openreview.net/forum?id=ZiBDVotA6g) | Input-conditioned gates dynamically select effective LoRA rank. | Rank selection is a diagonal/subset operation.  It cannot recover a general Cayley rotation without adding the load-bearing off-diagonal core. |
| [NaRA](https://arxiv.org/abs/2605.29716) | A dynamic dense `r x r` core is generated between low-rank factors. | This is the closest algebraic warning: a dense dynamic core is broad enough to contain a Cayley core.  CHHT's remaining distinction is the *specific* skew/event-composition constraint and exact zero-history identity, not the existence of a dense core.  An unconstrained dense-core control would therefore test the constraint, not prove novelty. |
| [Queryable LoRA](https://arxiv.org/abs/2605.08423) | Attention routes an input over learned low-rank atoms. | Soft routing over fixed atoms is not the same operator as an event-composed rotation, but a sufficiently large atom bank could approximate it.  Fixed-atom routing is a meaningful degeneration/control, not an application-only distinction. |
| [CoMoL](https://arxiv.org/abs/2603.00573) | Token-conditioned routing and soft merging of static low-rank experts. | CHHT neither selects experts nor mixes fixed score/weight branches.  Replacing the kernel with expert weights would remove the skew/Cayley constraint. |
| [RadarGate](https://arxiv.org/abs/2505.23184) | LoRA experts are aligned/rotated before gated mixture. | Its rotations organize reusable experts; CHHT generates a fresh request/candidate rotation from ordered evidence.  A static rotated-expert mixture remains a plausible approximation control. |
| [HyperGrid](https://arxiv.org/abs/2007.05891), [HyperFormer](https://arxiv.org/abs/2106.04489), [PHA](https://arxiv.org/abs/2310.11670) | Hypernetworks generate or share Transformer weight modulation/adapters from task/input descriptors. | They establish that internal hypernetwork modulation is not novel by itself.  CHHT's claim can only rent on the triadic skew/Cayley constraint and its ranking evidence contract. |

## Recommendation and personalization neighbors

| Neighbor | Relevant mechanism | Boundary |
|---|---|---|
| [DIN](https://arxiv.org/abs/1706.06978), [ZAM](https://arxiv.org/abs/2004.07972), [TEM](https://arxiv.org/abs/2005.08936) | Target/query-aware attention pools behavior into a candidate score representation. | CHHT may use attention for encoding, but its falsifiable intervention is an internal FFN operator update.  The output-gate and mean-history controls test degeneration back to representation/score interaction. |
| [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690) | Transformer sequence encoders produce contextual user/history representations for recommendation. | They motivate the backbone family, not CHHT's functional update.  A fixed sequence representation plus scorer is covered by the mean-history/output controls. |
| [RAISE](https://arxiv.org/abs/2201.05333) | User-specific mixtures of multiple Q/K/V matrices adapt sequential recommendation. | This is a strong recommendation-specific neighbor.  Its user mixture is reusable/user-level; CHHT is candidate- and request-specific and identity at no history.  Removing candidate/query conditioning yields the preregistered history-only control. |
| [Adapter4Rec](https://arxiv.org/abs/2305.15036) | Adapter tuning transfers pretrained representations to recommendation tasks. | Static task adapters are ordinary PEFT; the static-LoRA control covers this boundary. |
| [ColdNAS](https://arxiv.org/abs/2306.03387) and [HyperRS](https://arxiv.org/abs/2307.14345) | Hypernetworks/meta-learning generate recommender parameters from user/item side information. | Dynamic recommendation weights are prior art.  CHHT cannot claim novelty from “hypernetwork for recommendation”; only its constrained ranking-time operator is under test. |
| [TALLRec](https://arxiv.org/abs/2305.00447), [A-LLMRec](https://arxiv.org/abs/2404.11343) | LLM/adapter-based recommendation and alignment. | These establish the LLM4Rec setting but do not by themselves instantiate the locked candidate/event Cayley update. |
| [OPPU](https://arxiv.org/abs/2402.04401) | Each user receives a personalized PEFT module learned from that user's behavior. | CHHT explicitly forbids permanent per-user modules and test-time optimization; its update disappears after a single `(q,c,H)` forward pass. |

## Verdict

`not-reducible-to-the-audited-diagonal/static/profile-neighbors`, with an
important reservation: NaRA's general dynamic dense core may subsume the
algebraic class, so the screen can support only the usefulness of CHHT's
skew/event-composition constraint—not a global novelty claim.  The proposal
must be marked `reducible` if candidate masking leaves the core unchanged, if
the history-only/static/output controls match it, or if an unconstrained dense
core later reproduces the effect without the Cayley constraint.

## Frozen degeneration ablations

1. diagonal rank gate (`U diag(g) V^T`): DISeL/Ouroboros-style degeneration;
2. ordinary static LoRA: no request, query, candidate, or event conditioning;
3. output history gate: history affects the score only after the internal map;
4. mean-history residual: pooled target/history representation without weight
   modulation;
5. history-only Cayley adapter: candidate and query removed from the generator;
6. no preservation loss: tests whether repeat fidelity is paid for by the
   training constraint rather than the operator alone.
