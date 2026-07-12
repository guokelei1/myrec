# C01--C61 architecture-intervention coverage after C61

Date: 2026-07-12
Status: pre-outcome audit for the next architecture; no new labels were read.

## Decision

The serial search is moving closer to the scientific problem, but not yet
toward a validated result.  C56--C61 localized the latest failure more sharply:
history can create candidate-dependent numerical activity, and a conservative
base interface can preserve most of the strong ranking, but the learned
adjacent-edge likelihood in C61 is too small and seed-dependent to cross any
real ranking margin.  This is not evidence for another scale, threshold,
neighbourhood, epoch, or seed rescue.

The larger risk is now **overfitting to the history of experiments**.  A new
formula around the same exposed score surface could look increasingly tailored
to KuaiSearch while leaving the information-flow problem unchanged.  The next
candidate must therefore move to an untested Transformer state lifecycle and
use the identical operator in both KuaiSearch and Amazon-C4.

## Covered intervention positions

| Transformer/ranker position | Main candidates | What is now closed |
|---|---|---|
| prompt/prefix and factual-minus-NULL logits | C04, C12, C45 | paired prefixes and event innovations do not by themselves produce reliable transferable evidence |
| history-conditioned weights/admission/normalization | C05, C07, C09, C14, C16, C24, C34--C35, C47--C48, C52, C54, C56--C59 | another attention gate, support mass, candidate budget, or semantic confidence law is not a new direction |
| history-conditioned values and residual writes | C06, C10--C17, C22--C23, C36--C45, C49--C50 | protected, counterfactual, predictive, innovation, and metric-coupled values did not pay stable rent over their nearest controls |
| candidate comparison/listwise graph | C24, C27--C30, C34--C44, C53--C61 | set attention, pair contests, margin-local edges, candidate flow, and adjacent-edge transport are covered; C61 cannot be rescued by changing edge scale or locality |
| request-shared query transport/subspaces | C31--C43, C47--C52 | this contains the strongest weak result (C32), but tangent projection and later support laws did not survive all controls/domains |
| dynamic adapters/internal function changes | C02, C06, C08, C22, C29, C40--C45 | rotations, reversible commutators, block protection, causal mediation, and tied metrics are either inactive or reducible/unsupported |
| pooled/prequential memory readouts | C46--C51 | fixed pooled semantic state, KRR, innovation/DeltaNet values, dual memory, and covariance statistics are closed |
| output/loss interface around a strong anchor | C28--C30, C52--C55, C59--C61 | free residual direction is unstable, direct semantic writes overwrite the base, and hard capacity protection can become rank-inert |

The table does not claim that every possible Transformer is falsified.  It says
that renaming one of these already tested intervention positions is no longer a
valid successor.

## Uncovered position

No candidate has isolated the following state lifecycle as its load-bearing
primitive:

1. history writes a small set of latent preference tokens **without seeing the
   current query or candidate set**;
2. that memory is then frozen for the request;
3. query-candidate tokens read the same memory, without being allowed to write
   back into it;
4. final listwise logits are produced by the Transformer candidate states.

C08 used a reversible write-probe-undo commutator, C45 used per-event
counterfactual innovations, and C49/C50 used fixed KRR/DeltaNet-style memory
readouts.  None tested a trainable history-only latent bottleneck that is
formed once and read many times inside the end-to-end ranking Transformer.

This position directly addresses the observed problem: candidate/query-
conditioned history attention can re-select a different noisy history for
every candidate, whereas a query-independent memory must first represent a
stable user state.  Multi-slot memory can retain several preferences, while
the later query-candidate read decides which slot is relevant.  The claim is
architectural d-separation, not that latent tokens or attention are new by
themselves.

## C62 design boundary

The proposed C62 primitive is a **write-once preference-memory Transformer**:

- a history encoder and learned latent slots form the memory in a history-only
  write phase;
- a block-sparse read phase lets query-candidate tokens attend to immutable
  slots, followed by candidate-set Transformer interaction;
- empty history is an exact base identity and repeat-present requests retain
  the registered item-only fallback;
- there is no dataset, category, query-type, or exposed-slate branch;
- the same width, slots, loss, and control definitions apply in KuaiSearch and
  Amazon-C4.

The binding nearest controls are a same-capacity direct history cross-attender,
a query-conditioned slot writer, a single pooled slot, and a history-free
capacity control.  C62 advances only if the multi-slot state is mechanically
load-bearing, true history differs from wrong history, and the primary pays
utility rent over the strong base and nearest controls in both domains.  A
generic latent-memory model that merely ties those controls is a rejection,
not an architecture win.

## Fresh evidence boundary

- KuaiSearch training may use the already exposed C26 fit labels.  C26
  internal-A (1,200 requests) is still label-closed and may be registered as a
  fresh C62 role only after the C62 proposal and execution locks.
- Amazon-C4 training may use the exposed C38 fit labels.  The 399-request C39
  reserve has never been feature-materialized, scored, or label-opened; a
  deterministic 300/99 split can provide C62 A/delayed roles.
- The first real gate is an exposed-fit, dual-domain falsifier.  Fresh labels
  remain closed unless structural and label-free activity gates pass.
- Dev, test, and qrels remain inaccessible.

This boundary prevents a succession of KuaiSearch-only repairs from being
mistaken for increasingly problem-aligned architecture design.
