# B9 ProdSearch Adapter Protocol Decision

Date: 2026-07-10

Status: approved - Option A (minimal official adapter).

Authorization: user decision in chat on 2026-07-10. The authorization keeps
the frozen identity `official-code, adapter to KuaiSearch interface, not
externally aligned`; it permits only request-query transport and deterministic
data-format adaptation. It does not authorize catalog-wide pretraining or a
custom replacement trainer.

Scope: B9z ZAM and B9t TEM under
`doc/16_next_round_c3_router_neighbor_plan.md` Step 3. No test data or dev
qrels were read while producing this note.

## Why a decision is required

The locked ProdSearch source and the KuaiSearch standardized interface disagree
at three semantic boundaries that `doc/16` does not fully specify:

1. **Query identity.** The official test dataset obtains every query from
   `product_query_idx[prod_idx]`, and the train loader randomly chooses from the
   same product-level query list. KuaiSearch requires the current request's
   query. Leaving the official behavior unchanged would mix queries across
   requests and invalidate the query-conditioned claim. The minimal required
   patch is to carry and use each interaction's `review_query_idx` in both train
   and dev scoring.
2. **Multiple positives and frozen history.** KuaiSearch has 232,566 clicked
   candidate rows across 163,717 train requests, including multi-positive
   requests. The official format expects one target purchase in a sequential
   user history. A mapping must decide whether multiple clicked targets share a
   synthetic sequence or become separate examples.
3. **Variable candidate pools and cold products.** Official scoring assumes one
   fixed `test_candi_size`, while KuaiSearch dev candidate counts vary from 5 to
   1,500. In addition, only 46,459 of 396,822 unique dev candidates (11.71%) and
   126,620 of 575,609 dev candidate rows (22.00%) occur as clicked train targets.
   Under the unmodified official item-ID/PV training path, the remaining product
   embeddings receive no target-text training, even though their title, brand,
   and category are available.

These choices can materially change B9 NDCG and therefore the frozen
motivation wording branch. They must be fixed before looking at B9 dev metrics.

## Option A - Minimal official adapter (recommended)

- Patch only query transport: use the request-level query associated with each
  train/dev interaction instead of product-level query sampling. Save the
  upstream diff as `reports/b9_prodsearch_patch.diff`.
- Materialize one synthetic user/example per clicked train target. Copy only
  that request's frozen history into the synthetic sequence, so other positives
  from the request cannot enter its history.
- Hold out a deterministic train-only validation subset before training. Dev
  records remain label-free and are used only for final scoring.
- Pad each variable dev candidate pool to 1,500 with deterministic non-candidate
  products for the official scorer, then discard fillers and assert exact
  request/candidate equality before the shared evaluator.
- Feed `title + brand + category` as the official PV target text where an item
  is a clicked training target. Do not add catalog-wide pseudo-positives or a
  new text-pretraining stage. Report the 11.71% unique / 22.00% row train-target
  coverage as a cold-product limitation.
- If either model cannot pass the frozen internal-validity suite, especially
  significant improvement over Random, trigger `doc/16` Step 3 stop-loss and
  register it as `attempted, not runnable`. Do not repair it after seeing dev.

This option preserves the frozen identity
`official-code, adapter to KuaiSearch interface, not externally aligned`. Its
likely weakness is cold product coverage, but that weakness is measured rather
than hidden by a structural extension.

## Option B - Catalog-text pretraining extension

Before ranking training, add an unsupervised pass that trains every candidate
product embedding from `title + brand + category`, then run the official
ranking objective. This would make cold candidates text-aware, but it changes
the official training schedule and introduces a new training stage not covered
by `doc/16`. The identity would need to become a structural adaptation, and a
new budget/fairness rule would be required before execution.

## Option C - Custom direct runner

Reuse the official ZAM/TEM model classes but replace the official dataset,
dataloader, trainer, and ranklist path with project-native request batches. This
avoids candidate padding and makes the interface cleaner, but it weakens the
`official-code` claim more than Option A and creates a larger surface for silent
training differences.

## Decision

Option A is approved. Implementation must include assertions that deterministic
padding is removed before shared evaluation and that sibling positives from a
multi-positive request never enter one another's synthetic history. Options B
and C remain prohibited unless separately authorized.
