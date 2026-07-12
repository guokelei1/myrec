# 2026-07-12 — C42/C43 function alignment and catalog coverage audit

This post-terminal audit used only already-open C42/C43/C47 roles and their
frozen feature/score artifacts. It opened no reserve, dev, test, or qrel and is
not a rescue or a new result selection.

## Corrected C42/C43 diagnosis

For every request, the audit measured raw LM query-to-history cosine and the
within-candidate Pearson correlation between true- and wrong-history
corrections, then averaged the three frozen checkpoints.

| diagnostic | Amazon C42 | KuaiSearch C43 |
|---|---:|---:|
| mean history length | 30.46 | 7.45 |
| mean true query/history cosine | 0.4873 | 0.4111 |
| mean wrong query/history cosine | 0.4588 | 0.3239 |
| true-minus-wrong mean-cosine gap | +0.0285 | +0.0872 |
| true-minus-wrong max-cosine gap | +0.0488 | +0.1088 |
| mean true/wrong correction correlation | 0.9680 | 0.4256 |
| median true/wrong correction correlation | 0.9919 | 0.4975 |
| registered true-minus-wrong NDCG@10 | +0.035234, CI positive | +0.000487, CI crosses zero |

The prior shorthand that C43 mainly ignored history identity is therefore too
strong. C43 changes its candidate function substantially when the donor
history changes, and the true history is much more query-related than the
wrong history. Those changes are simply not relevance-aligned. Conversely,
C42 obtains useful Amazon differences even though most of the true and wrong
correction function is shared.

The remaining problem is **behavioral direction**, not more history
sensitivity. Another mask, reference state, authentication bit, or
true/wrong activity threshold cannot make a direction useful.

## Catalog-ID coverage

A label-free scan used the fixed 6,000 C47 fit histories and the already-open
C47-A candidate pools. Direct item-transition modeling has a severe
open-catalog boundary:

| coverage object | KuaiSearch | Amazon-C4 |
|---|---:|---:|
| A candidate rows found in fit-history item vocabulary | 3.62% | 1.53% |
| A requests with any such candidate | 45.33% | 71.00% |
| A candidate rows found anywhere in fit candidate/history vocabulary | 22.34% | 62.04% |
| A candidate rows seen as a positive fit target | 1.71% | 0.88% |

An item-ID transition expert would therefore have sharply different and mostly
cold surfaces across the two domains. It may remain a baseline/control, but it
is not an admissible cross-domain proposed primitive.

## Next signal prerequisite

C46 trained behavioral sequence prediction on content states, but random
shuffling left an easy ordinary-semantic shortcut: the trained Transformer
beat shuffled pairing while tying frozen semantic mean. The next signal probe
must instead learn an open-catalog item relation from behavioral pairs against
**semantic-matched cross-user negatives**. It advances only if the learned
relation beats ordinary semantic attention, a random-negative model, and wrong
history on both domains. This is a signal gate, not yet an architecture claim;
contrastive item-language recommendation is established prior art.
