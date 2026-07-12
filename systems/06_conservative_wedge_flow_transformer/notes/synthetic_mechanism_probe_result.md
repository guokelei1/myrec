# C06 bidirectional synthetic probe result

Status: **PASS — conditional synthetic mechanism contract only**.

The one locked CPU execution used 12,288 requests per world, three generator
seeds, FP64 arithmetic and 10,000 paired bootstrap samples. It read no
repository data, qrels, dev/test records, checkpoints or prior scores. The
locked four-file manifest was `57e466...df5`; the lock was `7529a2...60b`; the
raw report is `897950...b53`.

## Binding results

| World / comparison | Pairwise delta | 95% CI | NDCG@10 delta |
|---|---:|---:|---:|
| aligned: local - `t=1` | +0.013409 | [+0.012739, +0.014074] | +0.017235 |
| aligned: local - global event gate | +0.013483 | [+0.012934, +0.014032] | +0.016768 |
| decoupled: local - `t=1` | -0.027713 | [-0.028398, -0.027026] | -0.034861 |
| adversarial: local - `t=1` | -0.055774 | [-0.056509, -0.055054] | -0.072751 |
| adversarial: direct oracle - local | +0.081019 | [+0.079769, +0.082246] | +0.104864 |

Every seed had the required direction. Generator integrity passed: skew error
was exactly zero, maximum cycle-row divergence was `1.51e-14`, Hodge recovery
error was `1.78e-15`, variance multisets were bit-identical, and the aligned /
decoupled / adversarial mean Spearman correlations were `1.0`, `0.000684`, and
`-1.0`.

## Interpretation

The candidate-local gate is load-bearing in the synthetic operator: it beats
both no trust and a global event gate when local cycle energy is correctly
coupled to potential error. It is not generic shrinkage—the gain disappears
and becomes harm when that relationship is removed, and becomes still more
harmful when reversed.

That negative-side behavior is the main design feedback. C06 must not assume
that cycle energy is reliability on real histories. Before any data claim, it
must beat a direct learned candidate gate and ordinary centered attention, and
wrong-history/query-mask/event-replacement controls must show that its trust is
authentic. Passing this probe authorizes design of those controls only; it does
not authorize cohort materialization, GPU use, train-internal fitting, dev or
test.
