# C21 train-only path-signal outcome

Status: **failed; terminal stop after the single locked attempt**.

The hash-locked run completed on physical A40 GPU 1 in 535.75 seconds.  All 15
fits (three seeds × five exactly parameter-matched modes) completed two epochs
with finite losses and gradients.  Candidate-set hashes, deterministic rescore,
candidate centring, query-absent fallback and all 512 no-history rows passed.
The failure is therefore scientific, not an inactive write, numeric error or
protocol violation.

## Primary result

| mode | seed-averaged train-internal NDCG@10 | delta vs frozen D2p |
|---|---:|---:|
| D2p | 0.6132530 | — |
| contiguous path | 0.6131887 | -0.0000643 |
| one step | 0.6138159 | +0.0005629 |
| unordered pair | 0.6133500 | +0.0000969 |
| endpoint only | 0.6133421 | +0.0000890 |
| pooled history | 0.6139342 | +0.0006812 |

The primary-minus-D2p paired 95% CI was
`[-0.0007363, +0.0006018]`; seed deltas were `-0.0006026`, `+0.0005360` and
`-0.0001264`.  It failed the minimum effect, CI, all-seed and hash-fold rules.
More decisively, the primary lost to `one_step` by `-0.0006272`, with paired
95% CI `[-0.0012506, -0.0000306]`, and lost in every seed.  It also failed every
other matched-control requirement.

The operator was load-bearing: seed-averaged scores changed some order on
46.03% of requests and top-10 membership on 4.67%; maximum candidate-centre
error was `9.54e-7`.  But clicked-minus-unclicked delta was only `+0.0002153`
with CI `[-0.0000938, +0.0005211]`.  Wrong-history and shuffled-event scores did
not collapse.  Because clean gain was negative, corruption retention is
undefined and fails by construction; both corruptions had numerically positive
mean gain over D2p (`+0.0001247` and `+0.0002112`).

## Interpretation

Short contiguous path closure is not an observable ranking law in these frozen
real states.  Adding multi-step segments accumulates history directions that
are not label-aligned; an adjacent-step control is significantly better, while
an order-free pooled control has the best point estimate.  Post-outcome
diagnostics show that neither apparent control gain has a positive paired CI,
so C21 does not establish a replacement positive signal.

C21 is closed and must not be tuned, retried, restricted to favourable history
lengths or promoted to a Transformer.  The result instead strengthens a common
portfolio conclusion: beyond exact recurrence, current frozen-state history
writes alter rankings without evidence-specific direction.  The next design
must change the representation/learning interface or make reliable recurrence
itself load-bearing; another geometric aggregation of the same frozen states is
not justified.

Raw artifacts:

- gate report SHA-256:
  `011a2c99428e7ef33a5e7e8677075f16f9133194bfe9fd2dcb3cc8058bb1ab58`;
- formal attempt SHA-256:
  `7ea60c6961f8f382e4244c5e8392e44ac965dfdd27c2fdb60bcf1f947543bd48`;
- proposal-lock SHA-256:
  `de0d203e78f6d3778d109fea2cfa8708a2cb87724c989f3a3c03b5d2c748188c`.

No original train-label array, C06 delayed role, dev/test record, qrel or shared
dev evaluator was opened.  `reports/dev_eval_log.jsonl` remained unchanged.
