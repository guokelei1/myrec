# Nearest-neighbor audit

Status: completed before C23 implementation outcome.  No global novelty claim
is made.

| Family | Closest behavior | Non-reducible C23 difference | Locked ablation |
|---|---|---|---|
| DIN / target attention | candidate weights all history events | RRST deletes every pre-last-recurrence edge and evolves an explicit reset state through the remaining ordered suffix | `unreset_history` |
| DIEN/TIEN target-aware GRU | target conditions interest evolution across the sequence | RRST's state origin and positional coordinate are deterministically reset by the candidate's last exact identity; it is not a target gate over a common user state | `unreset_history`, `query_independent` |
| RepeatNet | learned repeat/explore mixture | RRST has no repeat/explore scorer or router; the same listwise Transformer produces a bounded correction inside the repeat ranking | static item-only and D2p comparisons |
| TSRec | explicit user/item repeat intervals and repeat sequences | RRST tests post-last-recurrence displacement within a query-conditioned candidate-local attention graph, without item/user-specific interval tables | `orderless_suffix` |
| DeltaNet / Gated DeltaNet / SinkRec | recurrent fast-weight delta updates, including memory-conditioned recommendation | RRST changes the token graph at a candidate identity boundary and uses ordinary causal self-attention over a finite suffix; there is no matrix delta rule or semantic-memory loop | `unreset_history`; algebra audit |
| AC-TSR | position/noise calibration of attention | RRST uses a hard candidate-specific origin and excludes the prefix exactly; it is not a learned multiplicative attention calibrator | pre-anchor invariance |
| SASRec/BERT4Rec + query | common sequence representation followed by target scoring | RRST constructs a different lawful sequence and origin for each candidate | `unreset_history` |

Primary sources checked before coding:

- RepeatNet: https://arxiv.org/abs/1812.02646
- TSRec: https://arxiv.org/abs/2506.08531
- AC-TSR: https://arxiv.org/abs/2308.09419
- DeltaNet: https://arxiv.org/abs/2406.06484
- Gated DeltaNet: https://openreview.net/forum?id=r8H7xhYPwz
- SinkRec: https://arxiv.org/abs/2606.09888
- TIEN: https://openreview.net/forum?id=R39h0M8mTI

Verdict: `distinct-but-narrow`.  Last-exact-identity reset plus post-anchor
Transformer evolution was not found as the load-bearing operator in these
neighbors.  The ingredients are known; empirical rent over the matched
unreset/orderless/query-independent controls is mandatory.
