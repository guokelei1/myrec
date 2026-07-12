# C48 nearest-neighbor audit

| Neighbor | Overlap | Binding boundary |
|---|---|---|
| Cubit | KRR replaces attention token mixing | plain KRR must be beaten directly |
| Cog Attention | signed weights normalized by absolute mass | signed-L1 control must be beaten directly |
| Differential/negative attention | positive and negative maps cancel noise | C48 must pay rent without a learned second attention map |
| PolaFormer | explicitly retains same/opposite-polarity interactions | polarity handling itself is not novel |
| C47 | same history normal equation | candidate self-support is removed and cannot be retuned |

Primary sources:

- https://arxiv.org/abs/2605.06501
- https://arxiv.org/abs/2411.07176
- https://openreview.net/forum?id=2RDd8vpzrl
- https://openreview.net/forum?id=kN6MFmKUSK
