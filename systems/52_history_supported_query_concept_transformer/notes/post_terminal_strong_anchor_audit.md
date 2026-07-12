# C52 post-terminal strong-anchor audit

This diagnostic was run only after C52 had terminally failed.  It is not a
C52 rescue, model-selection result, or authorization to tune a coefficient.

The C47--C52 formulation gates used raw frozen-BGE cosine as their base because
they tested history information objects.  On the same 600 exposed Kuai C47-A
requests, the registered seed-20260708 D2p ranker reaches `0.603050` NDCG@10;
raw BGE reaches `0.300870`.  Therefore no C47--C52 standalone score is a viable
proposed ranker even when it improves over raw BGE.

As a fixed-coefficient complement diagnostic, each already-frozen history
correction was added unchanged to D2p, with no z-scoring, scale search, or
request selection.  Results were:

| correction added to D2p | NDCG@10 | delta vs D2p |
|---|---:|---:|
| none | 0.603050 | — |
| C52 nonlinear primary | 0.604415 | +0.001365 |
| C52 linearized reduction | 0.604962 | +0.001912 |
| C52 token softmax | 0.603539 | +0.000489 |
| C47 pooled plain KRR | 0.602668 | -0.000382 |
| C47 pooled softmax | 0.603517 | +0.000467 |
| C47 posterior | 0.602142 | -0.000908 |

For C52 primary versus D2p the paired interval was
`[-0.000269,+0.003511]`, one hash fold was negative, and primary-minus-wrong
was only `+0.000547` with a zero-crossing interval.  The linearized reduction
again exceeded the nonlinear primary.  Thus there is at most weak formulation
evidence that token-level history can complement the strong anchor; there is
no stable or C52-specific rent.

Binding consequence: every successor must report the registered strong base
before interpreting a raw-LM signal.  A new architecture must modify the
strong ranker's internal representation or demonstrate stable complementarity
to it; replacing it with raw BGE is no longer an admissible path.
