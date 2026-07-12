# Amazon full-token attention-edge attribution outcome

Date: 2026-07-12
Status: completed post-outcome attribution; C76 formulation evidence only

The three frozen checkpoints from Amazon token HSO were rescored on the same
1,200 already-open reserve requests under six preregistered layer-shared
attention masks.  No weight, checkpoint, history donor, candidate set, or
label changed.  The execution lock was
`427303b810e8b1238d3de6e4027d16861ae8f6c58187a0b44c4a51d8298b1dda`.

All three seeds passed deterministic and candidate-permutation checks exactly.
Isolating every history token from query/candidate tokens reproduced the
original null score with maximum absolute error zero, validating the edge-mask
intervention.

The original ensemble true-minus-null NDCG@10 effect was `+0.025298`.

| intervention | masked true-minus-null | retention | true-minus-wrong | classification |
|---|---:|---:|---:|---|
| isolate history | `0.000000` | `0.0%` | n/a | mechanical null |
| remove both Q-H directions | `+0.009520` | `37.6%` | `+0.028972` | partial |
| remove both C-H directions | `-0.037232` | `-147.2%` | `-0.006408` | destroyed |
| prevent Q from reading H | `+0.016037` | `63.4%` | `+0.035125` | partial |
| prevent C from reading H | `-0.004610` | `-18.2%` | `+0.012359` | destroyed |
| prevent H from reading Q/C | `-0.028696` | `-113.4%` | `+0.009687` | destroyed |

The next architecture cannot be a pooled user vector, query-only relay, or
single-direction target-attention law.  Candidate tokens reading raw history
tokens is the strongest load-bearing edge, but useful history tokens must also
be contextualized by query/candidate tokens and Q-H edges contribute material
utility.  The supported object is therefore multi-layer, bidirectional joint
token contextualization.

This is not fresh generalization evidence: it reuses the reserve whose labels
were opened by the HSO outcome.  It fixes the representation graph for C76.
The new primitive must preserve this full token graph while structurally
protecting the query-candidate base and isolating history-induced computation
inside the Transformer.  A final logit/state subtraction (C04/C65-C66), a
query relay (C73-C75), and any pooled/item-level transport are binding nearest
controls.

Authoritative report:
`reports/pps_amazon_token_edge_attribution_v1.json`.
