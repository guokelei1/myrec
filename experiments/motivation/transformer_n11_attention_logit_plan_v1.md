# N11 scaled-QK-logit operator plan

Status: preregistered before N11 outcomes. This is a diagnostic extension after
N8--N10; it does not modify their frozen manifests or authorize an architecture
change.

## Question

The existing attention-edge probes remove history keys or values. They cannot
distinguish a failure caused by the pre-softmax QK score geometry from a failure
caused by edge availability. N11 changes only the complete pre-mask scaled QK
logits at fixed readout rows, holding Q/K/V, RoPE phase, causal/additive mask,
and output projection fixed.

## Fixed grid

- Q2 and Q3, all 8,000 internal-dev requests, frozen content-neutral eligibility;
- blocks 13, 20, and 27, with no layer/head selection from outcomes;
- full and neutral-history paths, preserving token count, positions, masks, and
  candidate order;
- identity, half-scale, double-scale, and sign-flip logit operators;
- qrels-blind resumable bundles and one shared evaluator after integrity checks.

The primary contrasts are operator-minus-baseline on the full path, the same
contrast on the null path, and the change in the full-minus-null transfer gap.
The registered controls are full/null identity and native/manual attention
recomposition error. Any identity or path-contract failure is a mechanical
non-result.

## Interpretation boundary

An effect is only an operator-level diagnostic. It cannot identify a particular
head, layer, temperature, or new attention architecture. A useful signal must
survive both models, the fixed strict-transfer surface, normalized-query
cluster inference, and the predeclared scale/sign controls; otherwise the result
remains unresolved.

## Execution

`scripts/run_deep_dive_next_wave_n11_attention_logit_queue.sh` waits for the N8,
N9, and N10 evaluator sentinels. Its first wave uses all four physical GPUs for
Q2 b13/b20 and Q3 b13/b20, then reuses two cards for b27. It writes six
independent bundles and evaluates them with
`scripts/evaluate_deep_dive_attention_logits.py`.

