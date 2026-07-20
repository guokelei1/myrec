# N30--N34 Transformer follow-on operator plan (pre-registered, inactive)

This plan closes the remaining implementation-level causal debts in the
Transformer inventory. It is a frozen diagnostic extension, not a proposal
for a transfer architecture. It uses the existing standardized dev records,
Q2/Q3 frozen checkpoints, shared scorer/evaluator, qrels-blind bundles, and
the fixed functional blocks 13, 20, and 27. No layer, head, neuron, token, or
seed may be selected after observing an effect.

## Why this extension is needed

N25--N29 isolate SwiGLU formation, final readout, mask/softmax topology,
complete pre-mask QK formation, and attention--MLP interaction. The inventory
still has five independent operator debts that cannot be inferred from those
results: input embedding lookup; variance/gain in the two block RMSNorms; the
two residual addition rules; GQA query-to-shared-KV assignment; and the complete
Q3 q/v LoRA adapter contributions.

## N30: token embedding interface

Patch only the embedding rows corresponding to predeclared query, history, and
candidate spans. Token IDs, position IDs, attention masks, sequence lengths,
and all Transformer weights remain native. Same-token re-add identity is
mandatory, followed by zero/scale/sign and output-norm-matched random controls.
The full/null/wrong-user and reverse-removal contrasts must be replicated in
Q2 and Q3; tied lm-head observations cannot be reused as embedding causality.

## N31: input and post-attention RMSNorm operators

At every fixed block, intervene separately on RMS variance rescaling and
learned gain for `input_layernorm` and `post_attention_layernorm`. The incoming
residual, attention/MLP weights, and downstream readout stay fixed. Each cell
must include exact recomposition identity, variance-only versus gain-only
separation, norm-matched direction control, and same/wrong-history specificity.
A state patch or pre/post geometry is not an RMSNorm operator result.

## N32: residual addition rules

Capture the incoming residual and native attention or MLP increment, then change
only the coefficient used by the residual addition. The native coefficient-one
path must be an exact identity; zero, half, double, sign, and matched-random
increment controls are required. The estimator reports whether the addition
rule itself, rather than increment formation, is necessary for the transfer gap.

## N33: GQA query-to-KV grouping

Keep Q, K, V values, head count, mask, softmax, and output projection fixed
while replacing only the mapping from query heads to shared KV groups. Use a
predeclared group permutation and within-group rotation, with same-group
identity and reverse-removal controls. Interpret this as a GQA topology
diagnostic only after Q2/Q3 replication and exact tensor audits.

## N34: Q3 adapter contribution (integration of N19)

The existing N19 scoring/runtime/evaluator already isolates the complete scaled
adapter term `(alpha/r) B(A(x))` in Q3 q- and v-projection separately.  N34
therefore first audits and integrates that bundle; it must not duplicate the
expensive sweep.  Only if the N19 contract leaves a specific unresolved
operator cell may the smallest predeclared replication be launched.  Base
projection, adapter input, q/k normalization, RoPE, mask, and downstream score
remain native.  Re-add identity, zero/scale/sign, norm-matched random,
full/null/wrong-user, reverse-removal, and shared-prompt identity remain the
required gates.  This is an inference diagnostic and must not be presented as
a training method.

## Four-card schedule and stopping rule

After N25--N29 and the component-necessity gate close, run wave A with Q2/Q3
embedding and RMSNorm lanes. Run wave B with Q2/Q3 residual-addition and GQA
lanes. Run wave C with two independent Q3 q/v adapter lanes while the other
two cards run predeclared Q2/Q3 confirmation cells. Every lane has an
independent resumable directory and the existing four-hour wall boundary.

Stop only after the H0--H5 matrix records supported, contradicted, and
unresolved interfaces. Do not promote a design, open source test, switch the
dataset, or expand a family based on observed outcomes.
