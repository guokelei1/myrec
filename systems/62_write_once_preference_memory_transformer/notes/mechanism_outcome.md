# C62 mechanism outcome

Status: **failed terminal at data-free synthetic G0**.  No repository record,
fit label, fresh feature, fresh label, dev, test, or qrel was opened.

The intended state lifecycle was implemented correctly.  Across all three GPU
seeds, primary memory was exactly invariant to query substitution, empty
history and repeat fallbacks had zero error, deterministic error was zero,
candidate-permutation error was at most `5.36e-7`, all four modes had 316,704
parameters, and ranking gradients reached the history encoder, slot writer,
memory reader, candidate-set Transformer, and score head.

The mechanism itself failed.  On the planted two-interest task, primary clean
accuracy was only `0.4844/0.4902/0.5156` against the frozen `0.85` threshold.
Its margin over the same-parameter single-slot reduction was
`+0.0137/-0.0254/+0.0039`, never approaching the required `+0.10`.
Wrong-history accuracy drops were `0.0000/+0.0039/-0.0059`, and wrong history
changed only `0.98%--1.37%` of orders.  Slot variance was merely
`4.42e-5--6.15e-5`: numerically distinct learned seeds survived, but their
content was effectively pooled.

The failure is therefore not an inactive optimizer, broken fallback, or output
scale problem.  Standard per-slot softmax cross-attention does not create the
event-to-slot competition needed to bind multiple independently addressable
preferences.  C62 may not be rescued by more slots, width, depth, epochs,
learning rate, seed selection, or real data.  A successor must either introduce
a genuinely different binding law, with standard slot attention as a binding
control, or leave latent-memory architecture entirely.
