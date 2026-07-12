# C45 prefix-conditioned event innovation terminal

C45 tested whether a history event should be represented by the shared
recurrent-Transformer transition difference
`F(prefix,event)-F(prefix,NULL)`. Three GPU seeds and four equal-parameter modes
completed on a locked data-free latent-transition task.

The primary strongly learned the synthetic ranking task, but raw event tokens
were better in every seed and event reversal retained 61%--100% of the primary
gain. Thus causal prefix state and local NULL subtraction were not load-bearing.
The raw-event transition path was also gradient-inactive, exposing a frozen-D0
versus aggregate-checker mismatch; the independent D1 failure already makes the
candidate terminal.

This closes another representation transformation, not all event
representations. Combined with C43--C45, the next discriminating question is
whether frozen LM semantics omit the behavioral/collaborative relation needed
for cross-item personalization. That signal must be tested before wrapping it
in another attention geometry.
