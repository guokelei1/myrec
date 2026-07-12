# C33 fresh paired confirmation outcome

C33 terminated at A1 under its frozen rule.  Proposal and execution locks
remain valid; the authoritative report is
`artifacts/c33_tangent_transport_confirmatory_comparison/train_gate_v1/train_gate_report.json`
with SHA-256
`0df8191071f0cffc951f60d644d7fd057dac2bab9153dab59cfe37ee14a81cdd`.

G0 passed all five authentication checks.  A0 then passed all 25 checks for
both modes: the three tangent/control pairs had identical initial states and
16,384 trainable parameters, both adapter tensors received gradients,
candidate permutation and repeated execution were exact, and all repeat,
no-history, no-authentication, and query-absent fallbacks held.  Tangent changed
33.0% of complete orders and 4.67% of top-10 sets relative to unprojected
transport, so the comparison was mechanically active.

On the fresh 600-request A cohort, tangent minus D2p was +0.002988 NDCG@10.
All three seeds (+0.002910/+0.002861/+0.003192) and all three fixed folds
(+0.000219/+0.001869/+0.007294) were positive, but the paired 95% bootstrap
interval [-0.001361,+0.007181] crossed zero.  Tangent minus the matched
unprojected control was +0.000583; all seeds were positive, but its interval
[-0.000800,+0.002100] crossed zero and fold 0 was -0.000537.  True minus wrong
history also crossed zero.  Thus the direction replicated beyond C32's cohort,
but neither absolute certainty nor tangent-specific architecture rent was
established.

Interpretation: C32/C33 are problem-aligned rather than dataset-branch tuning,
and the positive transport direction is not confined to one selected cohort.
However, both experiments are still KuaiSearch-only, and a request-global query
move supplies weak, heterogeneous candidate direction.  The clicked-minus-
unclicked correction remained slightly negative and uncertain.  C33 therefore
does not authorize delayed-B, escrow, dev, test, projection-strength tuning, or
another tangent rescue.  The tangent line closes here.  A successor must use a
new candidate ID and untouched cohort and must alter candidate-level
Transformer information flow rather than rescale or relocate this query-global
transport.
