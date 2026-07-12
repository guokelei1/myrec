# C55 signal outcome

Status: `failed_residual_signal_terminal`.  C55 is closed.  It used only a
label-isolated 80/20 split of the previously authorized C53 fit role.  C53-A,
reserve, dev, test, and qrels remained closed.

Both domains used request-standardized base logits and the exact zero-sum
probability residual `y-softmax(base)` as the training target.  Three paired
seeds trained equal-capacity history-carrier and raw-candidate models.

On Kuai, ensemble NDCG@10 was exactly `0.5797939822` for both base and primary.
Primary residual MSE improved over the zero predictor by only `0.5548%`, below
the frozen 1% floor, and wrong history had slightly lower MSE than true history.
The raw control also ranked nominally better.  One seed missed the frozen loss
trend, independently failing execution aggregation.

On Amazon, primary improved over zero MSE by `1.4565%`, but raw-candidate MSE
was significantly lower: raw-minus-primary was `-0.00004615`, 95% interval
`[-0.00006113,-0.00003095]`.  Primary NDCG gain over base was only
`+0.0002716`, interval `[-0.0004953,+0.0010245]`, and primary lost nominally
to raw by `-0.0007487`.  True and wrong histories were effectively tied.

Thus common score units remove the misleading weak-anchor overwrite seen in
C53/C54, but changing the loss does not expose stable user-specific residual
signal in frozen pooled LM states.  This closes pooled-state probability-
residual rescue.  The next candidate must change token representation
formation inside the Transformer and bind pooled/raw controls; it may not tune
residual scale, loss weight, epoch, seed, or domain.
