# D2 Internal Popularity Leakage Fix

The first D2 alpha-calibration pass incorrectly loaded
`item_log_click_full_train.npy`. That array is legal for final dev scoring, but
it contains clicks from the held-out 10% internal validation segment and is
therefore illegal for selecting D2p alpha.

The affected alpha result (selected alpha 0.3, apparent NDCG@10 0.6067) is
invalid and retained in the calibration summary under
`invalid_alpha_calibration`. D2t epoch selection is unaffected because it does
not use popularity: epoch 3 remains selected at internal NDCG@10 0.3050 under
the locked min-delta rule.

Before any D2 dev scoring/evaluation, the code was corrected to use
`item_log_click_internal_train.npy`. Only alpha is recalculated from the saved
epoch-3 checkpoint. D2t is not retrained, the alpha grid is unchanged, and no
dev/test qrels are read. Final D2 scoring will continue to use full-train
counts, which is the legal training scope once dev is the evaluation split.
