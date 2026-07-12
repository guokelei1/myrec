# C48 v2 formulation execution abort

The v2 command again stopped on the first request before label loading,
metrics, bootstrap, output, or fresh-role access.  That request had a
length-one history.  NumPy reports some length-one negative-stride views as
logically C-contiguous, so `np.ascontiguousarray` returned the original view
instead of copying it; `torch.from_numpy` therefore rejected it.

The only authorized v3 repair is an unconditional `np.array(..., copy=True,
order="C")` at the NumPy-to-Torch boundary, with a regression fixture whose
reversed dimension has length one.  All scientific settings remain unchanged.
