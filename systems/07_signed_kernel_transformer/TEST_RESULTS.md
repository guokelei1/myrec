# CPU Prototype Verification

Date: 2026-07-11
Device: CPU only
PyTorch: 2.12.1+cu130 (CUDA not invoked)

Command:

```bash
python -m pytest -q \
  systems/07_signed_kernel_transformer/tests/test_signed_kernel_transformer.py
```

Result:

```text
15 passed, 5 warnings in 2.61s
```

The five warnings are the same PyTorch informational warning that nested-tensor
fast paths are disabled for `TransformerEncoder(norm_first=True)`.  They do not
indicate test failures or GPU use.

Covered contracts:

- hand-computed three-candidate signed weights;
- open dead-zone abstention;
- candidate conservation and common-mode invariance;
- no-history exact fallback;
- candidate/history permutation behavior;
- `tau=0` centered-attention degeneration;
- non-factorization witness against a scalar-gated centered direction;
- declared two-candidate degeneration;
- float64 gradcheck and active gradients;
- end-to-end masked-information barrier and absence of label/qrels API;
- end-to-end no-history equality;
- end-to-end permutation equivariance;
- exact-recurrence activation;
- gradients to query, history, and candidate evidence;
- fail-closed invalid normalization parameters.

This verifies G0 structure only.  `pre_outcome_gate.md` G1 has not been run.
