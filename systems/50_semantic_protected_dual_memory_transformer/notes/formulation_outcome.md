# C50 exposed formulation outcome

All six zero-step rescoring runs passed exact candidate equivariance,
determinism, no-history zero, and the orthogonality invariant (maximum absolute
inner product 8.20e-8).  The utility gate failed on both exposed domains.

| domain | primary | base | raw semantic | unprojected sum | best C47 fixed |
|---|---:|---:|---:|---:|---:|
| KuaiSearch | 0.307006 | 0.300870 | 0.308307 | 0.307307 | plain 0.310208 |
| Amazon-C4 | 0.234996 | 0.253202 | 0.274713 | 0.240648 | posterior 0.277001 |

Orthogonalization improved over innovation alone on Amazon but remained far
below raw semantic memory and even below base.  It also reduced the Kuai raw
memory result.  A vector-space invariant therefore did not make the behavioral
values relevant.

Decision: `failed_formulation_terminal`.  Close C50 before training or fresh
reserve; do not tune projection strength, scale, or a raw/innovation mixture.
