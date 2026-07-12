# C19 nearest-neighbour audit

The search was completed before any C19 learned outcome.

| Neighbour | Source | Boundary |
|---|---|---|
| induction heads | [Olsson et al., 2022](https://arxiv.org/abs/2209.11895) | match a previous token/context and copy its successor; this is C19's `F` control, not the reverse-subtracted cofactor |
| selective induction heads | [D'Angelo, Croce, and Flammarion, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d7ed243b13831bdd468f35039936bcef-Abstract-Conference.html) | learn which causal lag to copy; they motivate the free/forward controls and make a generic induction novelty claim unavailable |
| kernel view of attention | [Tsai et al., EMNLP 2019](https://aclanthology.org/D19-1006/) | covers attention as kernel smoothing and kernel composition; OLT's claim must rest on the tied diagonal/skew temporal law, not on using a kernel |
| Differential Transformer | [Ye et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) | subtracts attention maps; the free signed-lag control tests whether OLT is merely a fixed two-map difference |
| structured directional kernels | general symmetric/antisymmetric kernel decomposition | establishes high prior-art risk; C19 does not claim to invent antisymmetric bilinear forms |
| C03 triadic transport | `../../03_triadic_transport_transformer/notes/final_report.md` | intersects same-event query/history/candidate transport plus query-candidate mass; OLT uses off-diagonal temporal orientation and no transport solver |
| C06 wedge flow | `../../06_conservative_wedge_flow_transformer/notes/mechanism_fingerprint.md` | skews candidate pairs within each event and Hodge-projects divergence; OLT skews adjacent time positions for each candidate and directly tests transition direction |
| C07 signed kernel | `../../07_signed_kernel_transformer/mechanism_fingerprint.md` | candidate-axis dead-zone competition; OLT has no candidate contest/dead zone and signs only temporal orientation |

Pre-outcome verdict: **distinct from closed local candidates; high global
nearest-neighbour risk**.  Synthetic advancement requires OLT to beat forward,
symmetric and diagonal controls and remain competitive with the free signed-lag
control.  A tie with free signed lag blocks a strong performance-innovation
claim and must be carried into any real gate.
