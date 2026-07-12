# C51 nearest neighbors

| Neighbor | Overlap | Boundary |
|---|---|---|
| Self-attention as covariance readout | context covariance read | C51 uses per-user query/candidate affinity profiles |
| XCiT | cross-covariance attention | feature-axis covariance itself is not novel |
| Correlated Attention | cross-correlation mixer | Pearson and uncentered controls are binding |
| CKA/HSIC | centered normalized alignment | Pearson-style control must not win |
| C47/Cubit | semantic history token mixer | plain KRR and softmax must be beaten |

Primary sources:

- https://arxiv.org/abs/2605.10466
- https://openreview.net/forum?id=kzPtpIpF8o
- https://arxiv.org/abs/2311.11959
