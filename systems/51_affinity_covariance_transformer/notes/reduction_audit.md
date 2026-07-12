# C51 reduction audit

`cov(Hq,Hc)` is a centered covariance readout.  Covariance attention is not a
new family: self-attention has a covariance-limit interpretation, XCiT uses
cross-covariance attention, and correlated attention uses feature/time-series
cross-correlations.  Pearson normalization is also the one-dimensional form
of centered alignment and is a direct control.

C51's only potentially distinct claim is the PPS information object: query and
candidate affinity profiles are paired over a single user's history-event
axis, so common user semantic affinity cancels before ranking.  If centered
covariance does not pay rent over the uncentered product and standard history
mixers, the candidate has no novelty or utility claim.

Verdict: `boundary-only-with-high-uncertainty`.
