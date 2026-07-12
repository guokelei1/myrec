# 2026-07-12 — C70 logged-choice architecture data gate

C69's negative result suggests that positive-only item sequences do not contain
the signed relevance direction needed by PPS. C70 therefore formulates a
Transformer whose fast-weight history write is the gradient of a user's past
choice against the alternatives actually shown under that past query.

This is a problem-aligned architecture hypothesis, but its input prerequisite
is currently asymmetric. KuaiSearch can recover 147,405 unique historical
choice episodes and covers 97.26% of history-present train requests. Amazon-C4
recovers none because the standardized target set does not contain the users'
prior purchase requests/slates. The local JDsearch release is only a sample,
and the public schema contains historical query/positive-item pairs but not
historical candidate slates.

The data gate therefore rejects implementation and GPU use. Running C70 only
on KuaiSearch, or synthesizing missing Amazon negatives from categories or
nearest titles, would make the architecture increasingly fit the dataset and
would recreate the failed C69 negative-construction premise. The next action
requires either a second real logged-choice dataset/interface or an explicit
scope decision to study KuaiSearch-only logged-choice PPS.
