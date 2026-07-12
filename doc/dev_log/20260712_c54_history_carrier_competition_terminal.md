# 2026-07-12 — C54 history-carrier competition terminal

C54 responded to C53's history-free list shortcut by restricting candidate
self-attention values to factual-minus-null history carriers.  D0 proved the
restriction exactly and all six GPU fits reduced mean epoch loss.  The
mechanism reduced Kuai true/wrong correction correlation materially, but its
null contrast, candidate edges, and wrong-history intervention did not
consistently affect Top-10 under strong D2p.  C54 failed before labels.

The two-domain asymmetry is diagnostic rather than supportive.  Amazon's raw
BGE base has score standard deviation 0.087 while learned correction standard
deviation exceeded 1.28, so structural activity there largely measures base
overwrite.  Kuai correction magnitude was not small, but correlated about
0.85--0.87 with D2p and mostly reinforced its existing order.  The next round
must not interpret weak-anchor activity as cross-domain generality.
