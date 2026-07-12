# Synthetic construct audit before lock

The C10 generator made its non-repeat positive the smallest variant unseen in
history, which created a candidate-local shortcut.  C11 samples every candidate
role's variant iid uniform **before** constructing history membership.  For an
exact request, history copies the already sampled positive item.  For a
non-repeat request, colliding history variants—not the candidate—receive a
non-zero uniform modular shift.  Candidate marginals therefore remain intact.

The registered audit uses 32,768 examples and seed `2026071191`.  Current
deterministic values are:

| audit | value | frozen maximum/requirement |
|---|---:|---:|
| target-position max deviation | 0.002537 | 0.010 |
| positive/negative variant TV | 0.011225 | 0.020 |
| positive/negative attribute×variant TV | 0.024631 | 0.040 |
| positive variant-0 rate | 0.058899 | — |
| hard-negative variant-0 rate | 0.062747 | difference ≤ 0.010 |
| repeat membership equals flag | true | true |
| ≥3 query-compatible hard negatives/request | true | true |

These checks concern only candidate-local construct validity.  The executable
gate separately rejects the whole run if a trained history-blind base exceeds
NDCG 0.80 on non-repeat requests.  This base-ceiling guard prevents a subtler
unregistered shortcut from being converted into a positive architecture claim.
