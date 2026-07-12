# 2026-07-12 — C74 semantic-conservative relay

C74 introduced a two-hop query relay whose learned maps may route evidence but
may not rewrite raw LM semantic values or candidate energy.  Its fresh
data-free gate passed all three seeds with primary NDCG@10 `0.941--0.947`,
base gains `+0.226--+0.231`, and large margins over coupled, pooled, and
factual reductions.

The separately locked pretrained BGE probe passed G0 and completed twelve GPU
fits.  At label-free A0 it was unusually active and history-specific, changing
about 40% of Top-10 sets versus base and 35%--39% versus wrong history.  It
still closed because two seeds failed the all-mode loss-trend condition; no
validation labels opened.

One seed-report boolean incorrectly equated a loaded training-label container
with validation-label access.  The scoring call path remained label-free and
the independent loss failure makes the defect non-decision-changing.

A label-free final-versus-initial audit found substantial carrier drift after
unfreezing BGE's final layers (history/candidate cosine as low as `0.786/0.792`).
The next proposal may therefore freeze the pretrained LM carrier and learn only
the token-level routing operator.  That is a new architectural constraint, not
a C74 continuation; it requires its own ID, source tree, fixed-anchor training
check, lock, and matched controls.
