# 2026-07-12 — C48 signed influence consensus terminal

C48 was a deliberately cheap response to C47: replace candidate span support
with the cancellation ratio of the exact signed eventwise KRR decomposition.
The proposal explicitly treated Cubit, Cog/negative attention, plain KRR, and
softmax as nearest controls and used only already-open C47 A as a formulation
surface.

The operator was active, finite, deterministic, permutation-invariant,
contractive, and exact-zero without history.  It produced the best fixed point
estimate among C47/C48 operators on KuaiSearch (0.310643), with positive
clicked correction and clicked true-minus-wrong intervals.  However, it beat
plain KRR by only 0.000481 with a zero-crossing interval and one negative fold;
true-minus-wrong NDCG also crossed zero.  On Amazon it beat base and wrong
history but lost to softmax by 0.005874.

The diagnostic explanation is structural: mean coherence was 0.9815 on Kuai
and 0.8849 on Amazon.  The KRR event influences rarely cancel, so their sign
agreement cannot certify relevance.  C48 therefore closes before fresh data.
Together C47/C48 rule out both obvious confidence statistics derived from the
same frozen semantic normal equation: span membership and signed
decomposition agreement.  A successor must change the learned value
representation or memory target, not add another scalar contraction of KRR.
