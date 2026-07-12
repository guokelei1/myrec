# C48 reduction audit

The exact KRR mean `e_c` is Cubit/plain KRR and is a mandatory control.  The
ratio `e_c / sum|z_jc|` resembles absolute-normalized signed attention (Cog
Attention and polarity-aware attention) and is a mandatory direct control.
C48's output is instead `e_c |e_c| / sum|z_jc|`: it preserves KRR magnitude
only to the degree supported by sign consensus.

Algebraically this can be written as a deterministic candidate gate times KRR,
so gating itself is not novel.  The only potentially distinct claim is that
the *exact dual KRR influence decomposition* supplies a fidelity statistic that
pays incremental ranking rent over KRR and signed attention.  If it does not,
C48 reduces to known ingredients and closes.

Pre-outcome novelty status: `distinct-combination-with-high-uncertainty`.
