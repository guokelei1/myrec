# C42 weights-preserving confirmation outcome

C42 closed at A1 under its frozen all-conditions gate. The authoritative
report is `reports/pps_c42_confirmation.json`, SHA-256
`bd9ce91a6780ef8dd4f23ae81889b4eec245b77e1a6a611c476e1a10449eeba0`.

The confirmation used the exact three C41 checkpoint groups without training,
checkpoint selection, or parameter changes. C42-A contained 1,200 requests
from untouched C38 escrow with zero overlap against every previously opened
C38/C39/C41 feature cohort. Proposal lock, label-free feature collection, G0,
and execution lock completed before scoring; all A0 checks passed before A
labels opened. Dev/test records and qrels remained closed, and C42 executed
zero optimizer steps.

The core result replicated out of cohort. Seed-averaged NDCG@10 was 0.222629
for base, 0.323097 for C38 unprojected, and 0.333347 for metric-coupled
transport. Primary-minus-base was `+0.110718`, CI
`[0.085689,0.135329]`; primary-minus-C38 was `+0.010250`, CI
`[0.004672,0.015830]`. Both comparisons were positive in every frozen seed and
hash fold. True-minus-wrong history was `+0.035234`, CI
`[0.024063,0.046410]`, also positive in every seed and fold; clicked direction
had mean 0.543646 with CI `[0.505814,0.580582]`.

C42 nevertheless failed the preregistered uniqueness requirement. The primary
nominally exceeded `semantic_routing` by `+0.015583` and
`asymmetric_routing` by `+0.016531`, with every seed positive, but the paired
intervals `[-0.006376,0.038278]` and `[-0.005581,0.039120]` crossed zero and
one hash fold was negative. It did significantly exceed the single-wide
routing control by `+0.024226`, CI `[0.002024,0.046296]`.

Status remains `failed_A1_terminal`: no C42 rescue, new Amazon cohort, seed,
threshold, rank/head, temperature, or checkpoint selection is allowed. The
evidence supports a robust query/history Transformer family and specifically
confirms the coupled model over base and C38, but does not yet identify metric
coupling as the uniquely load-bearing primitive. A successor must use a newly
frozen cross-dataset test or a pre-outcome falsifier that separates coupling
from the two close multi-head routing controls; it must not tune on C42-A.
