# Data adapters

The previous KuaiSearch Lite and Amazon-C4 materializers are archived.
`contracts.py` is the active dataset-independent record, temporal-causality,
candidate-uniqueness, and evaluation-label-isolation validator. Source-specific
Full-track adapters will be added only after their E0 field mapping is
reviewed.

New adapters must preserve temporal history boundaries, candidate manifests,
and physical dev/test label isolation.
