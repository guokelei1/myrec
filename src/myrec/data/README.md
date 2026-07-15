# Data adapters

The previous KuaiSearch Lite and Amazon-C4 materializers are archived.
`contracts.py` is the active dataset-independent record, temporal-causality,
candidate-uniqueness, and evaluation-label-isolation validator.
`kuaisearch_source_audit.py` is an exploratory, outcome-free source scanner;
it does not materialize a confirmation cohort.
`kuaisearch_scout.py` creates a local-only, source-train time-window pilot with
qrels physically separated from method-visible dev records. It is development
evidence, not confirmation.
`kuaisar_scout.py` creates the analogous KuaiSAR Small user-input-search scout,
using earlier positive search and recommendation clicks as causal history. Its
anonymized word IDs support functional replication only, not plaintext semantic
claims.
`jdsearch_standardize.py` verifies the public mirror against the official sample
and documented counts, then creates a stable-hash JDsearch scout with graded
qrels physically separated from method-visible dev records. History query term
IDs are retained alongside product interactions.

Source-specific exploratory adapters may be developed under the active
manifest. A confirmation adapter and its final field mapping are frozen only
after the relevant source observations have been reviewed.

New adapters must preserve temporal history boundaries, candidate manifests,
and physical dev/test label isolation.
