# E0 source and confirmation-admission protocol

Status: **reserved for the later confirmation lock.** Open exploration now
follows `exploration_protocol.md`; failure to meet this card on Lite does not
reject KuaiSearch Full or stop another dataset from being understood. This
file becomes binding only when a confirmation cohort is frozen.

## Decision question

Can the candidate dataset provide a label-free, temporally valid population
where the same or clustered query and overlapping candidate evidence can be
combined with different users' prior histories, and where later labels can
measure candidate-relative direction without post-outcome slice selection?

## Required audit outputs

1. source schema and provenance card;
2. request/candidate slate reconstruction counts and candidate-size summary;
3. strict history-before-request leakage audit;
4. text, item, user, session, and timestamp coverage;
5. label-free eligibility counts for all, strict-nonrepeat, and natural
   context-demand populations;
6. query repetition and candidate-overlap distribution;
7. train/dev/confirmation split and contamination audit;
8. training-only label distribution and minimum-detectable-effect estimate.

The audit must be executable from source data and must not inspect development
or confirmation outcomes. Counts are registered before any model result is
available.

## Admission rule

Admit a track only if it has a real request query, strictly prior history,
fixed exposed candidates, candidate-level labels, stable identifiers, an
independent time/user boundary, and enough eligible requests for the frozen
primary endpoints. If any binding condition fails, record the narrowest
supported role (`replication`, `stress test`, or `rejected`) and stop.

Candidate roles are KuaiSearch Full for natural-language search, KuaiSAR for
functional replication, JDsearch for graded behavioral robustness, and
Amazon-C4 for semantic stress. Exploration may inspect them in any cost-aware
order. Their final confirmation roles are assigned only after source and power
evidence is understood, before independent confirmation outcomes are opened.

## Deliverables

After human review, freeze one protocol and write only small tracked cards
under this directory. Put generated samples, joins, and records under
`data/`, and put audit JSON under `reports/` using the common report naming
convention. Do not create model configs or `systems/` source during E0.

## Current exploratory reuse note

Amazon-C4 motivation replication needs a fresh standardized scout from the
locally present official query, history, catalog, and Reviews-2023 metadata
files. The current implementation may rewrite only the raw field parsing,
temporal-target exclusion, deterministic BM25 candidate construction, metadata
join, and physical label-isolation logic from
`archive/legacy_20260714/source/src/myrec/data/data/amazon_c4_standardize.py`.
The deterministic candidate manifest may be used only as a computation cache
after current code asserts the current raw-catalog hash, rebuilt FTS index hash,
cache hash, request set, slate uniqueness, and positive inclusion. No archived
standardized record, score, outcome, threshold, gate, or admission decision is
promoted; current contracts and new tests are binding.
