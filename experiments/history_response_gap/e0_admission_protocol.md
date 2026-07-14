# E0 source and admission protocol — review draft

Status: **draft; no download, training, evaluator call, or dev/test label
opening is authorized by this file.** It is the operational companion to
`doc/34_history_response_direction_gap_validation_plan.md`.

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

KuaiSearch Full is audited first. KuaiSAR Full is the pre-registered
functional replication; JDsearch is the fallback only for claims supported by
its information object. Amazon-C4 remains non-binding for the natural-search
claim.

## Deliverables

After human review, freeze one protocol and write only small tracked cards
under this directory. Put generated samples, joins, and records under
`data/`, and put audit JSON under `reports/` using the common report naming
convention. Do not create model configs or `systems/` source during E0.

