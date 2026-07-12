# C38 cross-domain transfer start

C37 closed at A1 because its candidate-specific barycentric residual was
utility-indistinguishable from simpler global and uncentered transports.  To
avoid another KuaiSearch-conditioned geometry revision, C38 freezes a direct
Amazon-C4 transfer falsifier for the weak surviving shared global tangent
write.  C38 is explicitly confirmatory and cannot serve as the paper's novel
architecture.

## Source acquisition

- Amazon-C4 Hugging Face revision:
  `39322697749a88d179f88d322a2fe4765b655c98`.
- Temporal Amazon-C4 history release revision:
  `0e623aa3a9431aed873dda26f272ae34bcc96633`.
- `test.csv` SHA-256:
  `1372e8659ca475c3e08561ed09042837891ae0f7151650d38e5c6e93858e4186`.
- sampled-1M metadata SHA-256:
  `c307b1017e0ff15a799e479a91ad59b55df91fd1b52bf1ea4dee01aed944cc74`.
- combined temporal history SHA-256:
  `7938cf414ae6321700c55e4d832f86a0982b9d062cf3a2b795b9f219dd91c864`.
- query dictionary SHA-256:
  `d2e96594873b0305342e2b338d809d73ec7dc8455680430a9065e9c5509e584e`.
- Amazon Reviews 2023: 34 category metadata archives, total compressed size
  26,281,036,002 bytes; individual hashes are written by the C0 converter.
- JDsearch GitHub revision:
  `0cdbfb403c2b2abb195cac50bb0c8dc49ee0f143`.  Only schema samples are
  available non-interactively; the full data requires JD Cloud login/QR and
  is not used as experimental evidence.

## Pre-label source audit

The history train split has 11,199 unique requests and joins Amazon-C4 on qid,
user, and positive product at 100%.  Every row has history and the positive
product is absent from every released history.  Histories must be globally
timestamp-sorted because category grouping destroys global order; the unified
converter sorts then keeps the most recent 50.

The official sampled-1M catalog covers every positive but only 4.35% of unique
history products.  Therefore a transfer run using only that catalog would
confound architecture failure with missing history text.  Full category
metadata was downloaded before any model outcome.  The C0 gate requires at
least 95% history-event text coverage after the join.

## Frozen direction

Candidate construction is BM25 top-100 over the official sampled-1M catalog
plus the positive.  Long query cost is bounded without labels by selecting at
most eight unique query terms with the lowest frozen catalog document
frequency.  All model modes share the resulting candidate manifest.

C38 primary is the unchanged query-attended shared tangent write.  Equal-
parameter controls remove tangent projection or query-conditioned history
weights.  Exact recurrence is preserved in the common base and suppresses the
transport write; missing history or query also gives exact base equivalence.
Only upstream train is eligible.  Fit/A/B/escrow are hash-selected from
`records_train_blind.jsonl`; upstream dev/test are forbidden.

## Missing-history-text decision

The final nonblank-title audit covered 456,594 of 490,571 released history
events (93.07%) and 364,769 of 392,215 unique history products (93.00%).  All
33 named upstream categories joined completely; the shortfall was concentrated
in `Unknown`.  A frozen cross-category fallback scanned every other official
Amazon Reviews 2023 metadata archive for the 27,446 missing `Unknown` product
IDs and recovered zero, so category routing is not the explanation.

Before any model outcome, the formal label-free conversion showed that dropping
only history events with blank or whitespace-only joined titles retains 341,213
of 366,842 train events, 56,893 of 60,934 dev events, and 58,488 of 62,795 test
events.  No train request and no dev request becomes empty; one test request
does.  Retained median history length is 35/36/36 for train/dev/test.

C38 therefore freezes an evidence-availability mask rather than imputing or
inventing product text.  Every downstream method consumes only history events
with nonblank titles.  C0 must still report the raw 93.07% source coverage and
must additionally require consumed history text coverage >=95%, missing-event
drop fraction <=10%, zero empty train histories after masking, and retained
train median history length >=10.  These checks depend only on evidence
availability, never on dataset labels, categories, query types, or method IDs.

## Terminal train-gate result

C0, C1, the proposal lock, label-free BGE encoding, G0, and the execution lock
all passed in order.  Three A40 seeds completed and all 22 label-free A0 checks
passed.  On 1,200 untouched internal-A requests, query-attended tangent
transport reached 0.317952 NDCG@10 versus 0.243638 for frozen BGE
(`+0.074314`, 95% CI `[0.047307, 0.101719]`).  True history exceeded the
same-length-bin wrong-user control by `+0.041688`, CI
`[0.030953, 0.052623]`; query attention exceeded uniform-history aggregation
by `+0.010269`, CI `[0.000390, 0.019900]`.

C38 nevertheless failed its preregistered A1 because the equal-capacity
unprojected reduction reached 0.329332 and exceeded the tangent primary by
`+0.011381` (equivalently primary minus reduction `-0.011381`, CI
`[-0.020650, -0.002271]`).  The cross-domain result therefore validates the
problem-level history signal and query-conditioned selection, while falsifying
the tangent projection itself.  Delayed-B, escrow, dev, and test stay closed.
The next architecture moves to internal Transformer representation/value
learning and treats unprojected shared transport only as a strong control.
