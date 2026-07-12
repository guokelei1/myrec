# C70 logged-choice coverage audit

Date: 2026-07-12. No dev/test/qrel file was read.

For each train history event `(user_id, event_ts, item_id)`, the audit searched
the same dataset's train records for a request with the same user and timestamp
whose candidate set contained the historical item. It used only request/user/
time/query, history IDs, and candidate IDs. KuaiSearch's train file contains
labels, but the audit did not access candidate `clicked` or `purchased` fields;
Amazon used `records_train_blind.jsonl`.

| Quantity | KuaiSearch | Amazon-C4 |
|---|---:|---:|
| train requests | 163,717 | 11,199 |
| history event instances | 825,138 | 341,213 |
| unique `(user,time,item)` events | 152,649 | 336,325 |
| linked unique choice episodes | 147,405 | 0 |
| linked unique-event rate | 96.5647% | 0% |
| linked event-instance rate | 96.9889% | 0% |
| history-present requests with any linked episode | 93,411 / 96,042 | 0 / 11,199 |
| request coverage | 97.2606% | 0% |
| mean logged alternatives when linked | 81.77 | n/a |

All 825,138 KuaiSearch and 341,213 Amazon history instances were strictly
earlier than their recipient request.

The local JDsearch tree contains only the public sample and README. The public
format provides the current query/candidates/labels plus historical
query/item/action/time sequences, but not historical candidate slates. Thus it
cannot supply a logged historical alternative set without constructing pseudo
negatives.

Decision: KuaiSearch alone passes the structural prerequisite. The registered
dual-domain gate fails before model code or GPU use. C70 may resume only with a
second real logged-choice source or an explicitly revised paper scope; it may
not silently substitute mined negatives.
