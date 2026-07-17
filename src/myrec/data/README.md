# Current Motivation data adapters

The active data layer contains the standardized-record contract, label-free
history assignment builder, KuaiSearch scout/holdout materializer, request
manifest helper, and V1.2 release-lock validator. All generated data remains
under `data/` and is never committed.

Every adapter preserves strict temporal history, candidate identity, and
physical qrels isolation. Training and scoring code must not open confirmation
or source-test qrels.
