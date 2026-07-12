# C33 paired train-only protocol

Train both fixed modes for seeds 20260941/42/43 on the unchanged 10,000-request
fit cohort.  Each mode uses rank 16, temperature 0.1, scales 1/2, equal
listwise/direction loss, one epoch, at most 32 complete requests per batch, and
all candidates.  Paired modes use identical initialization and request order.

C33-A, delayed-B, and escrow are newly selected and exclude all C32 outcome and
reserved roles.  No A label opens until both modes pass A0.  A1 compares the
tangent primary with both D2p and unprojected transport.  Dev/test stay closed.
