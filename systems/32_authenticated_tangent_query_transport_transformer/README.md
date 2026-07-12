# C32 Authenticated Spherical-Tangent Query Transport Transformer

C32 isolates one geometric change to C31.  Strictly authenticated history still
forms one request-level profile in a frozen BGE space, and the same rank-16
adapter is shared by query, history, and candidates.  Before moving the query,
C32 removes the profile component parallel to the adapted query and writes only
the orthogonal tangent component on the unit sphere.

This is intended to preserve query identity while allowing only a
candidate-relative directional change.  Capacity, fit data, semantic attention,
temperature, correction scale, loss, epoch count, and fallback rules are
unchanged from C31.  Former C31-A is diagnostic-only and excluded.  Formal
C32-A is former C31 delayed-B, which has never been feature-materialized,
scored, or labeled.

No C32 delayed role, escrow, dev, or test access is authorized.
