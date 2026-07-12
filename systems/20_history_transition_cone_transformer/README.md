# C20 — History-Transition Cone Transformer (HTCT)

Status: **paper formulation passed only for a hash-locked synthetic falsifier;
no repository data, dev, test or qrels access is authorized**.

HTCT changes the internal evidence read from scalar endpoint attention to a
request-local conic reconstruction.  A lower Transformer forms a
`query→candidate` displacement and the ordered history's adjacent transition
vectors.  Fixed unrolled projected-gradient steps reconstruct the former using
only nonnegative combinations of the latter.  Only the reconstructed component
may update the candidate token before the shared rank head.

The nonnegativity restriction is load-bearing: an unconstrained span can use a
negative coefficient to call the reverse of an observed transition “support”.
The synthetic gate must beat that span, ordinary simplex retrieval, a one-step
ReLU degeneration and a matched pooled-transition MLP before any real-data
design is considered.
