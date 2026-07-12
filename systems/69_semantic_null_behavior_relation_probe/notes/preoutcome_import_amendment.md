# C69 pre-outcome mechanical amendment

The first invocation of the three formal Kuai seed processes stopped during
Python import, before store construction, training, checkpoint writing,
scoring, or label access. C47's dependency path had been prepended to
`sys.path`, so `from model` resolved C47's package instead of C69's package.

The failed execution lock is preserved as
`execution_lock_preoutcome_import_failure_v1.json` with SHA-256
`f096a2b0eeb438d5cb0b16ef9fe1e386875cb5feba5854ccf3372545f8dce329`.

The amendment makes C47/C38 roots fallback import paths and restores C69's
root immediately before the candidate-local concrete model import. The second
guard is necessary because C47's frozen lock helper independently prepends its
own root while loading. The hypothesis, model, data roles, negative
construction, seeds, optimizer, steps, thresholds, and outcome boundary are
unchanged. A formal-entry-point smoke test must pass before a replacement
execution lock is created.
