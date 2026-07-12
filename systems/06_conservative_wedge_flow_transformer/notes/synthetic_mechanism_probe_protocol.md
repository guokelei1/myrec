# C06 bidirectional synthetic mechanism probe

Status: **pre-outcome protocol; implemented but not executed**.

This probe is a deterministic, CPU-only structural falsifier for the
candidate-local Hodge trust hypothesis. It reads no repository dataset, qrels,
model checkpoint, prior score, dev record or test record. It has no optimizer.
The formal outcome is undefined until the configuration, source, this protocol
and its tiny structural tests have been locked.

## Question and non-claim

The probe asks one conditional question:

> If candidate potential error is locally coupled to the incident energy of a
> divergence-free cycle, does candidate-local Hodge trust recover rankings more
> reliably than no trust or one global event gate, and does that advantage
> cease to be a residual positive gain (and possibly become materially harmful)
> when the coupling is removed, and reverse when the coupling is reversed?

The reliability-aligned world deliberately plants the premise C06 hopes may be
useful. Passing it is only a positive-control result under that premise. It is
not evidence that recommendation histories have this coupling. Passing the
synthetic probe does not establish novelty, justify a real-data claim, or
authorize train-internal, dev, full training or test access.

## Frozen execution

- FP64 on CPU; no autocast and no training;
- generator seeds `20260711`, `20260712`, `20260713`;
- 4,096 independent requests per seed;
- 16 candidates and 6 equally weighted events per request;
- potential RMS `1.0`;
- candidate cycle log-scale standard deviation `0.75`;
- event cycle-ratio log standard deviation `0.7`;
- noise scale `0.8` and variance floor `0.05`;
- 10,000 paired bootstrap resamples with seed `20260714`.

No value above may be changed after an outcome. A changed distribution,
threshold or comparison constitutes a new hypothesis and requires a new lock;
the old outcome remains recorded.

Potential RMS `1.0` is a dimensionless standardized gauge for this synthetic
coordinate, not a claim about the scale of real LM logits. This probe does not
test whether every explicit synthetic field is realizable by C06's finite-rank
wedge factors, and it does not test the full model's final score bound. Those
are separate algebraic/implementation contracts.

## True potential and divergence-free cycle

For every request and event, sample and normalize a centered true potential:

```text
z_i ~ Normal(0, 1)
u*_i = RMS-normalize(z_i - mean(z))
G*_ik = u*_i - u*_k.
```

Let `P=I-11^T/n`, sample node scales `h_i~LogNormal(0,0.75)` and a Gaussian
matrix `R`, and construct:

```text
A = (h h^T) .* (R-R^T) / sqrt(2n)
C0 = P A P.
```

Algebraically `C0^T=-C0` and `C0 1=0`. It is therefore a strictly
divergence-free complete-graph cycle field. An event-level lognormal scalar
rescales `C0` so its mean incident energy has the sampled ratio to the clean
gradient energy. This global rescaling uses no candidate outcome.

For each candidate/event define:

```text
e_ij = sum_k C_ikj^2
v_ij = e_ij / mean_{i,j}(e_ij).
```

The run is invalid unless maximum skew error, row divergence and observed
Hodge-potential recovery are all at most `1e-12`.

## Paired worlds

The three worlds share bit-identical `u*`, `C`, Gaussian draw `xi`, and the
complete per-request multiset of planted noise variances. Only the assignment
of that multiset changes.

1. `reliability_aligned`: assign variance in the order of `v_ij`.
2. `reliability_decoupled`: independently permute those values within each
   request across candidate/event positions.
3. `reliability_adversarial`: assign the sorted values in reverse incident-
   energy rank.

For the assigned value `a_ij`:

```text
sigma_ij^2 = 0.8^2 * (a_ij + 0.05) / 1.05
eta_ij = sigma_ij * xi_ij
u_obs_:j = center(u*_:j + eta_:j)
F_ikj = u_obs_ij - u_obs_kj + C_ikj.
```

The integrity gate requires per-request mean Spearman correlation of at least
`0.999999999` in the aligned world, absolute mean at most `0.02` in the
decoupled world, and at most `-0.999999999` in the adversarial world. The sorted
variance arrays must be bit-identical across worlds.

## Four fixed gates

Compute observed incident energies:

```text
EG_ij = sum_k (u_obs_ij-u_obs_kj)^2
EC_ij = sum_k C_ikj^2.
```

Every comparison uses the same conservative edge and aggregation:

```text
T_ikj(g) = 0.5 g_ij g_kj (u_obs_ij-u_obs_kj)
d_ij(g) = mean_k T_ikj(g)
score_i(g) = mean_j d_ij(g).
```

Only `g` changes:

- local Hodge: `g_ij=EG_ij/(EG_ij+EC_ij+1e-12)`;
- global event: `g_ij=sqrt(sum_i EG_ij/(sum_i EG_ij+sum_i EC_ij+1e-12))`;
- untrusted projected flow: `g_ij=1`;
- direct reliability oracle: `g_ij=1/(1+sigma_ij^2)`.

The direct oracle reads the generator's planted variance. It is an unattainable
diagnostic reference, not a proposed model and not a promised mathematical
upper bound for every finite sample.

## Metrics and deterministic statistics

The true total score is `mean_j u*_ij`. The primary metric is request-equal
pairwise accuracy over all unordered candidate pairs. True differences below
`1e-10` are omitted and an exact predicted tie earns `0.5`.

The secondary metric is binary NDCG@10: the four candidates with the largest
true score are relevant. Ties use SHA256 of the synthetic request ID, candidate
ID and salt `20260708`. All 16 candidates remain in every ranking.

The script reports every gate in every world per seed and pooled. The five
binding pairwise comparisons use the same 10,000 request-resampling draws for
paired 95% bootstrap intervals.

## Bidirectional stop rules

All integrity checks are binding.

In the aligned world, local Hodge must exceed both `t=1` and the global event
gate by at least `0.01` pairwise accuracy. Both pooled interval lower bounds
must exceed zero, every seed direction must be positive, and both pooled
NDCG@10 directions must be positive.

In the decoupled world, the upper endpoint of the local-minus-`t=1` pairwise
interval must be no larger than `+0.002`. A material residual gain indicates
generic shrinkage or a generator confound rather than the planted coupling.
The decoupled rule is deliberately one-sided: a null effect or material harm is
allowed. Its only claim is that a positive aligned-world advantage must not
survive after cycle/error coupling is removed.

In the adversarial world:

- local-minus-`t=1` pairwise accuracy must be at most `-0.01`;
- its interval upper bound must be below zero and every seed must be negative;
- oracle-minus-local must be at least `+0.01` with interval lower bound above
  zero.

Failure of the aligned conditions means the primitive cannot exploit its own
favorable premise. Failure of either negative world means the probe has not
shown that local Hodge trust is load-bearing and conditional. Either case stops
the mechanism before real data. Parameter motion, attention mass, or a good
result in only one world cannot override the decision.

## Fixed paths, independent lock and invocation

The only accepted paths are these exact repo-relative strings:

```text
config   systems/06_conservative_wedge_flow_transformer/configs/c06_synthetic_mechanism_probe.yaml
script   systems/06_conservative_wedge_flow_transformer/experiments/run_synthetic_mechanism_probe.py
protocol systems/06_conservative_wedge_flow_transformer/notes/synthetic_mechanism_probe_protocol.md
test     systems/06_conservative_wedge_flow_transformer/tests/test_synthetic_probe.py
lock     systems/06_conservative_wedge_flow_transformer/notes/c06_synthetic_probe_lock.json
output   artifacts/c06_conservative_wedge_flow_transformer/synthetic_v1/report.json
```

The script derives the C06 and repository roots from `__file__`; CLI paths are
interpreted relative to that repository root, never relative to the caller's
working directory. Absolute paths, `./` aliases, alternate outputs and parent
traversal are rejected.

Before constructing an RNG or calling the generator, the runner hashes the
config, executing script, this protocol and the tiny test. The independent lock
must then match this schema exactly:

```json
{
  "lock_id": "c06_synthetic_probe_preoutcome_v1",
  "probe_id": "c06_local_hodge_bidirectional_synthetic_v1",
  "status": "locked_before_synthetic_outcome",
  "outcomes_observed_before_lock": false,
  "output_path": "artifacts/c06_conservative_wedge_flow_transformer/synthetic_v1/report.json",
  "files": {"<each exact path above except lock/output>": "<sha256>"},
  "combined_sha256": "<hash of sorted path, NUL, file hash, newline records>"
}
```

The lock itself is then hashed into the report. Missing/stale locks fail before
generation. The report reuses the pre-run manifest object and never recomputes
source hashes after observing outcomes.

An existing output is terminal for the command: it is never overwritten. The
new report is written to an exclusive temporary file and published by an
atomic hard link that also fails if another process wins the output race.

The command is documented but must not be run until the coordinator creates a
pre-outcome lock and authorizes the CPU outcome:

```bash
python systems/06_conservative_wedge_flow_transformer/experiments/run_synthetic_mechanism_probe.py \
  --config systems/06_conservative_wedge_flow_transformer/configs/c06_synthetic_mechanism_probe.yaml \
  --output artifacts/c06_conservative_wedge_flow_transformer/synthetic_v1/report.json
```

The JSON contains hashes for the config, executing source and this protocol;
integrity results; per-seed and pooled metrics; paired intervals; every frozen
check; and the terminal synthetic verdict.
