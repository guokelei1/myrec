# C68 mechanism outcome

Status: **failed terminal at data-free G0**.

All three registered GPU seeds completed five equal-parameter modes for 600
steps. Every fit was finite, every parameter group received gradient, losses
decreased, and determinism, candidate permutation, no-history, query-mask,
repeat, and equal-user/reference-set contracts were bit-exact. No repository
record, label, qrel, dev, or test input was read.

The population-relative primitive did not solve functional identity. Clean
accuracy was only `0.5352/0.5534/0.5547`, versus base
`0.5169/0.5195/0.5052`, below the frozen `0.62` threshold in every seed.
Replacing the user preference with a wrong history reduced accuracy by only
`0.0742/0.0508/0.0964`, below `0.15` in every seed. A request-independent zero
history carrier still produced correction RMS `0.3946/0.3402/0.4340`, above
the maximum `0.25` in every seed.

The finite-temperature log partition also paid no stable structural rent.
Seed 20264801 lost to both the mean interaction (`0.5469`) and pooled joint
Transformer (`0.5495`); seed 20264803 essentially tied the pooled control
(`0.5547` versus `0.5534`, far below the required `+0.025`). Only the
single-NULL and user-only controls failed more strongly, with unsupported
correction RMS around `0.71--0.79`.

The exact cancellations are real but insufficient: subtracting population and
candidate-free free energies removes separable query/candidate and
query/history functions, yet a fixed event state can still contrast against
the population reference and create a candidate function. This is the same
remaining failure in a new coordinate.

C68 is closed. No temperature, reference count, reservoir, sparse-event count,
dimension, loss weight, step, seed, threshold, generator, or real-data rescue
is authorized. A successor must make a request-constant event state an exact
null of the candidate function, not merely compare it with a population state.

Authoritative promoted report SHA-256:
`4f6d91b9edc1046a02ae0735abd80e2f32eef6c0ecbc6612604de18fba618326`.
