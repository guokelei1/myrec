# C44 partial-evidence logit-flow terminal outcome

C44 closed at its data-free D1 gate. The authoritative report is
`reports/pps_c44_design_gate.json`, SHA-256
`a6cf76cc52aa9c182d0433fc1d2497a0d6fb18faf03dd72e425af0800abf95ea`.
The design-lock SHA-256 is
`eacb1b61bf8cc2444f2ac2184cf2ac0f10f9bec8cca20d0b478a608a7d9edec2`.

All substantive structural checks passed. Every mode had 1,024 parameters and
paired initialization, both low-rank factors received finite gradients and
updated, candidate permutation/determinism/fallback contracts held, primary
mass was conserved, and the candidate correction was zero-sum within `9e-8`.

The planted task was solved perfectly but did not require the primitive. Base
NDCG@10 was 0.502184; primary and all three matched controls reached 1.0 in
every seed. Consequently, primary-minus-forced-flow, primary-minus-partial-
vector-write, and primary-minus-global-vector-write were all zero, below the
frozen `+0.02`. Clean-minus-wrong exceeded 0.51, and planted-candidate mass was
about 0.909, but irrelevant/wrong null mass was only about 0.488, below 0.50.

The machine report's `D0.status` is `failed` because the runner incorrectly
included negative access declarations (`repository_dataset_read=false`, etc.)
inside `all(checks.values())`. The scientific D0 checks are individually true;
the status-polarity bug is preserved and not repaired after outcome. It does
not affect the independent D1 terminal result: every matched reduction ties
the primary on clean utility.

Status remains `failed_D1_terminal`. C44 read no repository dataset, train
label, dev/test record, or qrels and receives no real-data authorization. No
teacher/noise/null threshold, temperature, seed, epoch, or loss rescue is
allowed. Candidate-axis partial logit flow is a valid operator but is not a
load-bearing architecture primitive under its own falsifier.
