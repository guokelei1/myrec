# C71 Logged-Choice Gradient Signal Probe

C71 is a KuaiSearch train-only information-object gate for C70. It tests
whether a historical choice relative to the candidate slate actually shown
under the historical query supplies a better ranking direction than
positive-only history, uniform-slate centering, ordinary semantic history, or
matched wrong-user episodes.

C71 is not the proposed architecture and cannot establish cross-domain
generality. Its target role is selected from the 66,778 KuaiSearch train
requests excluded from the historical packed pool, so it does not reuse any
prior candidate's packed outcome role. Dev, test, and qrels remain closed.
