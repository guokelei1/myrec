# C46 — Prequential Behavioral-Semantic Representation Probe

C46 is a leakage-safe signal gate, not a proposed novelty claim. It asks
whether a content-initialized Transformer trained only on chronologically
earlier user transitions learns cross-item behavioral information that is
useful and user-specific on untouched strict-nonrepeat requests.

The outcome role contains 600 label-free-selected KuaiSearch train requests.
The representation source is restricted to request indices `[0, 40000)`, and
its maximum timestamp is strictly below every outcome timestamp. No dev/test
record or qrel is authorized.

See `notes/signal_gate_protocol.md` for the staged GPU commands.
