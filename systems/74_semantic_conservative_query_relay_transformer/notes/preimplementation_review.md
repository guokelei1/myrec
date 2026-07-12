# C74 preimplementation review

Decision: `authorize_data_free_design_gate_only`.

The candidate is admissible because it changes the internal value lifecycle,
not a dataset field or output mixture.  It is distinct from C73's learned
V/O/FFN relay and from C40/C42's fully coupled metric.  C41 is the closest
known reduction and is bound by pooling before the candidate relay.

The zero-training formulation diagnostic is explicitly non-fresh and failed
the shuffle contract; it may justify the hypothesis but not count toward C74's
gate.  The formal generator seed is new and locked before training.

Only the data-free GPU gate is authorized.  Repository data, pretrained-LM
training, dev, test, qrels, and the shared evaluator remain closed.
