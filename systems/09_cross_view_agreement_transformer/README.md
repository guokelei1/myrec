# C09 — Cross-View Agreement Transformer

Status: **mathematical reduction audit conditionally passed; CPU structural
prototype passed; synthetic/data gates not run**.

Decision boundary: CMA is not a global scalar mixture or candidate-local
diagonal residual gate under the frozen architectural definition, but it **is**
a constrained ordinary set-attention instance.  Its only possible surplus is
the restricted-path residual-margin construction of the permission matrix.
Recommendation: **CPU synthetic probe yes; real-data/dev probe no unless the
synthetic matched controls pass**.

C09 tests one narrow hypothesis: history may alter a candidate ordering only
when two structurally restricted paths through one shared Transformer support
the same history-induced candidate margin.  The primitive is **Conjunctive
Margin Attention (CMA)**.  It is a candidate-pair attention matrix, not an
average of two rankers and not a scalar personalization gate.

The current tree contains no dataset reader, evaluator, checkpoint, model
weight, score dump, or outcome.  It does not read labels or qrels and does not
claim empirical ranking quality.

## Files

- `reduction_audit.md` — strict reduction analysis and the precise sense in
  which CMA is / is not distinguishable from a generic gate;
- `proposal.md` — architecture, equations, information flow, objective, and
  predicted failure modes;
- `mechanism_fingerprint.md` — collision-resistant mechanism fingerprint;
- `nearest_neighbor_audit.md` — primary-source literature audit with URLs;
- `pre_outcome_gate.md` — frozen synthetic and later dev stop conditions;
- `prototype.py` — self-contained CPU Transformer and CMA reference operator;
- `tests/test_prototype.py` — hand-computed and structural tests;
- `test_report.md` — exact local test command and result.

## Reproduce the CPU checks

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" \
python -m unittest discover \
  -s systems/09_cross_view_agreement_transformer/tests -v
```

The prototype requires only PyTorch.  It is intentionally small and is not
authorized for cohort access, dev evaluation, GPU use, or full training.
