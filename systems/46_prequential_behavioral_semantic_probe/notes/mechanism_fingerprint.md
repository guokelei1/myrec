# C46 mechanism fingerprint

- **Purpose:** representation signal gate, not novelty candidate.
- **Input state:** frozen 512-dimensional LM item-title vectors.
- **Behavior operator:** `[READ]` token Transformer over a strict user prefix.
- **Training signal:** next distinct clicked item against 63 source-only
  negatives; current outcome labels are absent.
- **Cold-item contract:** candidate representation is a shared function of LM
  content, never a source item-ID lookup.
- **Matched null:** identical Transformer trained with a deterministic global
  permutation of prefix/target pairing.
- **Fixed neighbor:** normalized mean semantic history-to-candidate similarity.
- **Inference inputs:** history and candidates only; current query is excluded
  intentionally because this gate isolates behavioral representation.
- **Complexity:** `O(H^2 d + C d)` with `H<=20`; no online LLM call.
- **Novelty verdict:** reducible to known content-based sequential
  recommendation family; ineligible as the final proposed system.
