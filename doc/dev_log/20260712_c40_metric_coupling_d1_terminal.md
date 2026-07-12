# C40 metric-coupling D1 terminal

C40 tested whether query/history selection, transported values, and candidate
readout should share one learned semantic metric. All 11 structural checks
passed and the planted multi-head teacher was strongly recoverable with
corruption specificity. The frozen gate still failed because one seed missed
the shifted-loop margin and the simpler selection-only reduction won every
seed.

The result narrows the C39 diagnosis. Arbitrary learned value/output/FFN maps
destroyed true-user specificity; fully coupling those maps does not repair it.
The stronger conditional rule is asymmetric: routing may learn, but history
content and candidate readout should remain in the pretrained LM coordinate.

This is not yet a paper claim. Fixed identity values/projections have direct
Transformer precedents, including *Simplifying Transformer Blocks*, and QKV
sharing is already studied. A successor must first test whether
semantic-content-preserving routing beats C38 and fixed-semantic attention on
untouched data, then identify a ranking-specific primitive rather than claim
identity values themselves as novel.

Authoritative report: `reports/pps_c40_design_gate.json`, SHA-256
`ac845ea629adaf9f142c5a18d19fdf15410da1399895309fc1a9d066acda9406`.
