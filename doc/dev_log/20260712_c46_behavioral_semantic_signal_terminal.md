# C46 behavioral-semantic signal terminal

C46 repaired the target-contamination problem by training only on source
requests strictly earlier than its 600-request untouched A role. The content
Transformer clearly distinguished real from shuffled user transitions, but it
tied frozen semantic mean history and lacked statistically stable true/wrong
specificity.

The result narrows the representation problem: next-item pretraining on LM item
states mostly recovers semantic co-occurrence. More pretraining or a larger
sequence encoder would refine an established UniSRec/Recformer neighbor without
adding the missing personalized information. The next safe diagnostic should
condition each past behavior on the query under which it occurred, testing a
search-specific preference residual rather than another item-only sequence.
