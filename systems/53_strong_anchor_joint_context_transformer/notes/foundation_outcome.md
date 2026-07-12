# C53 foundation outcome

Status: `failed_A0_terminal`.  C53 is closed without opening either domain's
internal-A labels.  Fresh reserve, dev, test, and qrels were never opened.

All five model tests passed, all 48 consumed external inputs were bound by the
proposal lock, and materialization placed Kuai query/history/candidate states
in the same fine-tuned D2 representation used by the strong D2p anchor.  Three
fixed seeds per domain then trained for two epochs on fit-only labels.

## Result

On Kuai, deleting candidate-to-candidate edges changed 50.3%--52.0% of full
orders and 2.0%--4.2% of Top-10 sets.  The listwise path was therefore active.
Replacing true history with a matched wrong user changed only 0--1 of 600
Top-10 sets, however, and two seeds failed the frozen loss-decrease check.
Kuai consequently failed A0.

Amazon passed its structural checks: deleting list edges changed every order
and 81.3%--86.0% of Top-10 sets; wrong history changed every order and
17.3%--34.0% of Top-10 sets.  This did not authorize labels because the gate
required both domains and every seed to pass.

A label-free post-terminal diagnostic explains the asymmetry.  True- versus
wrong-history correction correlations were 0.9976--0.9993 on Kuai and
0.9837--0.9954 on Amazon.  The ensemble changed only 2.5% of Kuai Top-10 sets
against strong D2p but 100% of Amazon Top-10 sets against raw BGE.  Thus the
shared Transformer mostly learned a history-invariant candidate-list function;
the weaker Amazon anchor merely allowed that function to overwrite more ranks.

## Consequence

Generic joint query/history/list contextualization is not a sufficient next
foundation.  A successor may not add width, layers, epochs, or a domain scale.
Its candidate-competition values must be algebraically zero unless history has
created a candidate-specific state, with ordinary independent history readout
and raw candidate self-attention retained as direct controls.
