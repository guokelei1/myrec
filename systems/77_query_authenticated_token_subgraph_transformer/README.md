# C77 — Query-Authenticated Token-Subgraph Transformer

C77 is the first post-C76 mechanism update under the terminal C80 budget.  It
uses a frozen pretrained semantic coordinate to construct a query-authenticated
candidate/history WordPiece subgraph, then runs a trainable bidirectional
Transformer only on that evidence-eligible subgraph.  Candidate-only shortcut
tokens cannot enter the personalized path merely because ranking labels favor
them.
