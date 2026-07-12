# C29 pre-lock implementation review

Status: passed for the preregistered label-free G0; no C29-A feature, score, or
label had been opened at review time.

The implementation instantiates one primitive rather than a score mixture:
strictly prior user memory controls which history events enter the LM's
history-to-candidate attention path.  User ID and the memory set have no logit
edge.  A shared full BGE Transformer produces factual and null logits; their
centered difference is the only non-repeat personalized write.

Review corrections made before lock:

- wrong donors must have a different user and zero recipient-candidate overlap;
- delayed-B was removed from G0 feature roles and stays fully closed through A1;
- the random-init control now uses the identical Transformer config and capacity;
- the readout starts at exact zero and all encoder dropout is disabled, making
  identical factual/null inputs cancel during training as well as inference;
- the measured model size is 23,954,432 parameters, not the earlier estimate;
- the known interactive qrels-schema incident is registered explicitly; C29
  code contains no qrels, records, dev/test, or metrics input path.

Six CPU tests pass.  A synthetic full-batch GPU backward with the formal shape
(48 candidates, factual/null true and wrong streams, length 128) used 3.37 GiB
allocated memory.  A ten-step steady-state timing probe measured 0.143 seconds
per optimizer step, projecting about 0.38 GPU-hours for 9,483 fit steps before
data collation and evaluation overhead.  These measurements contain no C29
outcome.

Residual risk: the temporal membership mask may prove useful only as provenance
filtering while the ordinary cross-encoder still fails to learn candidate
direction.  G0 therefore first demands strong true/wrong authentication
separation; A0 demands load-bearing corruption and order effects; A1 demands
stable ranking direction.  Any failure is terminal for this primitive.
