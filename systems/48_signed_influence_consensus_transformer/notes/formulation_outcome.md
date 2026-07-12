# C48 exposed formulation outcome

C48 passed every structural check after the two recorded negative-stride
layout repairs, but failed its binding two-domain mechanism gate.  No fresh
reserve, trainable architecture, dev/test record, or qrels was opened.

| domain | primary NDCG@10 | vs base | vs plain KRR | vs softmax | true vs wrong | mean coherence |
|---|---:|---:|---:|---:|---:|---:|
| KuaiSearch | 0.310643 | +0.009772, CI positive | +0.000481, CI crosses; one fold negative | +0.003676, CI crosses | +0.007547, CI crosses | 0.9815 |
| Amazon-C4 | 0.271014 | +0.017812, CI positive | +0.002109, CI crosses | -0.005874 | +0.018772, CI positive | 0.8849 |

Signed influence consensus retained the plain KRR signal on KuaiSearch and
made clicked direction/specificity significant, but did not pay stable rent
over KRR.  On Amazon it contracted useful semantic attention and lost to both
softmax and C47 posterior support.  Eventwise dual influences were already
almost fully sign-aligned, especially on KuaiSearch, so sign agreement did not
create a meaningful rejection surface.

Decision: `failed_formulation_terminal`.  Do not consume the C47 reserve or
implement/train a C48 Transformer.  Do not tune the coherence exponent,
epsilon, ridge, scale, or a domain/history-length mixture.
