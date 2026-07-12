# Problem alignment audit after C67

## Verdict

The search is methodologically moving closer to the problem, but empirical
quality is not improving yet. It is not currently fitting KuaiSearch labels:
C65 stopped before training labels, C66 stopped before validation labels, and
C67 stopped before reading any repository data. The gates are increasingly
good at exposing generic query-candidate shortcuts before utility evaluation.

That is useful localization, not a good architecture result. C64 showed that
trainable LM representations create ranking activity; C65--C66 showed that a
counterfactual hidden residual still does not make user identity load-bearing;
C67 showed that even history-only fast-weight writing can be bypassed by a
flexible downstream read function.

## Current closed family

Do not continue by tuning C67 or by deleting its query-candidate pair edge and
rerunning the same generator. More broadly, the following recipe is now closed:

```text
history -> some learned state
query/candidate + state -> flexible learned comparator
output or auxiliary loss is expected to make state identity matter
```

The comparator can learn a generic solution while using the state only as a
nonzero carrier. Wrong-history output penalties, NULL subtraction, slot
allocation, and held-out reconstruction did not prevent this.

## Next design gate

Before another architecture is implemented, its algebra must pass a stronger
functional-identifiability review:

1. replacing the written history state by any request-constant nonzero state
   must not preserve the candidate function;
2. the personalized candidate correction must be exactly zero under a
   history-scrubbed state, not merely trained toward zero;
3. query/candidate-only parameters must not be able to express the same
   correction family;
4. a matched function-equivalence control must be specified before GPU use;
5. the identical graph and thresholds must apply to KuaiSearch and Amazon-C4.

This is the right direction for avoiding dataset tuning, but no current
candidate satisfies both these requirements and positive utility evidence.
