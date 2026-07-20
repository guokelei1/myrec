"""Outcome-independent architecture opportunity cards for deep-dive closeout.

These are design opportunities, not implemented methods.  Their immutable fields
are fixed before the remaining D2--D7 evidence is available; closeout may only
rank, support, deprioritize, or reject them using admitted deliverables.
"""

from __future__ import annotations

import hashlib
import json


PRIOR_WORK_COMPARATORS = ("CoPPS", "BATA", "HMPPS", "MemRerank")
OPPORTUNITY_STAGE_BOUNDARY = "design_opportunity_only_not_implemented"


OPPORTUNITY_DESIGN_CATALOG = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": {
        "innovation_claim": (
            "Route a bounded query-conditioned subset of history events through an "
            "explicit abstention route before preference formation, making selection "
            "reversible and auditable under recurrence and candidate-disjoint transfer."
        ),
        "training_signal": (
            "Use the native ranking loss plus train-only relevant-versus-irrelevant "
            "history counterfactual consistency and a preregistered sparsity or entropy "
            "constraint; never select top-k from development outcomes."
        ),
        "training_data_requirements": (
            "Train-visible query-relevant and query-irrelevant histories matched on "
            "event count and approximate token length, with different-ID positives, "
            "candidate-excluded donors, wrong-user controls, and explicit usable coverage."
        ),
        "exact_null_recovery_invariant": (
            "When abstention is selected or routed event mass is zero, the history branch "
            "must contribute exact zero and expose the unchanged query-only/null ordering."
        ),
        "required_modules": [
            "Query-to-history event scorer with deterministic sparse top-k or sparse gating.",
            "Explicit null/abstention route whose mass is logged per request.",
            "Bounded routed-event interface with reversible selection ablations.",
        ],
        "critical_ablations": [
            "Dense attention versus sparse routing at matched parameter count.",
            "Router with versus without the explicit null route.",
            "Query-conditioned versus query-shuffled routing with identical visible tokens.",
            "Remove selected events versus equal-count unselected or random events.",
            "Same-request routing versus wrong-user and candidate-excluded donor controls.",
        ],
        "prior_work_differences": {
            "CoPPS": (
                "CoPPS uses sequence-view contrast and invariance; this card requires an "
                "explicit sparse per-query decision, exact abstention, and reversible "
                "recurrence/transfer attribution (KDD 2023, doi:10.1145/3580305.3599287)."
            ),
            "BATA": (
                "BATA injects external query/item relations as dense attention bias and "
                "auxiliary tasks; this card isolates a sparse inspectable selection and "
                "null route inside the unchanged ranker (TOIS 2025, doi:10.1145/3726864)."
            ),
            "HMPPS": (
                "HMPPS performs query-aware first-stage history filtering for an MLLM "
                "reranker; this card requires reversible internal routing and candidate-"
                "disjoint causal attribution (arXiv:2509.18682)."
            ),
            "MemRerank": (
                "MemRerank learns a downstream-reward preference memory; this card targets "
                "per-query sparse event routing with exact abstention before any preference "
                "compression (arXiv:2603.29247)."
            ),
        },
        "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
    },
    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK": {
        "innovation_claim": (
            "Force train-visible history into bounded, auditable, ID-free preference "
            "factors whose semantic content can be decoded, permuted, zeroed, and tested "
            "under different-ID transfer."
        ),
        "training_signal": (
            "Combine native ranking with semantic-preserving different-ID invariance, "
            "factor prediction or reconstruction, random-label controls, and separation "
            "from semantic-breaking histories."
        ),
        "training_data_requirements": (
            "Train-only attribute-preserving and attribute-breaking different-ID histories, "
            "factor labels from visible brand/category/text fields, candidate exclusions, "
            "missingness audits, label shuffles, and cross-user controls."
        ),
        "exact_null_recovery_invariant": (
            "Zeroing every preference slot and its mixture weights must contribute exact "
            "zero downstream and recover the unchanged query-only/null ordering without "
            "serializing raw item identity."
        ),
        "required_modules": [
            "ID-free history encoder over the existing visible-field whitelist.",
            "Bounded factor slots for brand, category, attributes, style, price, and intent.",
            "Query-conditioned slot mixture with training-only reconstruction heads.",
        ],
        "critical_ablations": [
            "Free-form history state versus fixed factor slots at matched capacity.",
            "Brand/category-only versus expanded attribute and intent factors.",
            "Different-ID invariance versus ranking loss alone.",
            "Factor-label shuffle, raw item-ID exposure, and slot permutation.",
            "Semantic-preserving versus semantic-breaking history replacement.",
        ],
        "prior_work_differences": {
            "CoPPS": (
                "CoPPS regularizes semantically related sequence views; this card makes "
                "the transferable factors explicit, decodable, intervenable, and linked "
                "to candidate-disjoint ranking (doi:10.1145/3580305.3599287)."
            ),
            "BATA": (
                "BATA supplies brand/category/query relations as dense bias and auxiliary "
                "reconstruction; this card makes those factors an explicit internal "
                "bottleneck rather than an external relation prior (doi:10.1145/3726864)."
            ),
            "HMPPS": (
                "HMPPS compresses product descriptions and filters histories for unseen "
                "items; this card preregisters auditable preference factors and semantic "
                "counterfactual interventions inside one ranker (arXiv:2509.18682)."
            ),
            "MemRerank": (
                "MemRerank compresses raw history into reward-trained memory; this card "
                "requires structured factors with slot-level causal ablations under a "
                "fixed candidate-disjoint contract (arXiv:2603.29247)."
            ),
        },
        "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
    },
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": {
        "innovation_claim": (
            "Decompose every candidate score into unchanged query relevance plus an "
            "auditable gated signed preference residual, so history must alter relative "
            "candidate advantages rather than only create a request-common shift."
        ),
        "training_signal": (
            "Combine native ranking with a signed residual pair/list loss, preference-state "
            "counterfactual swaps, slate permutation consistency, and a common-offset or "
            "zero-mean constraint on the residual."
        ),
        "training_data_requirements": (
            "Train-only same-query candidates differing in preference-compatible attributes, "
            "different-ID positives, query-matched hard negatives, identity and cross-request "
            "patches, slate permutations, and query-only controls."
        ),
        "exact_null_recovery_invariant": (
            "Setting the residual gate or preference residual to zero must algebraically "
            "recover the unchanged query-only/null scores and ordering; adding any common "
            "residual offset must leave ranks invariant."
        ),
        "required_modules": [
            "Candidate projection shared across the full request slate.",
            "Signed preference-to-candidate bilinear or cross-attention matcher.",
            "Per-request gate and separately logged residual score before final ranking.",
        ],
        "critical_ablations": [
            "Implicit LM readout versus explicit signed residual at matched capacity.",
            "Candidate-independent preference bias versus candidate-conditioned matching.",
            "Bilinear versus cross-attention residual matcher.",
            "Residual gate, common-offset, identity-patch, and cross-request controls.",
            "Slate permutation followed by inverse permutation.",
        ],
        "prior_work_differences": {
            "CoPPS": (
                "CoPPS optimizes history representations and downstream ranking; this card "
                "exposes an additive candidate-specific signed residual whose mediation can "
                "be patched and falsified (doi:10.1145/3580305.3599287)."
            ),
            "BATA": (
                "BATA uses dense relation-biased Transformer interactions; this card isolates "
                "history as a separate relative-score path with an exact query-only "
                "counterfactual (doi:10.1145/3726864)."
            ),
            "HMPPS": (
                "HMPPS applies pointwise MLLM reranking after filtering; this card compares "
                "all candidates through one shared signed preference residual inside the "
                "unchanged slate (arXiv:2509.18682)."
            ),
            "MemRerank": (
                "MemRerank trains preference memory for reranking utility; this card separates "
                "query relevance from a patchable candidate-conditioned residual and tests "
                "its mediation under fixed requests (arXiv:2603.29247)."
            ),
        },
        "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
    },
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": {
        "innovation_claim": (
            "Join an ID-free factor bottleneck to a separately measured signed candidate "
            "residual with abstention, so a preference factor counts as learned only when it "
            "causally changes relative scores under different-ID counterfactuals."
        ),
        "training_signal": (
            "Use native ranking, factor consistency/separation, signed residual ranking, "
            "common-offset control, and gate calibration that selects the query-only path "
            "when no reliable factor is present."
        ),
        "training_data_requirements": (
            "Train-only same-query different-ID pairs, factor-preserving and factor-breaking "
            "histories matched on exposure, candidate exclusions, missing-factor and wrong-"
            "user cases, label shuffles, and cross-request donors."
        ),
        "exact_null_recovery_invariant": (
            "Zeroing the factor slots, factor-to-candidate residual, or usefulness gate must "
            "each independently produce exact zero history contribution and recover the "
            "unchanged query-only/null ordering."
        ),
        "required_modules": [
            "Query-conditioned ID-free factor-slot encoder with an explicit empty slot.",
            "Candidate attribute projection aligned to the same factor basis.",
            "Signed factor-candidate residual head separated from query relevance.",
            "Causal audit interface for slot zeroing, permutation, and transplantation.",
        ],
        "critical_ablations": [
            "Free-form state versus factor slots at matched parameter count.",
            "Implicit LM score versus query-only plus signed residual decomposition.",
            "Factor loss only, residual loss only, and their joint objective.",
            "Explicit abstention versus an always-on preference gate.",
            "Same restoration, cross-request swap, slot permutation, ID exposure, and common offset.",
        ],
        "prior_work_differences": {
            "CoPPS": (
                "CoPPS regularizes sequence-view invariance; this card couples explicit slots "
                "to a separately falsifiable candidate residual and requires both representation "
                "and mediation evidence (doi:10.1145/3580305.3599287)."
            ),
            "BATA": (
                "BATA injects dense relation bias; this card enforces an ID-free bottleneck "
                "plus a decomposed relative-score path with exact zeroing and patch controls "
                "(doi:10.1145/3726864)."
            ),
            "HMPPS": (
                "HMPPS combines query filtering, description compression, and an MLLM reranker; "
                "this card tests one internal factor-to-residual causal path without changing "
                "the candidate population (arXiv:2509.18682)."
            ),
            "MemRerank": (
                "MemRerank learns reward-driven preference memory; this card structurally "
                "decomposes factor formation and candidate-relative use so each can fail "
                "independently under strict transfer (arXiv:2603.29247)."
            ),
        },
        "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
    },
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": {
        "innovation_claim": (
            "Expose and preregister how recurrence, candidate-disjoint transfer, and other "
            "surfaces consume the optimizer update budget, intervening only when measured "
            "gradient conflict survives label-shuffle and restored-state controls."
        ),
        "training_signal": (
            "Decompose native Q2 RankNet/ListNet and per-surface objectives, then compare "
            "original aggregation with preregistered normalization or conflict handling using "
            "exact one-step optimizer replay from identical state."
        ),
        "training_data_requirements": (
            "Train-only recurrence, strict-transfer, and overlap surfaces; fixed request-group "
            "exposure, query-matched hard negatives, different-ID preference pairs, within-"
            "request label shuffles, and identical optimizer-step budgets."
        ),
        "exact_null_recovery_invariant": (
            "With the surface controller coefficient set to zero, loss aggregation and the "
            "optimizer update from identical parameters, moments, variance, scheduler state, "
            "microbatches, and order must exactly recover the original training step."
        ),
        "required_modules": [
            "Surface-stratified train-only sampler with fixed exposure accounting.",
            "Per-surface objective and raw/effective update instrumentation.",
            "Optional gradient normalization or conflict handler with a zero-effect bypass.",
        ],
        "critical_ablations": [
            "Original mixture versus surface-balanced exposure at identical step count.",
            "Sampler balance versus loss weighting versus gradient projection.",
            "Raw-gradient proxy versus restored-state effective optimizer update.",
            "Observed labels versus within-request label shuffle.",
            "Recurrence-present versus recurrence-removed train-only controls.",
        ],
        "prior_work_differences": {
            "CoPPS": (
                "CoPPS uses contrastive sequence augmentation; this card accounts for optimizer "
                "updates across recurrence and transfer surfaces rather than assuming "
                "representation invariance fixes allocation (doi:10.1145/3580305.3599287)."
            ),
            "BATA": (
                "BATA adds relation bias and auxiliary reconstruction; this card targets "
                "measured surface-gradient allocation under fixed exposure and optimizer "
                "dynamics (doi:10.1145/3726864)."
            ),
            "HMPPS": (
                "HMPPS uses filtering and hard negatives; this card isolates recurrence-"
                "versus-transfer update competition without changing the evaluator population "
                "or reranker boundary (arXiv:2509.18682)."
            ),
            "MemRerank": (
                "MemRerank lets downstream reward train a preference extractor; this card is "
                "an optimizer-accounted surface budget with an exact disabled-controller "
                "recovery invariant (arXiv:2603.29247)."
            ),
        },
        "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
    },
}


OPPORTUNITY_IDS = tuple(OPPORTUNITY_DESIGN_CATALOG)
OPPORTUNITY_DESIGN_CATALOG_SHA256 = hashlib.sha256(
    json.dumps(
        OPPORTUNITY_DESIGN_CATALOG,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
