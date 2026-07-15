#!/usr/bin/env python
"""Forward/backward smoke test for the locked official HSTU and SASRec cores."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "baselines" / "hstu"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))

from myrec.utils.jsonl import write_json


UPSTREAM_COMMIT = "6135bc30398f97e5786674192558d91f2ef2fa90"


def _load_sparse_cumsum(torch, fbgemm_gpu) -> tuple[bool, str]:
    if hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        return False, "already_registered"
    library = Path(fbgemm_gpu.__file__).parent / "fbgemm_gpu_sparse_async_cumsum.so"
    if not library.exists():
        raise FileNotFoundError(f"missing fbgemm sparse cumsum library: {library}")
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"):
        raise RuntimeError("fbgemm asynchronous_complete_cumsum was not registered")
    return True, str(library)


def _build_core(architecture: str, *, embedding_dim: int, max_length: int):
    from generative_recommenders.research.modeling.sequential.embedding_modules import (
        LocalEmbeddingModule,
    )
    from generative_recommenders.research.modeling.sequential.hstu import HSTU
    from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
        LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    )
    from generative_recommenders.research.modeling.sequential.output_postprocessors import (
        L2NormEmbeddingPostprocessor,
    )
    from generative_recommenders.research.modeling.sequential.sasrec import SASRec
    from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
        DotProductSimilarity,
    )

    embedding = LocalEmbeddingModule(num_items=32, item_embedding_dim=embedding_dim)
    common = {
        "max_sequence_len": max_length,
        "max_output_len": 0,
        "embedding_dim": embedding_dim,
        "embedding_module": embedding,
        "similarity_module": DotProductSimilarity(),
        "input_features_preproc_module": LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=max_length,
            embedding_dim=embedding_dim,
            dropout_rate=0.0,
        ),
        "output_postproc_module": L2NormEmbeddingPostprocessor(embedding_dim),
        "verbose": False,
    }
    if architecture == "hstu":
        return HSTU(
            **common,
            num_blocks=1,
            num_heads=1,
            linear_dim=embedding_dim,
            attention_dim=embedding_dim // 2,
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=0.0,
            attn_dropout_rate=0.0,
        )
    if architecture == "sasrec":
        return SASRec(
            **common,
            num_blocks=1,
            num_heads=1,
            ffn_hidden_dim=embedding_dim,
            ffn_activation_fn="relu",
            ffn_dropout_rate=0.0,
            activation_checkpoint=False,
        )
    raise ValueError(architecture)


def _exercise(torch, architecture: str, device: str) -> dict:
    model = _build_core(architecture, embedding_dim=16, max_length=5).to(device)
    ids = torch.tensor([[3, 4, 2, 0, 0], [5, 2, 0, 0, 0]], device=device)
    lengths = torch.tensor([3, 2], device=device)
    timestamps = torch.tensor(
        [[1, 2, 3, 0, 0], [1, 3, 0, 0, 0]], device=device
    )
    def compute_loss():
        output = model.encode(
            past_lengths=lengths,
            past_ids=ids,
            past_embeddings=model.get_item_embeddings(ids),
            past_payloads={"timestamps": timestamps},
        )
        target = torch.linspace(
            -0.5, 0.5, output.numel(), device=device, dtype=output.dtype
        ).reshape_as(output)
        return output, (output - target).square().mean()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output, loss = compute_loss()
        loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    architecture_parameter_prefix = "_hstu" if architecture == "hstu" else "attention_layers"
    selected_name, selected_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and architecture_parameter_prefix in name
    )
    flat_gradient = selected_parameter.grad.reshape(-1)
    selected_index = int(flat_gradient.abs().argmax())
    analytic_gradient = float(flat_gradient[selected_index])
    epsilon = 1e-3
    with torch.no_grad():
        flat_parameter = selected_parameter.reshape(-1)
        original = float(flat_parameter[selected_index])
        flat_parameter[selected_index] = original + epsilon
        _, plus_loss = compute_loss()
        flat_parameter[selected_index] = original - epsilon
        _, minus_loss = compute_loss()
        flat_parameter[selected_index] = original
    numeric_gradient = float((plus_loss - minus_loss) / (2 * epsilon))
    gradient_absolute_error = abs(analytic_gradient - numeric_gradient)
    gradient_tolerance = max(5e-3, 0.1 * abs(numeric_gradient))
    return {
        "architecture": architecture,
        "finite_output": bool(torch.isfinite(output).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": all(bool(torch.isfinite(grad).all()) for grad in gradients),
        "gradient_tensor_count": len(gradients),
        "finite_difference": {
            "absolute_error": gradient_absolute_error,
            "analytic": analytic_gradient,
            "epsilon": epsilon,
            "numeric": numeric_gradient,
            "parameter": selected_name,
            "passed": gradient_absolute_error <= gradient_tolerance,
            "tolerance": gradient_tolerance,
        },
        "nonzero_gradient_tensor_count": sum(
            int(bool((grad != 0).any())) for grad in gradients
        ),
        "output_shape": list(output.shape),
        "warning_messages": sorted({str(w.message) for w in caught}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import fbgemm_gpu
    import torch
    import torchrec
    import triton

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    torch.manual_seed(20260715)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260715)
    manual_load, sparse_library = _load_sparse_cumsum(torch, fbgemm_gpu)
    checks = [_exercise(torch, name, args.device) for name in ("hstu", "sasrec")]
    passed = all(
        check["finite_output"]
        and check["finite_loss"]
        and check["finite_gradients"]
        and check["nonzero_gradient_tensor_count"] > 0
        and check["finite_difference"]["passed"]
        for check in checks
    )
    result = {
        "schema_version": 1,
        "decision": "pass" if passed else "fail",
        "device": args.device,
        "qrels_read": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_root": str(UPSTREAM),
        "package_versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "fbgemm_gpu": getattr(fbgemm_gpu, "__version__", "unknown"),
            "torchrec": getattr(torchrec, "__version__", "unknown"),
            "triton": triton.__version__,
            "python": sys.version.split()[0],
        },
        "sparse_cumsum_manual_load": manual_load,
        "sparse_cumsum_library": sparse_library,
        "checks": checks,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
