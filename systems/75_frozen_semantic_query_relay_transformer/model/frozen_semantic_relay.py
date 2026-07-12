"""C75 immutable pretrained-LM carrier plus trainable query relay."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
SOURCE = (
    REPO_ROOT
    / "systems/74_semantic_conservative_query_relay_transformer/model/adaptive_semantic_relay.py"
)
SPEC = importlib.util.spec_from_file_location("c74_adaptive_model_for_c75", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("C75 cannot load C74 semantic relay core")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PRIMARY = MODULE.PRIMARY
MODES = MODULE.MODES
AdaptiveSemanticRelayOutput = MODULE.AdaptiveSemanticRelayOutput
listwise_loss = MODULE.listwise_loss


class FrozenSemanticRelayLMRanker(MODULE.AdaptiveSemanticRelayLMRanker):
    """Keep the pretrained token coordinate bit-exact throughout route training."""

    def _configure_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenSemanticRelayLMRanker":
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode_flat_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return super()._encode_flat_tokens(input_ids, attention_mask)

    def backbone_state_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.backbone.state_dict().items()):
            digest.update(name.encode())
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(memoryview(array))
        return digest.hexdigest()

    def backbone_trainable_names(self) -> list[str]:
        return []
