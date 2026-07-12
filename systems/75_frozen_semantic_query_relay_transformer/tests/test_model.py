from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from model.frozen_semantic_relay import MODES, FrozenSemanticRelayLMRanker, listwise_loss  # noqa: E402


class FakeBackbone(nn.Module):
    def __init__(self, width=16, layers=4):
        super().__init__(); self.config=SimpleNamespace(hidden_size=width)
        self.embeddings=nn.Embedding(64,width); self.encoder=nn.Module()
        self.encoder.layer=nn.ModuleList([nn.Linear(width,width) for _ in range(layers)])
        self.dropout=nn.Dropout(0.5)
    def forward(self,*,input_ids,attention_mask)->Any:
        x=self.dropout(self.embeddings(input_ids))
        for layer in self.encoder.layer: x=x+torch.tanh(layer(x))
        return SimpleNamespace(last_hidden_state=x*attention_mask[...,None])


def make(mode):
    torch.manual_seed(3)
    return FrozenSemanticRelayLMRanker(
        backbone=FakeBackbone(),mode=mode,trainable_last_lm_layers=2,input_dim=16,
        route_rank=4,max_history=3,temperature=.1,profile_scale=1,
        correction_scale=3,route_init_std=.02)


def batch():
    torch.manual_seed(5);b,c,h,l=2,5,3,6
    q=torch.randint(1,63,(b,l));ci=torch.randint(1,63,(b,c,l));hi=torch.randint(1,63,(b,h,l))
    qm=torch.ones_like(q,dtype=torch.bool);cm=torch.ones_like(ci,dtype=torch.bool);hm=torch.ones_like(hi,dtype=torch.bool)
    return {"query_input_ids":q,"query_attention_mask":qm,"query_content_mask":qm,
        "candidate_input_ids":ci,"candidate_attention_mask":cm,"candidate_content_mask":cm,
        "history_input_ids":hi,"history_attention_mask":hm,"history_content_mask":hm,
        "history_event_mask":torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool),
        "candidate_mask":torch.tensor([[1]*5,[1,1,1,0,0]],dtype=torch.bool),
        "base_scores":torch.randn(b,c),"item_only_scores":torch.randn(b,c),
        "repeat_request":torch.zeros(b,dtype=torch.bool),"query_present":torch.ones(b,dtype=torch.bool)}


def test_backbone_is_frozen_and_forced_eval() -> None:
    value=make(MODES[0]); assert not any(p.requires_grad for p in value.backbone.parameters())
    value.train(); assert not value.backbone.training; assert value.training
    assert value.backbone_trainable_names()==[]


def test_backbone_hash_unchanged_and_routes_train() -> None:
    value=make(MODES[0]); rows=batch(); labels=torch.zeros_like(rows["base_scores"]);labels[:,0]=1
    before=value.backbone_state_hash();opt=torch.optim.AdamW([p for p in value.parameters() if p.requires_grad],lr=.01);active=set()
    for _ in range(3):
        out=value(**rows);loss=listwise_loss(out,labels,rows["candidate_mask"]);opt.zero_grad();loss.backward()
        active|={n for n,p in value.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any())};opt.step()
    assert value.backbone_state_hash()==before
    assert not any(name.startswith("backbone.") for name in active)
    assert "history_route.down.weight" in active and "candidate_route.down.weight" in active


def test_fallbacks_permutation_and_capacity() -> None:
    rows=batch();counts=set()
    for mode in MODES:
        value=make(mode).eval();counts.add((value.parameter_count(),value.trainable_parameter_count()))
        no=dict(rows);no["history_event_mask"]=torch.zeros_like(rows["history_event_mask"])
        assert torch.equal(value(**no).scores,rows["base_scores"].masked_fill(~rows["candidate_mask"],0))
        original=value(**rows).scores;rev=torch.arange(4,-1,-1);changed=dict(rows)
        for name in ("candidate_input_ids","candidate_attention_mask","candidate_content_mask","candidate_mask","base_scores","item_only_scores"):
            changed[name]=rows[name][:,rev]
        assert torch.allclose(original,value(**changed).scores[:,rev],atol=2e-6,rtol=0)
    assert len(counts)==1
