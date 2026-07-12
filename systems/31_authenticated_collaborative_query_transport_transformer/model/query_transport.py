"""Authenticated collaborative query transport in a shared LM embedding space."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


PRIMARY="collaborative_query_transport"
MODES=(PRIMARY,"semantic_identity_transport","unauthenticated_query_transport","uniform_history_transport")


class LowRankQueryTransport(nn.Module):
    def __init__(self,*,dim:int,rank:int,temperature:float,profile_scale:float,correction_scale:float,seed:int):
        super().__init__(); self.dim=int(dim); self.rank=int(rank); self.temperature=float(temperature); self.profile_scale=float(profile_scale); self.correction_scale=float(correction_scale)
        if min(self.dim,self.rank)<=0 or min(self.temperature,self.profile_scale,self.correction_scale)<=0: raise ValueError('C31 dimensions/scales must be positive')
        self.down=nn.Linear(self.dim,self.rank,bias=False); self.up=nn.Linear(self.rank,self.dim,bias=False); generator=torch.Generator().manual_seed(int(seed)); nn.init.normal_(self.down.weight,std=0.02,generator=generator); nn.init.zeros_(self.up.weight)

    def adapt(self,x:torch.Tensor)->torch.Tensor:
        return F.normalize(x+self.up(self.down(x)),dim=-1,eps=1e-6)

    def forward(self,query:torch.Tensor,history:torch.Tensor,candidates:torch.Tensor,*,uniform_history:bool=False)->torch.Tensor:
        if query.ndim!=1 or history.ndim!=2 or candidates.ndim!=2 or query.shape[0]!=self.dim or history.shape[1]!=self.dim or candidates.shape[1]!=self.dim: raise ValueError('C31 embedding shape differs')
        if len(history)==0: return torch.zeros(len(candidates),device=candidates.device,dtype=candidates.dtype)
        raw_query=F.normalize(query,dim=-1,eps=1e-6); raw_history=F.normalize(history,dim=-1,eps=1e-6)
        weights=torch.full((len(history),),1.0/len(history),device=history.device,dtype=history.dtype) if uniform_history else torch.softmax(raw_history.mv(raw_query)/self.temperature,dim=0)
        q=self.adapt(raw_query); h=self.adapt(raw_history); c=self.adapt(F.normalize(candidates,dim=-1,eps=1e-6)); profile=(weights[:,None]*h).sum(0); personalized=F.normalize(q+self.profile_scale*profile,dim=-1,eps=1e-6)
        return self.correction_scale*(c.mv(personalized)-c.mv(q))

    def trainable_parameter_count(self)->int: return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrozenBGETransportRanker(nn.Module):
    """Reference end-to-end form; cached embeddings are an exact optimization."""
    def __init__(self,snapshot:str,transport:LowRankQueryTransport):
        super().__init__(); self.encoder=AutoModel.from_pretrained(snapshot,local_files_only=True); self.encoder.requires_grad_(False); self.encoder.eval(); self.transport=transport

    def encode(self,input_ids:torch.Tensor,attention_mask:torch.Tensor)->torch.Tensor:
        with torch.no_grad(): state=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
        return F.normalize(state.float(),dim=-1,eps=1e-6)

    def forward(
        self,
        query_input_ids:torch.Tensor,
        query_attention_mask:torch.Tensor,
        history_input_ids:torch.Tensor,
        history_attention_mask:torch.Tensor,
        candidate_input_ids:torch.Tensor,
        candidate_attention_mask:torch.Tensor,
        *,
        uniform_history:bool=False,
    )->torch.Tensor:
        query=self.encode(query_input_ids,query_attention_mask)
        if len(query)!=1: raise ValueError('C31 expects one query per request')
        history=(self.encode(history_input_ids,history_attention_mask) if len(history_input_ids)
                 else query.new_empty((0,self.transport.dim)))
        candidates=self.encode(candidate_input_ids,candidate_attention_mask)
        return self.transport(query[0],history,candidates,uniform_history=uniform_history)
