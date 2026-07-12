"""Frozen C35 embeddings, causal authentication, and staged label access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any,Mapping,Sequence

import numpy as np
import torch

from train.structure import PackedStructure,candidate_key_sha256,read_json,sha256_file


@dataclass(frozen=True)
class CompactLabels:
    request_indices:np.ndarray; offsets:np.ndarray; values:np.ndarray
    def __post_init__(self):
        if len(self.offsets)!=len(self.request_indices)+1 or int(self.offsets[-1])!=len(self.values): raise ValueError('C35 compact labels differ')
    @property
    def positions(self): return {int(x):i for i,x in enumerate(self.request_indices)}
    def rows(self,indices:Sequence[int],counts:Sequence[int])->list[np.ndarray]:
        pos=self.positions; output=[]
        for raw,count in zip(indices,counts):
            index=int(raw)
            if index not in pos: raise PermissionError(f'C35 label unavailable: {index}')
            row=pos[index]; start,stop=int(self.offsets[row]),int(self.offsets[row+1])
            if stop-start!=int(count): raise ValueError('C35 label count differs')
            output.append(np.asarray(self.values[start:stop],dtype=np.float32).copy())
        return output


def open_original_labels(*,data:PackedStructure,indices:Sequence[int],path:str|Path,selection_path:str|Path,selection_sha256:str)->CompactLabels:
    if sha256_file(selection_path)!=selection_sha256: raise RuntimeError('C35 selection changed before labels')
    source=np.load(path,mmap_mode='r'); rows=[]; offsets=[0]
    for raw in indices:
        index=int(raw); start,stop=int(data.candidate_offsets[index]),int(data.candidate_offsets[index+1]); row=np.asarray(source[start:stop],dtype=np.float32).copy(); rows.append(row); offsets.append(offsets[-1]+len(row))
    return CompactLabels(np.asarray(indices,dtype=np.int64),np.asarray(offsets,dtype=np.int64),np.concatenate(rows).astype(np.float32,copy=False))


def zscore_row(values:np.ndarray)->np.ndarray:
    x=np.asarray(values,dtype=np.float32); scale=float(np.asarray(x,dtype=np.float64).std()) if len(x) else 0
    return np.zeros_like(x) if scale<=1e-8 else ((x-float(np.asarray(x,dtype=np.float64).mean()))/scale).astype(np.float32)


class FrozenTransportStore:
    def __init__(self,config:Mapping[str,Any]):
        self.config=config; self.data=PackedStructure(config['paths']['packed_train_root']); self.selection=read_json(config['paths']['selection']); root=Path(config['paths']['artifact_root'])
        self.feature_indices=np.load(root/'feature_request_indices.npy',mmap_mode='r'); self.score_offsets=np.load(root/'feature_candidate_offsets.npy',mmap_mode='r'); self.base_scores=np.load(root/'base_scores.npy',mmap_mode='r'); self.query_embeddings=np.load(root/'query_embeddings.npy',mmap_mode='r'); self.auth_request_indices=np.load(root/'authentication_request_indices.npy',mmap_mode='r'); self.auth_true_offsets=np.load(root/'auth_true_offsets.npy',mmap_mode='r'); self.auth_true_items=np.load(root/'auth_true_items.npy',mmap_mode='r'); self.auth_wrong_offsets=np.load(root/'auth_wrong_offsets.npy',mmap_mode='r'); self.auth_wrong_items=np.load(root/'auth_wrong_items.npy',mmap_mode='r'); self.raw_items=np.load(config['paths']['raw_item_embeddings'],mmap_mode='r')
        self.feature_position={int(x):i for i,x in enumerate(self.feature_indices)}; self.auth_position={int(x):i for i,x in enumerate(self.auth_request_indices)}
        if len(self.feature_position)!=len(self.feature_indices): raise ValueError('C35 feature requests overlap')
        if not np.array_equal(self.feature_indices,self.auth_request_indices): raise ValueError('C35 feature/auth requests differ')
        if len(self.score_offsets)!=len(self.feature_indices)+1 or int(self.score_offsets[-1])!=len(self.base_scores): raise ValueError('C35 score offsets differ')
        expected_dim=int(config['model']['embedding_dim'])
        if self.query_embeddings.shape!=(len(self.feature_indices),expected_dim): raise ValueError('C35 query embedding shape differs')
        if self.raw_items.ndim!=2 or self.raw_items.shape[1]!=expected_dim: raise ValueError('C35 item embedding shape differs')
    def role_indices(self,role): return [int(x) for x in self.selection['roles'][role]['indices']]
    def candidate_hash(self,indices): return candidate_key_sha256(self.data,indices)
    def candidate_count(self,index): return int(self.data.candidate_offsets[index+1]-self.data.candidate_offsets[index])
    def candidate_embedding_indices(self,index): return self.data.candidate_indices(index).astype(np.int64,copy=False)
    def candidate_item_ids(self,index):
        start,stop=int(self.data.candidate_offsets[index]),int(self.data.candidate_offsets[index+1]); return np.asarray(self.data.candidate_item_ids[start:stop]).copy()
    def base_row(self,index):
        row=self.feature_position[int(index)]; start,stop=int(self.score_offsets[row]),int(self.score_offsets[row+1])
        if stop-start!=self.candidate_count(int(index)): raise ValueError('C35 base-score candidate count differs')
        return zscore_row(np.asarray(self.base_scores[start:stop],dtype=np.float32))
    def query(self,index): return np.asarray(self.query_embeddings[self.feature_position[int(index)]],dtype=np.float32)
    def item_embeddings(self,indices): return np.asarray(self.raw_items[np.asarray(indices,dtype=np.int64)],dtype=np.float32)
    def authenticated_history(self,index,source):
        row=self.auth_position[int(index)]
        if source=='true': start,stop=int(self.auth_true_offsets[row]),int(self.auth_true_offsets[row+1]); return np.asarray(self.auth_true_items[start:stop],dtype=np.int64)
        if source=='wrong': start,stop=int(self.auth_wrong_offsets[row]),int(self.auth_wrong_offsets[row+1]); return np.asarray(self.auth_wrong_items[start:stop],dtype=np.int64)
        if source=='none': return np.empty(0,dtype=np.int64)
        raise ValueError('C35 history source differs')
    def has_repeat(self,index): return bool(set(int(x) for x in self.candidate_embedding_indices(index))&set(int(x) for x in self.data.history_indices(index)))
    def item_only_row(self,index):
        candidates=self.candidate_embedding_indices(index); history=self.data.history_indices(index).astype(np.int64,copy=False); start,stop=int(self.data.history_offsets[index]),int(self.data.history_offsets[index+1]); weights=np.asarray(self.data.history_event_weights[start:stop],dtype=np.float32)
        if len(history): reverse=np.maximum(len(history)-np.arange(len(history)),1).astype(np.float32); component=3*((candidates[:,None]==history[None,:]).astype(np.float32)*(weights/np.sqrt(reverse))[None,:]).sum(1)
        else: component=np.zeros(len(candidates),np.float32)
        beta=float(self.config['base']['item_only_beta']); return beta*self.base_row(index)+(1-beta)*zscore_row(component)


def to_tensor(value:np.ndarray,device:torch.device)->torch.Tensor: return torch.from_numpy(np.asarray(value,dtype=np.float32)).to(device)
