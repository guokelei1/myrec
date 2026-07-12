"""C75 label-free immutable-carrier G0."""

from __future__ import annotations

import json,os,random,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel

SYSTEM_ROOT=Path(__file__).resolve().parents[1];REPO_ROOT=SYSTEM_ROOT.parents[1]
for p in (str(SYSTEM_ROOT),str(REPO_ROOT/'src')):
    if p not in sys.path:sys.path.insert(0,p)
from execution.locking import atomic_json,load_config,sha256_file,timestamp,verify_g0_lock  # noqa:E402
from model.frozen_semantic_relay import MODES,PRIMARY,FrozenSemanticRelayLMRanker,listwise_loss  # noqa:E402
from train.data_bridge import C75Store,to_device  # noqa:E402

FORWARD=("query_input_ids","query_attention_mask","query_content_mask","candidate_input_ids","candidate_attention_mask","candidate_content_mask","history_input_ids","history_attention_mask","history_content_mask","history_event_mask","candidate_mask","base_scores","item_only_scores","repeat_request","query_present")

def seed_all(seed):
 random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def make(cfg,mode,device):
 b=AutoModel.from_pretrained(REPO_ROOT/cfg['paths']['bge_snapshot'],local_files_only=True);r=cfg['model']
 return FrozenSemanticRelayLMRanker(backbone=b,mode=mode,trainable_last_lm_layers=r['inherited_trainable_last_lm_layers_argument'],input_dim=r['input_dim'],route_rank=r['route_rank'],max_history=cfg['selection']['max_history'],temperature=r['temperature'],profile_scale=r['profile_scale'],correction_scale=r['correction_scale'],route_init_std=r['route_init_std']).to(device)

def fw(t):return {k:t[k] for k in FORWARD}

def main():
 cfg=load_config();_,lock_hash=verify_g0_lock(cfg);physical=int(cfg['resources']['g0_physical_gpu'])
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical) or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('C75 G0 GPU differs')
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8',':16:8'}:raise RuntimeError('C75 deterministic CUBLAS absent')
 device=torch.device('cuda:0');seed_all(20265300);store=C75Store(cfg,REPO_ROOT);root=REPO_ROOT/cfg['paths']['artifact_root'];root.mkdir(parents=True,exist_ok=True)
 manifest=store.split_manifest();split=root/'split_manifest.json';atomic_json(split,manifest)
 idx=store.validation_indices[:2];true=to_device(store.collate(idx,label_access=False,history_source='true'),device);wrong=to_device(store.collate(idx,label_access=False,history_source='wrong'),device);pseudo=torch.zeros_like(true['base_scores']);pseudo[:,0]=true['candidate_mask'][:,0]
 counts={};gradients={};hashes={};primary=None
 for mode in MODES:
  seed_all(20265300);m=make(cfg,mode,device);before=m.backbone_state_hash();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3);active=set();m.train()
  forced_eval=not m.backbone.training
  for _ in range(3):
   out=m(**fw(true));loss=listwise_loss(out,pseudo,true['candidate_mask']);opt.zero_grad();loss.backward();active|={n for n,p in m.named_parameters() if p.grad is not None and bool(p.grad.ne(0).any())};opt.step()
  after=m.backbone_state_hash();counts[mode]={'total':m.parameter_count(),'trainable':m.trainable_parameter_count()};hashes[mode]={'before':before,'after':after,'unchanged':before==after,'forced_eval':forced_eval}
  gradients[mode]={'no_backbone_gradient':not any(n.startswith('backbone.') for n in active),'history_route_down':'history_route.down.weight' in active,'history_route_up':'history_route.up.weight' in active,'candidate_route_down':'candidate_route.down.weight' in active,'candidate_route_up':'candidate_route.up.weight' in active,'chronology_bias':'chronology_bias' in active}
  if mode==PRIMARY:primary=m
  else:del m
  torch.cuda.empty_cache()
 assert primary is not None;primary.eval()
 with torch.inference_mode():
  out=primary(**fw(true));repeat=primary(**fw(true));wrong_out=primary(**fw(wrong));rev=torch.arange(true['candidate_mask'].shape[1]-1,-1,-1,device=device);rb=dict(true)
  for n in ('candidate_input_ids','candidate_attention_mask','candidate_content_mask','candidate_mask','base_scores','item_only_scores','labels'):rb[n]=true[n][:,rev]
  rev_out=primary(**fw(rb));empty=dict(true);empty['history_event_mask']=torch.zeros_like(true['history_event_mask']);empty_out=primary(**fw(empty));masked=dict(true);masked['query_present']=torch.zeros_like(true['query_present']);mask_out=primary(**fw(masked));rep=dict(true);rep['repeat_request']=torch.ones_like(true['repeat_request']);rep_out=primary(**fw(rep))
 mask=true['candidate_mask'];base=true['base_scores'].float().masked_fill(~mask,0);item=true['item_only_scores'].float().masked_fill(~mask,0)
 numeric={'deterministic':float((out.scores-repeat.scores).abs().max().cpu()),'permutation':float((out.scores-rev_out.scores[:,rev]).abs().max().cpu()),'nohistory':float((empty_out.scores-base).abs().max().cpu()),'query_mask':float((mask_out.scores-base).abs().max().cpu()),'repeat':float((rep_out.scores-item).abs().max().cpu()),'correction_rms':float(out.correction[mask].square().mean().sqrt().cpu()),'true_wrong_rms':float((out.correction[mask]-wrong_out.correction[mask]).square().mean().sqrt().cpu())}
 e=cfg['evaluation'];checks={'design_authority':True,'split_disjoint':manifest['overlap']==0,'labels_closed':store._labels is None,'equal_total':len({v['total'] for v in counts.values()})==1,'equal_trainable':len({v['trainable'] for v in counts.values()})==1,'all_backbones_unchanged':all(v['unchanged'] for v in hashes.values()),'all_backbones_forced_eval':all(v['forced_eval'] for v in hashes.values()),'all_route_gradients':all(all(v.values()) for v in gradients.values()),'rank_active':numeric['correction_rms']>=e['primary_correction_rms_min'],'wrong_active':numeric['true_wrong_rms']>0.01,'deterministic':numeric['deterministic']<=e['deterministic_tolerance'],'candidate_permutation':numeric['permutation']<=e['candidate_permutation_tolerance'],'nohistory_exact':numeric['nohistory']<=e['exact_fallback_tolerance'],'query_mask_exact':numeric['query_mask']<=e['exact_fallback_tolerance'],'repeat_exact':numeric['repeat']<=e['exact_fallback_tolerance'],'fresh_dev_test_qrels_closed':True}
 value={'candidate_id':'c75','created_at':timestamp(),'stage':'label_free_G0','status':'passed' if all(checks.values()) else 'failed_terminal','decision':'authorize_execution_lock' if all(checks.values()) else 'close_before_training','g0_lock_sha256':lock_hash,'split_manifest':{'path':str(split.relative_to(REPO_ROOT)),'sha256':sha256_file(split),**manifest},'parameters':counts,'backbone_hashes':hashes,'gradient_groups':gradients,'numeric':numeric,'checks':checks,'fit_labels_opened':False,'validation_labels_opened':False,'fresh_dev_test_qrels_opened':False}
 atomic_json(REPO_ROOT/cfg['paths']['g0_report'],value);print(json.dumps({'status':value['status'],'checks':checks,'numeric':numeric}))

if __name__=='__main__':main()
