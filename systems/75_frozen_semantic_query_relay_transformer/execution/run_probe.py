"""C75 route-only training, label-free A0, and staged exposed-fit A1."""

from __future__ import annotations

import argparse,hashlib,json,os,random,sys
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
import torch
from transformers import AutoModel

SYSTEM_ROOT=Path(__file__).resolve().parents[1];REPO_ROOT=SYSTEM_ROOT.parents[1]
for p in (str(SYSTEM_ROOT),str(REPO_ROOT/'src')):
    if p not in sys.path:sys.path.insert(0,p)
from execution.locking import atomic_json,load_config,sha256_file,timestamp,verify_execution_lock  # noqa:E402
from model.frozen_semantic_relay import MODES,PRIMARY,FrozenSemanticRelayLMRanker,listwise_loss  # noqa:E402
from myrec.eval.metrics import ScoredCandidate,ndcg_at_k,sort_candidates  # noqa:E402
from train.data_bridge import C75Store,iter_training_batches,iter_validation_batches,to_device  # noqa:E402
from train.gate_metrics import bootstrap,compare  # noqa:E402

FORWARD=("query_input_ids","query_attention_mask","query_content_mask","candidate_input_ids","candidate_attention_mask","candidate_content_mask","history_input_ids","history_attention_mask","history_content_mask","history_event_mask","candidate_mask","base_scores","item_only_scores","repeat_request","query_present")
SCORES=("base",PRIMARY,"wrong_history","coupled_value_relay","pooled_semantic_relay","factual_semantic_relay","primary_correction","wrong_correction")

def seed_all(seed):
 random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')

def make_model(cfg,mode,device):
 b=AutoModel.from_pretrained(REPO_ROOT/cfg['paths']['bge_snapshot'],local_files_only=True);r=cfg['model']
 return FrozenSemanticRelayLMRanker(backbone=b,mode=mode,trainable_last_lm_layers=r['inherited_trainable_last_lm_layers_argument'],input_dim=r['input_dim'],route_rank=r['route_rank'],max_history=cfg['selection']['max_history'],temperature=r['temperature'],profile_scale=r['profile_scale'],correction_scale=r['correction_scale'],route_init_std=r['route_init_std']).to(device)

def fw(t):return {k:t[k] for k in FORWARD}

def anchor_indices(store,cfg):
 seed=int(cfg['selection']['fixed_anchor_selection_seed'])
 return sorted(store.train_indices,key=lambda i:hashlib.sha256(f"c75-anchor:{seed}:{store.data.request_ids[i]}".encode()).digest())[:int(cfg['selection']['fixed_anchor_requests'])]

def anchor_loss(model,store,cfg,device):
 model.eval();indices=anchor_indices(store,cfg);rng=np.random.default_rng(int(cfg['selection']['fixed_anchor_candidate_seed']));total=0.;count=0
 with torch.inference_mode():
  for start in range(0,len(indices),int(cfg['training']['max_requests_per_batch'])):
   part=indices[start:start+int(cfg['training']['max_requests_per_batch'])];batch=store.collate(part,label_access=True,history_source='true',sampled_candidates=int(cfg['selection']['sampled_candidates']),rng=rng);t=to_device(batch,device);out=model(**fw(t));loss=listwise_loss(out,t['labels'],t['candidate_mask']);total+=float(loss.cpu())*len(part);count+=len(part)
 return total/count

def train_model(model,store,cfg,seed,device):
 tr=cfg['training'];before_hash=model.backbone_state_hash();initial_anchor=anchor_loss(model,store,cfg,device);params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=float(tr['route_learning_rate']),weight_decay=float(tr['weight_decay']));losses=[];active=set();steps=0
 for epoch in range(int(tr['epochs'])):
  model.train();rng=np.random.default_rng(seed+epoch*1009+75)
  for ix in iter_training_batches(store.train_indices,seed=seed+epoch*1009,batch_size=int(tr['max_requests_per_batch'])):
   b=store.collate(ix,label_access=True,history_source='true',sampled_candidates=int(cfg['selection']['sampled_candidates']),rng=rng);t=to_device(b,device)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):out=model(**fw(t));loss=listwise_loss(out,t['labels'],t['candidate_mask'])
   if not bool(torch.isfinite(loss)):raise RuntimeError(f'C75 {model.mode} nonfinite loss')
   opt.zero_grad(set_to_none=True);loss.backward()
   for n,p in model.named_parameters():
    if p.grad is not None:
     if not bool(torch.isfinite(p.grad).all()):raise RuntimeError(f'C75 nonfinite gradient {n}')
     if bool(p.grad.ne(0).any()):active.add(n)
   torch.nn.utils.clip_grad_norm_(params,float(tr['gradient_clip_norm']));opt.step();losses.append(float(loss.detach().cpu()));steps+=1
 final_anchor=anchor_loss(model,store,cfg,device);after_hash=model.backbone_state_hash();window=min(50,max(1,len(losses)//2));groups={'no_backbone_gradient':not any(n.startswith('backbone.') for n in active),'history_route_down':'history_route.down.weight' in active,'history_route_up':'history_route.up.weight' in active,'candidate_route_down':'candidate_route.down.weight' in active,'candidate_route_up':'candidate_route.up.weight' in active,'chronology_bias':'chronology_bias' in active}
 return {'steps':steps,'trace_loss_first':float(np.mean(losses[:window])),'trace_loss_last':float(np.mean(losses[-window:])),'trace_loss_decreased':float(np.mean(losses[-window:]))<float(np.mean(losses[:window])),'fixed_anchor_initial_loss':initial_anchor,'fixed_anchor_final_loss':final_anchor,'fixed_anchor_final_over_initial':final_anchor/initial_anchor,'finite':bool(np.isfinite(losses).all()),'gradient_groups':groups,'all_gradient_groups':all(groups.values()),'backbone_hash_before':before_hash,'backbone_hash_after':after_hash,'backbone_unchanged':before_hash==after_hash,'total_parameters':model.parameter_count(),'trainable_parameters':model.trainable_parameter_count(),'chronology_bias':model.chronology_bias.detach().float().cpu().tolist()}

def reverse_candidates(t):
 out=dict(t);rev=torch.arange(t['candidate_mask'].shape[1]-1,-1,-1,device=t['candidate_mask'].device)
 for n in ('candidate_input_ids','candidate_attention_mask','candidate_content_mask','candidate_mask','base_scores','item_only_scores','labels'):out[n]=t[n][:,rev]
 return out

def score_model(model,store,cfg,device,include_wrong):
 model.eval();rows={n:[] for n in ('true','wrong','base','correction','wrong_correction')};det=perm=noerr=qerr=reperr=0.;first=True
 with torch.inference_mode():
  for ix in iter_validation_batches(store,store.validation_indices,max_requests=int(cfg['training']['validation_max_requests_per_batch']),max_sequences=int(cfg['training']['max_encoded_sequences_per_batch'])):
   b=store.collate(ix,label_access=False,history_source='true');t=to_device(b,device)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):out=model(**fw(t))
   wo=None
   if include_wrong:
    wt=to_device(store.collate(ix,label_access=False,history_source='wrong'),device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):wo=model(**fw(wt))
   if first:
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
     repeated=model(**fw(t));rt=reverse_candidates(t);ro=model(**fw(rt));empty=dict(t);empty['history_event_mask']=torch.zeros_like(t['history_event_mask']);eo=model(**fw(empty));masked=dict(t);masked['query_present']=torch.zeros_like(t['query_present']);qo=model(**fw(masked));rep=dict(t);rep['repeat_request']=torch.ones_like(t['repeat_request']);rpo=model(**fw(rep))
    rev=torch.arange(out.scores.shape[1]-1,-1,-1,device=device);mask=t['candidate_mask'];base=t['base_scores'].float().masked_fill(~mask,0);item=t['item_only_scores'].float().masked_fill(~mask,0);det=float((out.scores-repeated.scores).abs().max().cpu());perm=float((out.scores-ro.scores[:,rev]).abs().max().cpu());noerr=float((eo.scores-base).abs().max().cpu());qerr=float((qo.scores-base).abs().max().cpu());reperr=float((rpo.scores-item).abs().max().cpu());first=False
   for r,count in enumerate(t['candidate_mask'].sum(-1).tolist()):
    count=int(count);rows['true'].append(out.scores[r,:count].float().cpu().numpy());rows['base'].append(t['base_scores'][r,:count].float().cpu().numpy());rows['correction'].append(out.correction[r,:count].float().cpu().numpy())
    if wo is not None:rows['wrong'].append(wo.scores[r,:count].float().cpu().numpy());rows['wrong_correction'].append(wo.correction[r,:count].float().cpu().numpy())
 return rows,{'deterministic_max_abs':det,'candidate_permutation_max_abs':perm,'nohistory_max_abs':noerr,'query_mask_max_abs':qerr,'repeat_max_abs':reperr,'validation_labels_opened':False}

def flatten(rows):
 off=[0]
 for r in rows:off.append(off[-1]+len(r))
 return np.asarray(off,dtype=np.int64),np.concatenate(rows).astype(np.float32,copy=False)

def unflatten(off,v):return [np.asarray(v[int(off[i]):int(off[i+1])],dtype=np.float32).copy() for i in range(len(off)-1)]

def ranking(rid,items,values):return [r.item_id for r in sort_candidates(rid,[ScoredCandidate(str(i),float(s)) for i,s in zip(items,values)])]

def activity(rids,item_ids,left,right):
 order=[];top=[]
 for rid,items,a,b in zip(rids,item_ids,left,right):
  ra=ranking(rid,items,a);rb=ranking(rid,items,b);order.append(ra!=rb);top.append(set(ra[:10])!=set(rb[:10]))
 return {'requests':len(order),'order_change_count':int(sum(order)),'order_change_fraction':float(np.mean(order)),'top10_change_count':int(sum(top)),'top10_change_fraction':float(np.mean(top))}

def run_seed(cfg,seed,device):
 _,lock_hash=verify_execution_lock(cfg);physical=int(cfg['resources']['seed_to_physical_gpu'][str(seed)])
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical) or str(device)!='cuda:0' or torch.cuda.device_count()!=1:raise RuntimeError('C75 GPU registration differs')
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8',':16:8'}:raise RuntimeError('C75 deterministic CUBLAS absent')
 store=C75Store(cfg,REPO_ROOT);split=json.loads((REPO_ROOT/cfg['paths']['artifact_root']/'split_manifest.json').read_text());expected=split['validation_candidate_hash']
 if store.candidate_hash(store.validation_indices)!=expected:raise RuntimeError('C75 candidate hash differs')
 root=REPO_ROOT/cfg['paths']['artifact_root'];ckroot=REPO_ROOT/cfg['paths']['checkpoint_root'];root.mkdir(parents=True,exist_ok=True);ckroot.mkdir(parents=True,exist_ok=True);allrows={};training={};scoring={};cks={}
 for mode in MODES:
  seed_all(seed);m=make_model(cfg,mode,device);training[mode]=train_model(m,store,cfg,seed,device);print(f'C75 seed={seed} mode={mode} trained',flush=True);ck=ckroot/f'seed_{seed}_{mode}.pt'
  if ck.exists():raise FileExistsError(ck)
  torch.save({'candidate_id':'c75','seed':seed,'mode':mode,'execution_lock_sha256':lock_hash,'state_dict':m.state_dict()},ck);cks[mode]={'path':str(ck.relative_to(REPO_ROOT)),'sha256':sha256_file(ck)};rows,sr=score_model(m,store,cfg,device,mode==PRIMARY);scoring[mode]=sr;allrows[mode]=rows['true']
  if mode==PRIMARY:allrows['base']=rows['base'];allrows['wrong_history']=rows['wrong'];allrows['primary_correction']=rows['correction'];allrows['wrong_correction']=rows['wrong_correction']
  print(f'C75 seed={seed} mode={mode} scored',flush=True);del m;torch.cuda.empty_cache()
 off,_=flatten(allrows['base']);sp=root/f'seed_{seed}_scores.npz';rp=root/f'seed_{seed}_report.json'
 if sp.exists() or rp.exists():raise FileExistsError(sp)
 with sp.open('wb') as h:np.savez(h,request_indices=np.asarray(store.validation_indices),offsets=off,**{n:flatten(allrows[n])[1] for n in SCORES})
 rids=[store.data.request_ids[i] for i in store.validation_indices];items=[store.data.candidate_ids(i) for i in store.validation_indices];acts={'primary_vs_base':activity(rids,items,allrows[PRIMARY],allrows['base']),'true_vs_wrong':activity(rids,items,allrows[PRIMARY],allrows['wrong_history']),'primary_vs_coupled':activity(rids,items,allrows[PRIMARY],allrows['coupled_value_relay']),'primary_vs_pooled':activity(rids,items,allrows[PRIMARY],allrows['pooled_semantic_relay']),'primary_vs_factual':activity(rids,items,allrows[PRIMARY],allrows['factual_semantic_relay'])};corr=np.concatenate(allrows['primary_correction']);wc=np.concatenate(allrows['wrong_correction']);e=cfg['evaluation'];mechanics={'all_finite':all(v['finite'] for v in training.values()),'all_fixed_anchor_decreased':all(v['fixed_anchor_final_over_initial']<=float(e['fixed_anchor_final_over_initial_loss_max']) for v in training.values()),'all_gradient_groups':all(v['all_gradient_groups'] for v in training.values()),'all_backbones_unchanged':all(v['backbone_unchanged'] for v in training.values()),'equal_parameters':len({(v['total_parameters'],v['trainable_parameters']) for v in training.values()})==1,'deterministic':all(v['deterministic_max_abs']<=e['deterministic_tolerance'] for v in scoring.values()),'candidate_permutation':all(v['candidate_permutation_max_abs']<=e['candidate_permutation_tolerance'] for v in scoring.values()),'nohistory_exact':all(v['nohistory_max_abs']<=e['exact_fallback_tolerance'] for v in scoring.values()),'query_mask_exact':all(v['query_mask_max_abs']<=e['exact_fallback_tolerance'] for v in scoring.values()),'repeat_exact':all(v['repeat_max_abs']<=e['exact_fallback_tolerance'] for v in scoring.values()),'candidate_hash':True,'validation_labels_closed_during_scoring':all(not v['validation_labels_opened'] for v in scoring.values()),'fresh_dev_test_qrels_closed':True};report={'candidate_id':'c75','created_at':timestamp(),'stage':'route_training_and_label_free_validation','status':'scored' if all(mechanics.values()) else 'failed_terminal','seed':seed,'physical_gpu':physical,'execution_lock_sha256':lock_hash,'validation_candidate_hash':expected,'training':training,'scoring':scoring,'activity':acts,'correction':{'primary_rms':float(np.sqrt(np.mean(corr**2))),'true_wrong_difference_rms':float(np.sqrt(np.mean((corr-wc)**2)))},'mechanics':mechanics,'checkpoints':cks,'scores':{'path':str(sp.relative_to(REPO_ROOT)),'sha256':sha256_file(sp)},'fit_train_labels_opened':True,'validation_labels_opened':False,'fresh_dev_test_qrels_opened':False};atomic_json(rp,report);return report

def a0(cfg):
 _,lock_hash=verify_execution_lock(cfg);root=REPO_ROOT/cfg['paths']['artifact_root'];reports=[]
 for seed in cfg['training']['seeds']:
  p=root/f'seed_{int(seed)}_report.json';r=json.loads(p.read_text());
  if sha256_file(REPO_ROOT/r['scores']['path'])!=r['scores']['sha256']:raise RuntimeError('C75 scores changed')
  reports.append((p,r))
 e=cfg['evaluation'];checks={'three_seed_reports':len(reports)==3,'every_seed_scored':all(r['status']=='scored' for _,r in reports),'same_lock':all(r['execution_lock_sha256']==lock_hash for _,r in reports),'same_candidate_hash':len({r['validation_candidate_hash'] for _,r in reports})==1,'every_seed_primary_active':all(r['activity']['primary_vs_base']['order_change_fraction']>=e['primary_order_change_fraction_min'] and r['activity']['primary_vs_base']['top10_change_fraction']>=e['primary_top10_change_fraction_min'] and r['correction']['primary_rms']>=e['primary_correction_rms_min'] for _,r in reports),'every_seed_wrong_active':all(r['activity']['true_vs_wrong']['order_change_fraction']>=e['wrong_order_change_fraction_min'] and r['activity']['true_vs_wrong']['top10_change_fraction']>=e['wrong_top10_change_fraction_min'] for _,r in reports),'every_seed_controls_distinct':all(all(r['activity'][n]['order_change_fraction']>=e['control_order_change_fraction_min'] and r['activity'][n]['top10_change_fraction']>=e['control_top10_change_fraction_min'] for n in ('primary_vs_coupled','primary_vs_pooled','primary_vs_factual')) for _,r in reports),'validation_labels_closed':all(not r['validation_labels_opened'] for _,r in reports),'fresh_dev_test_qrels_closed':True};passed=all(checks.values());value={'candidate_id':'c75','created_at':timestamp(),'stage':'A0_label_release_gate','status':'passed' if passed else 'failed_terminal','decision':'authorize_exposed_validation_labels' if passed else 'close_c75_before_validation_labels','execution_lock_sha256':lock_hash,'checks':checks,'seed_reports':{str(r['seed']):{'path':str(p.relative_to(REPO_ROOT)),'sha256':sha256_file(p)} for p,r in reports},'training_anchor':{str(r['seed']):{m:{'initial':v['fixed_anchor_initial_loss'],'final':v['fixed_anchor_final_loss'],'ratio':v['fixed_anchor_final_over_initial']} for m,v in r['training'].items()} for _,r in reports},'activity':{str(r['seed']):r['activity'] for _,r in reports},'correction':{str(r['seed']):r['correction'] for _,r in reports},'validation_labels_opened':False,'fresh_dev_test_qrels_opened':False};atomic_json(REPO_ROOT/cfg['paths']['a0_report'],value);return value

def load_scores(path):
 with np.load(path,allow_pickle=False) as v:
  off=np.asarray(v['offsets'],dtype=np.int64);return {n:unflatten(off,v[n]) for n in SCORES}

def ndcg_rows(rids,items,scores,labels):
 out=[]
 for rid,it,s,l in zip(rids,items,scores,labels):out.append(ndcg_at_k(ranking(rid,it,s),{str(x) for x,y in zip(it,l) if y>0},10))
 return np.asarray(out,dtype=np.float64)

def mean_rows(rows):return [np.mean(np.stack([r[i] for r in rows]),axis=0).astype(np.float32) for i in range(len(rows[0]))]

def a1(cfg):
 _,lock_hash=verify_execution_lock(cfg);gate=json.loads((REPO_ROOT/cfg['paths']['a0_report']).read_text())
 if gate['status']!='passed' or gate['execution_lock_sha256']!=lock_hash:raise PermissionError('C75 labels not authorized')
 store=C75Store(cfg,REPO_ROOT);split=json.loads((REPO_ROOT/cfg['paths']['artifact_root']/'split_manifest.json').read_text());expected=split['validation_candidate_hash'];seeds=[int(s) for s in cfg['training']['seeds']];sets=[load_scores(REPO_ROOT/cfg['paths']['artifact_root']/f'seed_{s}_scores.npz') for s in seeds];rids=[store.data.request_ids[i] for i in store.validation_indices];items=[store.data.candidate_ids(i) for i in store.validation_indices];labels=[store.labels(i) for i in store.validation_indices];ens={n:mean_rows([s[n] for s in sets]) for n in SCORES};ndcg={n:ndcg_rows(rids,items,v,labels) for n,v in ens.items() if n not in {'primary_correction','wrong_correction'}};e=cfg['evaluation'];refs=('base','coupled_value_relay','pooled_semantic_relay','factual_semantic_relay','wrong_history');comparisons=compare(rids,ndcg[PRIMARY],{n:ndcg[n] for n in refs},samples=e['bootstrap_samples'],seed=e['bootstrap_seed'],folds=e['hash_folds']);seed_diff={}
 for seed,scores in zip(seeds,sets):
  sn={n:ndcg_rows(rids,items,scores[n],labels) for n in (PRIMARY,*refs)};seed_diff[str(seed)]={n:float((sn[PRIMARY]-sn[n]).mean()) for n in refs}
 thresholds={'base':e['primary_minus_base_min'],'coupled_value_relay':e['primary_minus_coupled_min'],'pooled_semantic_relay':e['primary_minus_pooled_min'],'factual_semantic_relay':e['primary_minus_factual_min'],'wrong_history':e['true_minus_wrong_min']};checks={'candidate_hash_asserted':store.candidate_hash(store.validation_indices)==expected,'labels_opened_only_after_A0':True,'fresh_dev_test_qrels_closed':True}
 for n,t in thresholds.items():
  row=comparisons[n];checks[n+'_mean_threshold']=row['mean']>=t;checks[n+'_positive_interval']=row['percentile_95_ci'][0]>0;checks[n+'_each_seed_positive']=all(seed_diff[str(s)][n]>0 for s in seeds);checks[n+'_two_of_three_folds_positive']=sum(f['mean_difference']>0 for f in row['hash_folds'])>=2
 clicked=[]
 for c,l in zip(ens['primary_correction'],labels):
  p=np.asarray(l)>0
  if p.any():clicked.extend(np.asarray(c)[p].tolist())
 clicked_report=bootstrap(np.asarray(clicked),samples=e['bootstrap_samples'],seed=e['bootstrap_seed']+17);passed=all(checks.values());result={'candidate_id':'c75','created_at':timestamp(),'stage':'exposed_fit_A1','status':'passed' if passed else 'failed_terminal','decision':'authorize_same_graph_on_amazon_exposed_fit' if passed else 'close_c75_without_fresh_access','execution_lock_sha256':lock_hash,'validation_requests':len(rids),'candidate_hash':expected,'metrics':{n:float(v.mean()) for n,v in ndcg.items()},'comparisons':comparisons,'seed_differences':seed_diff,'clicked_primary_correction':clicked_report,'checks':checks,'fit_train_labels_opened':True,'validation_exposed_fit_labels_opened_after_A0':True,'fresh_features_scores_labels_opened':False,'dev_test_qrels_opened':False};atomic_json(REPO_ROOT/cfg['paths']['promoted_report'],result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('seed','a0','a1'),required=True);p.add_argument('--seed',type=int);a=p.parse_args();cfg=load_config()
 if a.stage=='seed':
  if a.seed is None:p.error('--seed required')
  v=run_seed(cfg,a.seed,torch.device('cuda:0'))
 elif a.stage=='a0':v=a0(cfg)
 else:v=a1(cfg)
 print(json.dumps({'stage':a.stage,'status':v['status'],'decision':v.get('decision')}))

if __name__=='__main__':main()
