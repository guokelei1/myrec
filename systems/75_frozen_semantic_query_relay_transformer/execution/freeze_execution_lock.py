"""Freeze C75 route-only training and data after passed G0."""

from __future__ import annotations
import json,sys
from pathlib import Path
SYSTEM_ROOT=Path(__file__).resolve().parents[1];REPO_ROOT=SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:sys.path.insert(0,str(SYSTEM_ROOT))
from execution.locking import atomic_json,load_config,sha256_file,timestamp,verify_g0_lock  # noqa:E402

SOURCES=("systems/75_frozen_semantic_query_relay_transformer/configs/kuai_probe.yaml","systems/75_frozen_semantic_query_relay_transformer/model/frozen_semantic_relay.py","systems/75_frozen_semantic_query_relay_transformer/train/data_bridge.py","systems/75_frozen_semantic_query_relay_transformer/train/gate_metrics.py","systems/75_frozen_semantic_query_relay_transformer/execution/locking.py","systems/75_frozen_semantic_query_relay_transformer/execution/freeze_execution_lock.py","systems/75_frozen_semantic_query_relay_transformer/execution/run_probe.py","src/myrec/eval/metrics.py")

def data_paths(c):
 a=c['paths']['c26_artifact_root'];p=c['paths']['packed_train_root'];s=c['paths']['bge_snapshot']
 return (c['paths']['c26_selection'],f'{a}/feature_request_indices.npy',f'{a}/feature_candidate_offsets.npy',f'{a}/base_scores.npy',f'{a}/item_embedding_indices.npy',f'{a}/item_token_ids.npy',f'{a}/item_attention_mask.npy',f'{a}/item_content_mask.npy',f'{a}/query_token_ids.npy',f'{a}/query_attention_mask.npy',f'{a}/query_content_mask.npy',f'{a}/fit_request_indices.npy',f'{a}/fit_label_offsets.npy',f'{a}/fit_labels.npy',f'{p}/request_ids.jsonl',f'{p}/candidate_offsets.npy',f'{p}/candidate_embedding_indices.npy',f'{p}/candidate_item_ids.npy',f'{p}/history_offsets.npy',f'{p}/history_embedding_indices.npy',f'{p}/history_event_weights.npy',f'{s}/config.json',f'{s}/model.safetensors',f"{c['paths']['artifact_root']}/split_manifest.json",c['paths']['g0_lock'],c['paths']['g0_report'],c['paths']['c74_design_report'],c['paths']['c74_a0_report'],c['paths']['c74_adaptive_model'],c['paths']['c74_data_bridge'],'systems/64_end_to_end_lm_representation_probe/train/data.py')

def main():
 c=load_config();_,g0h=verify_g0_lock(c);g0p=REPO_ROOT/c['paths']['g0_report'];g=json.loads(g0p.read_text())
 if g['status']!='passed' or g['g0_lock_sha256']!=g0h:raise PermissionError('C75 execution requires G0')
 t=REPO_ROOT/c['paths']['execution_lock'];v={'candidate_id':'c75','created_at':timestamp(),'decision':'authorize_three_route_only_exposed_fit_seeds','g0_lock_sha256':g0h,'g0_report_sha256':sha256_file(g0p),'source_sha256':{x:sha256_file(REPO_ROOT/x) for x in SOURCES},'data_sha256':{x:sha256_file(REPO_ROOT/x) for x in data_paths(c)},'outcome_boundary':{'fit_train_labels_authorized':True,'validation_labels_before_A0':False,'fresh_dev_test_qrels_opened':False}};atomic_json(t,v);print(json.dumps({'path':str(t),'sha256':sha256_file(t)}))

if __name__=='__main__':main()
