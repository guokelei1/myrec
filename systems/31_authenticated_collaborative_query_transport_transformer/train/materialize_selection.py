"""Freeze C31 roles before any C31-A feature, score, or label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SYSTEM_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(SYSTEM_ROOT))
from train.authentication import load_user_ids  # noqa:E402
from train.structure import (ROLE_COUNTS,PackedStructure,candidate_key_sha256,donor_key_sha256,load_config,read_json,sha256_file,stable_key,write_json_once)  # noqa:E402


def length_bin(length:int,edges:list[int])->int:
    return next((i for i,e in enumerate(edges) if length<=e),len(edges))


def materialize(config_path:str|Path)->dict:
    config=load_config(config_path); paths=config['paths']; seed=int(config['selection']['seed'])
    for name,expected in (('c29_selection','c29_selection_sha256'),('c29_g0_report','c29_g0_report_sha256'),('c29_train_report','c29_train_report_sha256'),('c30_report','c30_report_sha256'),('packed_manifest','packed_manifest_sha256'),('label_free_request_metadata','label_free_request_metadata_sha256'),('label_free_request_manifest','label_free_request_manifest_sha256'),('schema_incident_report','schema_incident_report_sha256')):
        if sha256_file(paths[name])!=paths[expected]: raise RuntimeError(f'C31 source changed: {name}')
    source=read_json(paths['c29_selection']); g0=read_json(paths['c29_g0_report']); c29=read_json(paths['c29_train_report']); c30=read_json(paths['c30_report'])
    if g0.get('delayed_B_features_labels_scores_opened') is not False: raise PermissionError('C31 source delayed-B was materialized')
    if c29.get('delayed_B_features_labels_scores_opened') is not False or c30.get('delayed_B_escrow_dev_test_opened') is not False: raise PermissionError('C31 source A/B boundary differs')
    if c30.get('internal_A_labels_opened') is not True: raise PermissionError('C31 C30 terminal state differs')
    data=PackedStructure(paths['packed_train_root']); users=load_user_ids(paths['label_free_request_metadata'],data)
    nonrepeat=[i for i in range(len(data.request_ids)) if data.history_count(i)>0 and data.repeat_candidate_count(i)==0]
    prior_roles={role:[int(x) for x in row['indices']] for role,row in source['roles'].items()}; prior_donors={int(x) for row in source['wrong_history_donors'].values() for x in row['indices']}; prior_footprint={x for values in prior_roles.values() for x in values}|prior_donors
    roles={'fit':prior_roles['fit'],'internal_A':prior_roles['delayed_B'],'delayed_B':prior_roles['escrow'],'structural_repeat':prior_roles['structural_repeat'],'structural_nohistory':prior_roles['structural_nohistory']}
    new_pool=[i for i in nonrepeat if i not in prior_footprint]; ordered=sorted(new_pool,key=lambda i:(stable_key(seed,'escrow',data.request_ids[i]),i)); roles['escrow']=ordered[:ROLE_COUNTS['escrow']]
    if {k:len(v) for k,v in roles.items()}!=ROLE_COUNTS: raise AssertionError('C31 role counts differ')
    flat=[x for values in roles.values() for x in values]
    if len(flat)!=len(set(flat)): raise AssertionError('C31 roles overlap')
    outcome=set(flat); reserve=[i for i in nonrepeat if i not in outcome and i not in prior_footprint]
    edges=[int(x) for x in config['selection']['donor_length_bins']]; quantiles=int(config['selection']['donor_time_quantiles']); time_edges=np.quantile(np.asarray(data.timestamps[reserve],dtype=np.float64),np.linspace(0,1,quantiles+1)[1:-1])
    def bucket(i): return length_bin(data.history_count(i),edges),int(np.searchsorted(time_edges,float(data.timestamps[i]),side='right'))
    grouped={}; by_length={}
    for i in reserve: grouped.setdefault(bucket(i),[]).append(i); by_length.setdefault(bucket(i)[0],[]).append(i)
    for key,values in grouped.items(): values.sort(key=lambda i:(stable_key(seed,f'donor:{key}',data.request_ids[i]),i))
    for key,values in by_length.items(): values.sort(key=lambda i:(stable_key(seed,f'donor_length:{key}',data.request_ids[i]),i))
    reserve.sort(key=lambda i:(stable_key(seed,'donor_fallback',data.request_ids[i]),i))
    def donor_for(recipient):
        candidates=grouped.get(bucket(recipient),[]) or by_length.get(bucket(recipient)[0],[]) or reserve; start=int.from_bytes(stable_key(seed,'donor_start',data.request_ids[recipient])[0][:8],'big')%len(candidates); recipient_candidates=set(int(x) for x in data.candidate_indices(recipient))
        for step in range(len(candidates)):
            donor=int(candidates[(start+step)%len(candidates)])
            if users[donor]!=users[recipient] and recipient_candidates.isdisjoint(int(x) for x in data.history_indices(donor)): return donor
        raise RuntimeError('C31 donor unavailable')
    donors={
      'fit':[int(x) for x in source['wrong_history_donors']['fit']['indices']],
      'internal_A':[int(x) for x in source['wrong_history_donors']['delayed_B']['indices']],
      'delayed_B':[donor_for(i) for i in roles['delayed_B']],
    }
    if any(d in outcome for values in donors.values() for d in values): raise AssertionError('C31 donor intersects roles')
    for role,values in donors.items():
        for recipient,donor in zip(roles[role],values):
            if users[recipient]==users[donor] or not set(int(x) for x in data.candidate_indices(recipient)).isdisjoint(int(x) for x in data.history_indices(donor)): raise AssertionError('C31 donor contract differs')
    result={
      'candidate_id':'c31','selection_id':'c31_authenticated_collaborative_query_transport_selection_v1','status':'frozen_before_any_c31_A_feature_score_or_label','seed':seed,
      'roles':{role:{'indices':values,'request_ids':[data.request_ids[i] for i in values],'candidate_key_sha256':candidate_key_sha256(data,values)} for role,values in roles.items()},
      'wrong_history_donors':{role:{'indices':values,'request_ids':[data.request_ids[i] for i in values],'mapping_sha256':donor_key_sha256(data,roles[role],values)} for role,values in donors.items()},
      'donor_matching':{'history_length_edges':edges,'timestamp_quantiles':quantiles,'same_user_forbidden':True,'recipient_candidate_overlap_forbidden':True,'reserve_requests':len(reserve)},
      'sources':{'c29_selection_sha256':paths['c29_selection_sha256'],'c29_g0_report_sha256':paths['c29_g0_report_sha256'],'c29_train_report_sha256':paths['c29_train_report_sha256'],'c30_report_sha256':paths['c30_report_sha256']},
      'checks':{'fit_labels_previously_opened':True,'c30_A_labels_previously_opened_but_not_reused':True,'c31_internal_A_features_scores_labels_opened':False,'c31_delayed_B_features_scores_labels_opened':False,'roles_pairwise_disjoint':True,'strict_nonrepeat_fit_A_B_escrow':True,'donor_candidate_overlap_zero':True,'donor_user_overlap_zero':True,'selection_label_access':False,'c31_code_dev_test_qrels_metrics_read':False},
    }
    write_json_once(paths['selection'],result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args(); result=materialize(args.config); print(json.dumps({'selection':result['selection_id'],'role_counts':{k:len(v['indices']) for k,v in result['roles'].items()},'checks':result['checks']},indent=2,sort_keys=True))


if __name__=='__main__': main()
