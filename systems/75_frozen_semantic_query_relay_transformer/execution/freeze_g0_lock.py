"""Freeze C75 before label-free G0."""

from __future__ import annotations

import json
from pathlib import Path
import sys


SYSTEM_ROOT=Path(__file__).resolve().parents[1];REPO_ROOT=SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:sys.path.insert(0,str(SYSTEM_ROOT))
from execution.locking import atomic_json,load_config,sha256_file,timestamp  # noqa:E402

SOURCES=(
"systems/75_frozen_semantic_query_relay_transformer/README.md",
"systems/75_frozen_semantic_query_relay_transformer/configs/kuai_probe.yaml",
"systems/75_frozen_semantic_query_relay_transformer/model/frozen_semantic_relay.py",
"systems/75_frozen_semantic_query_relay_transformer/train/data_bridge.py",
"systems/75_frozen_semantic_query_relay_transformer/execution/locking.py",
"systems/75_frozen_semantic_query_relay_transformer/execution/freeze_g0_lock.py",
"systems/75_frozen_semantic_query_relay_transformer/execution/run_g0.py",
"systems/75_frozen_semantic_query_relay_transformer/tests/test_model.py",
"systems/75_frozen_semantic_query_relay_transformer/tests/test_protocol.py",
"systems/75_frozen_semantic_query_relay_transformer/notes/proposal.md",
"systems/75_frozen_semantic_query_relay_transformer/notes/mechanism_fingerprint.md",
"systems/75_frozen_semantic_query_relay_transformer/notes/nearest_neighbors.md",
"systems/75_frozen_semantic_query_relay_transformer/notes/train_gate_protocol.md",
"systems/75_frozen_semantic_query_relay_transformer/notes/preimplementation_review.md",
)

def main():
    cfg=load_config();design=REPO_ROOT/cfg['paths']['c74_design_report'];a0=REPO_ROOT/cfg['paths']['c74_a0_report']
    d=json.loads(design.read_text());prior=json.loads(a0.read_text())
    if not d['passed'] or d['decision']!='pass_authorize_pretrained_probe':raise PermissionError('C75 requires passed C74 design gate')
    if prior['validation_labels_opened']:raise PermissionError('C75 premise requires C74 labels closed')
    target=REPO_ROOT/cfg['paths']['g0_lock'];value={
      'candidate_id':'c75','created_at':timestamp(),'decision':'authorize_one_label_free_G0',
      'source_sha256':{p:sha256_file(REPO_ROOT/p) for p in SOURCES},
      'authority_sha256':{
        cfg['paths']['c74_design_report']:sha256_file(design),
        cfg['paths']['c74_a0_report']:sha256_file(a0),
        cfg['paths']['c74_adaptive_model']:sha256_file(REPO_ROOT/cfg['paths']['c74_adaptive_model']),
        cfg['paths']['c74_data_bridge']:sha256_file(REPO_ROOT/cfg['paths']['c74_data_bridge']),
      },
      'outcome_boundary':{'fit_labels_opened':False,'validation_labels_opened':False,'fresh_dev_test_qrels_opened':False}}
    atomic_json(target,value);print(json.dumps({'path':str(target),'sha256':sha256_file(target)}))

if __name__=='__main__':main()
