from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]; REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0,str(SYSTEM_ROOT)); sys.path.append(str(REPO_ROOT/"src")); sys.path.append(str(REPO_ROOT/"systems/38_cross_domain_global_tangent_transfer"))
from execution.c71_helpers import load_helpers  # noqa: E402
from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.selection import candidate_key_sha256  # noqa: E402
from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[offsets[i]:offsets[i+1]],dtype=np.float32).copy() for i in range(len(offsets)-1)]


def comparison_pass(row: Mapping[str,Any], minimum: float) -> bool:
    return bool(row["mean"]>=minimum and row["percentile_95_ci"][0]>0 and all(fold["mean_difference"]>0 for fold in row["hash_folds"]))


def main() -> None:
    config=load_config(SYSTEM_ROOT/"configs/diagnostic.yaml"); _,lock_hash=verify_execution_lock(config); paths=config["paths"]
    root=REPO_ROOT/paths["artifact_root"]; a0_path=root/"a0_report.json"; a0=json.loads(a0_path.read_text()); target=REPO_ROOT/paths["promoted_report"]
    if a0["execution_lock_sha256"]!=lock_hash: raise RuntimeError("C72 A0 lock differs")
    if not a0["passed_A0"]:
        value={"schema":"myrec.c72.gate.v1","candidate_id":"c72","created_at":timestamp(),"A0_passed":False,"passed":False,"decision":"failed_A0_terminal","claim_boundary":{"fresh":False,"formulation_only":True,"dev_test_qrels":False}}
        atomic_json(target,value); print(target.relative_to(REPO_ROOT)); print(value["decision"]); return
    selection=json.loads((REPO_ROOT/paths["selection"]).read_text()); wanted={row["request_id"] for row in selection["targets"]}; raw={}
    with (REPO_ROOT/paths["records_train"]).open() as handle:
        for line in handle:
            row=json.loads(line); rid=str(row["request_id"])
            if rid in wanted: raw[rid]=row
    if set(raw)!=wanted: raise RuntimeError("C72 exposed labels incomplete")
    structural=[]; labels=[]
    for expected in selection["targets"]:
        row=raw[expected["request_id"]]; ids=[str(value["item_id"]) for value in row["candidates"]]
        if ids!=expected["candidate_ids"]: raise RuntimeError("C72 candidate order changed")
        structural.append({"request_id":expected["request_id"],"candidate_ids":ids}); labels.append(np.asarray([float(value[config["evaluation"]["label_field"]]) for value in row["candidates"]],dtype=np.float32))
    if candidate_key_sha256(structural)!=selection["candidate_key_sha256"]: raise RuntimeError("C72 candidate hash changed at evaluation")
    score_path=REPO_ROOT/a0["score_artifact"]["path"]
    if sha256_file(score_path)!=a0["score_artifact"]["sha256"]: raise RuntimeError("C72 score artifact changed")
    helpers=load_helpers()
    with np.load(score_path,allow_pickle=False) as source:
        offsets=np.asarray(source["offsets"],dtype=np.int64); scores={name:unflatten(offsets,source[name]) for name in helpers.SCORE_NAMES}
    request_ids=[row["request_id"] for row in selection["targets"]]; item_ids=[row["candidate_ids"] for row in selection["targets"]]
    metrics={}
    for name,rows in scores.items():
        if name in ("primary_correction","wrong_correction"): continue
        values=[]
        for rid,ids,row,label in zip(request_ids,item_ids,rows,labels):
            ranked=helpers.rankings(ids,row); positives={str(item) for item,value in zip(ids,label) if value>0}; values.append(ndcg_at_k(ranked,positives,10))
        metrics[name]=np.asarray(values,dtype=np.float64)
    ev=config["evaluation"]
    comparisons=compare(request_ids,metrics["primary_true"],{"base":metrics["base"],"positive_only":metrics["positive_only"],"uniform_slate":metrics["uniform_slate"],"semantic_history":metrics["semantic_history"],"primary_wrong":metrics["primary_wrong"]},samples=int(ev["bootstrap_samples"]),seed=int(ev["bootstrap_seed"]),folds=int(ev["hash_folds"]))
    direction=bootstrap(clicked_direction(scores["primary_correction"],labels),samples=int(ev["bootstrap_samples"]),seed=int(ev["bootstrap_seed"])+20)
    checks={"primary_beats_base":comparison_pass(comparisons["base"],float(ev["primary_minus_base_min"])),"primary_beats_positive_only":comparison_pass(comparisons["positive_only"],float(ev["primary_minus_positive_only_min"])),"primary_beats_uniform_slate":comparison_pass(comparisons["uniform_slate"],float(ev["primary_minus_uniform_slate_min"])),"primary_beats_semantic_history":comparison_pass(comparisons["semantic_history"],float(ev["primary_minus_semantic_history_min"])),"true_beats_wrong":comparison_pass(comparisons["primary_wrong"],float(ev["true_minus_wrong_min"])),"clicked_direction":direction["percentile_95_ci"][0]>0}
    passed=all(checks.values()); value={"schema":"myrec.c72.gate.v1","candidate_id":"c72","created_at":timestamp(),"execution_lock_sha256":lock_hash,"A0_passed":True,"labels_already_exposed":True,"requests":len(request_ids),"mean_ndcg@10":{name:float(rows.mean()) for name,rows in metrics.items()},"comparisons":comparisons,"clicked_direction":direction,"checks":checks,"passed":passed,"decision":"supports_second_domain_acquisition_only" if passed else "failed_exposed_formulation_terminal","source_A0":{"path":str(a0_path.relative_to(REPO_ROOT)),"sha256":sha256_file(a0_path)},"claim_boundary":{"fresh":False,"formulation_only":True,"c70_validated":False,"dev_test_qrels":False}}
    atomic_json(target,value); print(target.relative_to(REPO_ROOT)); print(sha256_file(target)); print(value["decision"])


if __name__=="__main__": main()
