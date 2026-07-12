from pathlib import Path
import sys,yaml

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from train.data_bridge import C75Store  # noqa:E402


def test_protocol_and_store_boundary():
    cfg=yaml.safe_load((ROOT/'configs/kuai_probe.yaml').read_text());s=C75Store(cfg,REPO);m=s.split_manifest()
    assert cfg['model']['backbone_frozen'] and cfg['model']['backbone_forced_eval']
    assert cfg['authorization']['validation_labels_after_A0_only']
    assert not cfg['authorization']['dev'] and not cfg['authorization']['test'] and not cfg['authorization']['qrels']
    assert m['train_requests']==4800 and m['validation_requests']==1200 and m['overlap']==0 and s._labels is None
