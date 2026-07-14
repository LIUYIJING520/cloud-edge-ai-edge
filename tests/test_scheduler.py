"""Tests for cloud-edge scheduler."""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config
from src.edge_model import EdgeLightModel, CloudFullModel
from src.scheduler import CloudEdgeScheduler, RoutingDecision
from src.edge_model import EdgePerceptionResult


@pytest.fixture
def sched():
    cfg = get_default_config()
    edge = EdgeLightModel(cfg.edge_model, seed=42)
    cloud = CloudFullModel(cfg.cloud_model, seed=42)
    tf = np.random.randn(50, 7).astype(np.float32)
    tl = np.array(["normal"] * 17 + ["wear"] * 17 + ["defect"] * 16)
    edge.fit(n_features=7, labels=["normal", "wear", "defect"],
             train_feats=tf, train_labels=tl)
    cloud.fit(n_features=7, labels=["normal", "wear", "defect"],
              train_feats=tf, train_labels=tl)
    return CloudEdgeScheduler(cfg, edge, cloud)


def _res(conf=0.9):
    return EdgePerceptionResult(
        node_id="n1", sample_id=0,
        features=np.zeros(7, dtype=np.float32),
        prediction="wear", confidence=conf,
        latency_ms=18.0, memory_mb=620.0,
    )


def test_high_confidence_edge(sched):
    d = sched.decide(0, "n1", _res(0.95), network_quality=0.85)
    assert d.route == "edge"


def test_low_confidence_escalates(sched):
    d = sched.decide(0, "n1", _res(0.40), network_quality=0.85)
    assert d.route == "cloud"


def test_low_network_offline(sched):
    d = sched.decide(0, "n1", _res(0.95), network_quality=0.10)
    assert d.route == "edge_offline"


def test_dispatch(sched):
    raw = np.random.randn(64).astype(np.float32)
    d, r = sched.dispatch(raw, "n1", network_quality=0.85, sample_id=0)
    assert isinstance(d, RoutingDecision)
    assert "prediction" in r
    assert r["source"] in ["edge", "cloud", "edge_offline"]


def test_dispatch_offline(sched):
    raw = np.random.randn(64).astype(np.float32)
    _, r = sched.dispatch(raw, "n1", network_quality=0.05, sample_id=0)
    assert r["source"] == "edge_offline"
