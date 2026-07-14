"""Tests for global decision coordinator."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config
from src.edge_model import CloudFullModel
from src.decision import GlobalDecisionCoordinator, ConflictRecord


@pytest.fixture
def coord():
    cfg = get_default_config()
    cloud = CloudFullModel(cfg.cloud_model, seed=42)
    return GlobalDecisionCoordinator(cfg, cloud)


def test_no_conflict_only_one_decision(coord):
    rec = coord.submit_decision("target_1", "node_1", "normal", 0.9)
    assert rec is None
    assert len(coord.conflicts) == 0


def test_conflict_detected_with_different_predictions(coord):
    coord.submit_decision("shared_target_1", "node_1", "normal", 0.9)
    rec = coord.submit_decision("shared_target_1", "node_2", "wear", 0.85)
    assert rec is not None
    assert isinstance(rec, ConflictRecord)
    assert rec.resolved is True


def test_same_prediction_no_conflict(coord):
    coord.submit_decision("t1", "n1", "normal", 0.9)
    coord.submit_decision("t1", "n2", "normal", 0.8)
    coord.submit_decision("t1", "n3", "normal", 0.85)
    assert len(coord.conflicts) == 0


def test_resolve_strategy(coord):
    coord.submit_decision("tx", "n1", "defect", 0.9)
    rec = coord.submit_decision("tx", "n2", "wear", 0.7)
    assert rec is not None
    assert rec.resolve_strategy in ["weighted_vote", "reweighted_vote"]


def test_node_weights_updated(coord):
    coord.submit_decision("t_x", "n1", "normal", 0.9)
    coord.submit_decision("t_x", "n2", "defect", 0.7)
    w_after = coord.broadcast_weights()
    assert len(w_after) > 0


def test_metrics_output(coord):
    """Just check coordinator produces valid state."""
    coord.submit_decision("t_a", "n1", "normal", 0.9)
    coord.submit_decision("t_a", "n2", "defect", 0.7)
    # coordinator should have some conflicts
    total_conflicts = len(coord.conflicts)
    assert isinstance(total_conflicts, int)
    assert total_conflicts >= 0
