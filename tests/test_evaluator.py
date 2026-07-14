"""Tests for evaluator."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import Evaluator


def test_prf_perfect():
    yt = ["a", "a", "b", "b"]
    yp = ["a", "a", "b", "b"]
    acc, p, r, f1 = Evaluator._prf(yt, yp, ["a", "b"])
    assert acc == 1.0
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_prf_zero():
    yt = ["a", "a", "b", "b"]
    yp = ["b", "b", "a", "a"]
    acc, *_ = Evaluator._prf(yt, yp, ["a", "b"])
    assert acc == 0.0


def test_prf_partial():
    yt = ["a", "a", "b", "b", "a"]
    yp = ["a", "b", "b", "b", "a"]
    acc, *_ = Evaluator._prf(yt, yp, ["a", "b"])
    assert 0 < acc < 1


def test_percentile_empty():
    assert Evaluator._percentile([], 95) == 0.0


def test_safe_div_zero():
    assert Evaluator._safe_div(1, 0) == 0.0
    assert Evaluator._safe_div(10, 2) == 5.0


def test_evaluate_scenario_end_to_end():
    traces = [
        {"label": "a", "prediction": "a", "source": "edge",
         "latency_ms": 100.0, "memory_mb": 620.0, "confidence": 0.9},
        {"label": "b", "prediction": "b", "source": "edge",
         "latency_ms": 200.0, "memory_mb": 620.0, "confidence": 0.8},
    ]
    net_states = [{"n1": {"quality": 0.85}}]
    metrics = Evaluator.evaluate_scenario(
        scenario_name="test",
        traces=traces, conflicts=[],
        network_states=net_states, labels=["a", "b"],
        edge_memory_mb_avg=620.0, cloud_memory_mb_avg=12000.0,
        num_nodes=1,
    )
    assert metrics.accuracy == 1.0
    assert metrics.end_to_end_latency_ms_mean == 150.0
    assert metrics.edge_route_count == 2
