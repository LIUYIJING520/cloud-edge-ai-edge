"""Tests for network simulator."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.network_sim import NetworkSimulator, NetworkCondition


def test_step_returns_state_per_node():
    nodes = ["n1", "n2", "n3"]
    sim = NetworkSimulator(nodes, seed=42, baseline_quality=0.85)
    states = sim.step()
    assert set(states.keys()) == set(nodes)
    for nid, st in states.items():
        assert isinstance(st, NetworkCondition)
        assert 0.0 <= st.quality <= 1.0


def test_with_fault_simulation():
    nodes = ["n1"]
    sim = NetworkSimulator(nodes, seed=1, baseline_quality=0.85, fault_probability=0.5)
    fault_seen = False
    for _ in range(50):
        states = sim.step()
        if states["n1"].quality < 0.55:
            fault_seen = True
            break
    assert fault_seen


def test_bandwidth_scales_with_quality():
    nodes = ["n1"]
    sim = NetworkSimulator(nodes, seed=42)
    states = sim.step()
    expected_bw = 100.0 * states["n1"].quality
    assert abs(states["n1"].bandwidth_mbps - expected_bw) < 1e-6


def test_packet_loss_during_faults():
    nodes = ["n1"]
    sim = NetworkSimulator(nodes, seed=1, baseline_quality=0.85, fault_probability=0.5)
    max_loss = 0
    for _ in range(100):
        s = sim.step()["n1"]
        if s.quality < 0.55:
            max_loss = max(max_loss, s.packet_loss_pct)
    assert max_loss > 0
