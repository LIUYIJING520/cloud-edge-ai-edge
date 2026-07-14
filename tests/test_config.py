"""Tests for config module."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, SystemConfig


def test_default_config_loads():
    cfg = get_default_config()
    assert isinstance(cfg, SystemConfig)
    assert cfg.end_to_end_latency_target_ms == 200.0


def test_hard_targets_match_question():
    cfg = get_default_config()
    assert cfg.ttft_reduction_target == 0.75
    assert cfg.capability_retention_min == 0.80
    assert cfg.capability_retention_max == 0.90
    assert cfg.business_keep_rate_target == 0.90


def test_edge_model_spec():
    cfg = get_default_config()
    assert cfg.edge_model.param_count_m == 1300.0
    assert cfg.edge_model.memory_footprint_mb <= 1500.0


def test_scenarios_present():
    cfg = get_default_config()
    assert len(cfg.scenarios) >= 2
    for sc in cfg.scenarios:
        assert sc.samples > 0
        assert sc.num_edge_nodes > 0


def test_cloud_bigger_than_edge():
    cfg = get_default_config()
    # 671B cloud > 1.3B edge (in raw comparison unit: B units, so 671 > 1.3)
    assert cfg.cloud_model.param_count_b > cfg.edge_model.param_count_m / 1000
