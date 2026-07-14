"""Tests for data generators."""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import (
    IndustrialInspectionDataGen,
    SmartCityTrafficDataGen,
    AnomalyDataGenerator,
)


def test_industrial_gen_basic():
    gen = IndustrialInspectionDataGen(seed=42, samples=20, num_nodes=3)
    samples, shared = gen.generate()
    assert len(samples) > 0
    assert all(s.raw.shape[0] == 64 for s in samples)
    assert all(s.label in ["normal", "wear", "defect"] for s in samples)


def test_traffic_gen_basic():
    gen = SmartCityTrafficDataGen(seed=42, samples=20, num_nodes=3)
    samples, shared = gen.generate()
    assert all(s.raw.shape[0] == 48 for s in samples)
    assert all(s.label in ["smooth", "slow", "congested"] for s in samples)


def test_anomaly_generator_4_types():
    gen = AnomalyDataGenerator(seed=42, samples_per_type=20)
    all_anomaly = gen.generate_all()
    types = set(s.anomaly_type for s in all_anomaly)
    assert types == {"composite_defect", "em_interference", "weather_blur", "net_outage"}


def test_composite_defect_has_high_freq():
    gen = AnomalyDataGenerator(seed=42, samples_per_type=10)
    samples = gen.composite_defect()
    assert len(samples) == 10
    sig = samples[0].raw
    assert sig.std() > 0.1


def test_em_interference_has_pulse_zero():
    gen = AnomalyDataGenerator(seed=42, samples_per_type=10)
    samples = gen.em_interference()
    zero_count = sum((s.raw == 0).sum() > 0 for s in samples)
    assert zero_count >= 1


def test_net_outage_has_gap():
    gen = AnomalyDataGenerator(seed=42, samples_per_type=10)
    samples = gen.net_outage()
    for s in samples[:3]:
        assert (s.raw == 0).sum() >= 10
