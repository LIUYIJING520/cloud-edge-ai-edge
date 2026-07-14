"""Tests for edge/cloud models."""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_model import (
    EdgeLightModel, CloudFullModel, extract_features,
)
from src.config import get_default_config


def test_extract_features_shape():
    raw = np.random.randn(256).astype(np.float32)
    feats = extract_features(raw)
    assert feats.shape == (7,)


def test_extract_features_deterministic():
    a = np.random.randn(128)
    assert (extract_features(a) == extract_features(a)).all()


def test_edge_inference():
    cfg = get_default_config()
    edge = EdgeLightModel(cfg.edge_model, seed=42)
    tf = np.random.randn(50, 7).astype(np.float32)
    tl = np.array(["normal"] * 17 + ["wear"] * 17 + ["defect"] * 16)
    edge.fit(n_features=7, labels=["normal", "wear", "defect"],
             train_feats=tf, train_labels=tl)
    raw = np.random.randn(64).astype(np.float32)
    r = edge.infer(0, raw, "edge_1")
    assert r.prediction in ["normal", "wear", "defect"]
    assert 0.0 <= r.confidence <= 1.0
    assert r.memory_mb <= cfg.edge_model.memory_footprint_mb + 1.0


def test_edge_offline():
    cfg = get_default_config()
    edge = EdgeLightModel(cfg.edge_model, seed=42)
    tf = np.random.randn(50, 7).astype(np.float32)
    tl = np.array(["normal"] * 17 + ["wear"] * 17 + ["defect"] * 16)
    edge.fit(n_features=7, labels=["normal", "wear", "defect"],
             train_feats=tf, train_labels=tl)
    raw = np.random.randn(64).astype(np.float32)
    r = edge.infer(0, raw, "edge_1", offline=True)
    assert r.offline is True
    assert r.confidence >= 0.5


def test_cloud_inference():
    cfg = get_default_config()
    cloud = CloudFullModel(cfg.cloud_model, seed=42)
    tf = np.random.randn(50, 7).astype(np.float32)
    tl = np.array(["normal"] * 17 + ["wear"] * 17 + ["defect"] * 16)
    cloud.fit(n_features=7, labels=["normal", "wear", "defect"],
              train_feats=tf, train_labels=tl)
    raw = np.random.randn(64).astype(np.float32)
    r = cloud.infer(0, raw)
    assert "prediction" in r
    assert "latency_ms" in r
