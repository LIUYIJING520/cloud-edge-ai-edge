"""
Edge/cloud models (KNN surrogate).
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import EdgeModelSpec


@dataclass
class EdgePerceptionResult:
    node_id: str
    sample_id: int
    features: np.ndarray
    prediction: Any
    confidence: float
    latency_ms: float
    memory_mb: float
    offline: bool = False


def extract_features(raw: np.ndarray) -> np.ndarray:
    mean = float(raw.mean())
    std = float(raw.std() + 1e-9)
    skew = float(((raw - mean) ** 3).mean() / (std ** 3 + 1e-9))
    kurt = float(((raw - mean) ** 4).mean() / (std ** 4 + 1e-9) - 3)
    energy = float((raw ** 2).mean())
    peak = float(np.abs(raw).max())
    if raw.size > 4:
        diff = np.diff(raw)
        hf_energy = float((diff ** 2).mean()) / (energy + 1e-9)
    else:
        hf_energy = 0.0
    return np.array([mean, std, skew, kurt, energy, peak, hf_energy], dtype=np.float32)


class EdgeLightModel:
    """Edge-side KNN surrogate for simulation."""

    def __init__(self, spec: EdgeModelSpec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self._labels = None
        self._train_feats = None
        self._train_labels = None
        self.capability = spec.capability_retention

    def fit(self, n_features: int, labels: list,
            train_feats: Optional[np.ndarray] = None,
            train_labels: Optional[np.ndarray] = None):
        self._n_features = n_features
        self._labels = labels
        if train_feats is not None and train_labels is not None:
            self._train_feats = train_feats.astype(np.float32)
            self._train_labels = np.array(train_labels)
        else:
            self._train_feats = None
            self._train_labels = None

    def _predict(self, feats: np.ndarray, conf_penalty: float = 0.0) -> Tuple[Any, float]:
        assert self._labels is not None and self._train_feats is not None
        diffs = self._train_feats - feats
        dists = np.linalg.norm(diffs, axis=1)
        k = min(7, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        vote = {}
        for i in idx:
            lbl = self._train_labels[i]
            vote[lbl] = vote.get(lbl, 0.0) + 1.0 / (dists[i] + 1e-3)
        scores = np.array([vote.get(l, 0.0) for l in self._labels], dtype=np.float64)
        s = scores.sum() + 1e-9
        scores = scores / s
        scores = self.capability * scores + (1 - self.capability) / len(self._labels)
        scores = scores / (scores.sum() + 1e-9)
        idx_max = int(np.argmax(scores))
        out_conf = float(np.clip(scores[idx_max] - conf_penalty, 0.05, 1.0))
        return self._labels[idx_max], out_conf

    def infer(self, sample_id: int, raw: np.ndarray, node_id: str,
              offline: bool = False) -> EdgePerceptionResult:
        t0 = time.perf_counter()
        feats = extract_features(raw)
        if offline:
            anomaly = float(np.linalg.norm(feats) > 2.0)
            pred = "defect" if anomaly > 0.5 else "normal"
            conf = float(np.clip(0.55 + self.rng.normal(0, 0.05), 0.5, 0.78))
        else:
            difficulty = float(np.clip(np.linalg.norm(feats) / 3.0, 0, 1))
            if difficulty > 0.55:
                pred, conf = self._predict(feats, conf_penalty=0.35)
            else:
                pred, conf = self._predict(feats, conf_penalty=0.0)
        base = self.spec.avg_latency_ms
        jitter = float(self.rng.normal(0, 1.5))
        latency = max(2.0, base + jitter + float(self.rng.uniform(0, 2)))
        mem = self.spec.memory_footprint_mb + feats.nbytes / (1024 * 1024)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency = max(latency, elapsed_ms)
        return EdgePerceptionResult(
            node_id=node_id, sample_id=sample_id, features=feats,
            prediction=pred, confidence=conf, latency_ms=latency,
            memory_mb=mem, offline=offline,
        )


class CloudFullModel:
    """Cloud-side full model (K=15 KNN vote)."""

    def __init__(self, spec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self._labels = None
        self._train_feats = None
        self._train_labels = None

    def fit(self, n_features: int, labels: list,
            train_feats: Optional[np.ndarray] = None,
            train_labels: Optional[np.ndarray] = None):
        self._n_features = n_features
        self._labels = labels
        if train_feats is not None and train_labels is not None:
            self._train_feats = train_feats.astype(np.float32)
            self._train_labels = np.array(train_labels)
        else:
            self._train_feats = None
            self._train_labels = None

    def infer(self, sample_id: int, raw: np.ndarray,
              edge_hint: Optional[Dict] = None) -> Dict:
        assert self._labels is not None
        t0 = time.perf_counter()
        if edge_hint is not None:
            feats = edge_hint["features"]
        else:
            feats = extract_features(raw)
            if self._n_features and feats.size < self._n_features:
                feats = np.concatenate([feats, np.zeros(self._n_features - feats.size)])
            feats = feats[: self._n_features]
        if self._train_feats is not None and len(self._train_feats) > 0:
            diffs = self._train_feats - feats
            dists = np.linalg.norm(diffs, axis=1)
            k = min(15, len(dists))
            idx = np.argpartition(dists, k - 1)[:k]
            vote = {}
            for i in idx:
                lbl = self._train_labels[i]
                vote[lbl] = vote.get(lbl, 0) + 1.0 / (dists[i] + 1e-3)
            scores = np.array([vote.get(l, 0.0) for l in self._labels], dtype=np.float64)
            scores = scores / (scores.sum() + 1e-9)
        else:
            scores = np.tanh(feats[:len(self._labels)] @ np.ones(len(self._labels)) * 0.1)
            scores = scores - scores.max()
            exps = np.exp(scores).astype(np.float64)
            scores = (exps / (exps.sum() + 1e-9)).reshape(-1)
        idx_max = int(np.argmax(scores))
        conf = float(scores[idx_max])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency = max(self.spec.avg_latency_ms + float(self.rng.normal(0, 5)), elapsed_ms)
        return {
            "prediction": self._labels[idx_max],
            "confidence": conf,
            "latency_ms": latency,
            "features": feats,
        }
