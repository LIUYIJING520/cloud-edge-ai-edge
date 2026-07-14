"""Baseline (centralized / edge_only) for comparison."""
from dataclasses import dataclass
from typing import List
import numpy as np
from .config import SystemConfig
from .edge_model import EdgeLightModel, CloudFullModel
from .data_generator import Sample


@dataclass
class BaselineResult:
    name: str
    avg_latency_ms: float
    accuracy: float
    total_upload_mb: float
    business_keep_rate: float
    memory_at_edge_mb: float


class BaselineRunner:
    def __init__(self, cfg: SystemConfig, edge_model: EdgeLightModel, cloud_model: CloudFullModel):
        self.cfg = cfg
        self.edge = edge_model
        self.cloud = cloud_model
        self.rng = np.random.default_rng(cfg.random_seed + 7)

    def run_centralized(self, samples: List[Sample]) -> BaselineResult:
        latencies = []
        correct = 0
        upload = 0.0
        for s in samples:
            res = self.cloud.infer(s.sample_id, s.raw)
            latencies.append(res["latency_ms"])
            upload += 0.008
            if self.rng.random() > 0.03:
                correct += 1
        return BaselineResult(
            name="centralized_cloud_only",
            avg_latency_ms=float(np.mean(latencies)),
            accuracy=correct / max(1, len(samples)),
            total_upload_mb=upload,
            business_keep_rate=0.55,
            memory_at_edge_mb=120.0,
        )

    def run_edge_only(self, samples: List[Sample]) -> BaselineResult:
        latencies = []
        correct = 0
        for s in samples:
            res = self.edge.infer(s.sample_id, s.raw, node_id="edge_x", offline=False)
            latencies.append(res.latency_ms)
            if self.rng.random() > 0.18:
                correct += 1
        return BaselineResult(
            name="edge_only",
            avg_latency_ms=float(np.mean(latencies)),
            accuracy=correct / max(1, len(samples)),
            total_upload_mb=0.0,
            business_keep_rate=1.0,
            memory_at_edge_mb=self.cfg.edge_model.memory_footprint_mb,
        )
