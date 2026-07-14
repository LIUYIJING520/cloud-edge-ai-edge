"""
Cloud-edge scheduler with 3-level routing.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .edge_model import EdgeLightModel, CloudFullModel, EdgePerceptionResult


@dataclass
class RoutingDecision:
    sample_id: int
    node_id: str
    route: str
    edge_confidence: float
    network_quality: float
    estimated_latency_ms: float
    reason: str


class CloudEdgeScheduler:
    def __init__(self, cfg: SystemConfig, edge: EdgeLightModel, cloud: CloudFullModel):
        self.cfg = cfg
        self.edge = edge
        self.cloud = cloud
        self.rng = np.random.default_rng(cfg.random_seed + 17)
        self.conf_threshold = 0.65
        self.net_threshold = 0.55

    def decide(self, sample_id, node_id, edge_result, network_quality):
        conf = edge_result.confidence
        if network_quality < self.net_threshold:
            return RoutingDecision(
                sample_id=sample_id, node_id=node_id, route="edge_offline",
                edge_confidence=conf, network_quality=network_quality,
                estimated_latency_ms=edge_result.latency_ms,
                reason=f"network_quality={network_quality:.2f} < {self.net_threshold}, edge autonomy",
            )
        if conf >= self.conf_threshold:
            return RoutingDecision(
                sample_id=sample_id, node_id=node_id, route="edge",
                edge_confidence=conf, network_quality=network_quality,
                estimated_latency_ms=edge_result.latency_ms,
                reason=f"edge_conf={conf:.2f} >= {self.conf_threshold}, fast path",
            )
        return RoutingDecision(
            sample_id=sample_id, node_id=node_id, route="cloud",
            edge_confidence=conf, network_quality=network_quality,
            estimated_latency_ms=self.cfg.cloud_model.avg_latency_ms,
            reason=f"edge_conf={conf:.2f} < {self.conf_threshold}, escalate to cloud",
        )

    def dispatch(self, raw, node_id, network_quality, sample_id):
        edge_res = self.edge.infer(sample_id, raw, node_id, offline=False)
        decision = self.decide(sample_id, node_id, edge_res, network_quality)
        if decision.route == "edge":
            return decision, {
                "prediction": edge_res.prediction,
                "confidence": edge_res.confidence,
                "latency_ms": edge_res.latency_ms,
                "memory_mb": edge_res.memory_mb,
                "source": "edge",
                "edge_features": edge_res.features,
            }
        if decision.route == "edge_offline":
            offline_res = self.edge.infer(sample_id, raw, node_id, offline=True)
            return decision, {
                "prediction": offline_res.prediction,
                "confidence": offline_res.confidence,
                "latency_ms": offline_res.latency_ms,
                "memory_mb": offline_res.memory_mb,
                "source": "edge_offline",
                "edge_features": offline_res.features,
            }
        net_rtt_ms = 80.0 * (1.0 - network_quality) + 20.0
        cloud_res = self.cloud.infer(sample_id, raw, edge_hint={"features": edge_res.features})
        return decision, {
            "prediction": cloud_res["prediction"],
            "confidence": cloud_res["confidence"],
            "latency_ms": edge_res.latency_ms + net_rtt_ms + cloud_res["latency_ms"],
            "memory_mb": edge_res.memory_mb + self.cfg.cloud_model.param_count_b * 60,
            "source": "cloud",
            "edge_features": edge_res.features,
        }
