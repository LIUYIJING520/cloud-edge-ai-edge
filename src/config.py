"""
全局配置：赛题硬指标参数化。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class EdgeModelSpec:
    name: str
    param_count_m: float
    capability_retention: float
    avg_latency_ms: float
    memory_footprint_mb: float


@dataclass
class CloudModelSpec:
    name: str
    param_count_b: float
    capability_retention: float
    avg_latency_ms: float


@dataclass
class ScenarioConfig:
    name: str
    description: str
    num_edge_nodes: int
    samples: int
    end_to_end_budget_ms: float = 200.0


@dataclass
class SystemConfig:
    ttft_reduction_target: float = 0.75
    capability_retention_min: float = 0.80
    capability_retention_max: float = 0.90
    memory_budget_mb: float = 1500.0
    business_keep_rate_target: float = 0.90
    decision_conflict_rate_target: float = 0.05
    conflict_resolve_rate_target: float = 0.90
    end_to_end_latency_target_ms: float = 200.0
    random_seed: int = 42

    edge_model: EdgeModelSpec = field(default_factory=lambda: EdgeModelSpec(
        name="EdgeLight-DistillDeepSeek-1.3B",
        param_count_m=1300.0,
        capability_retention=0.85,
        avg_latency_ms=18.0,
        memory_footprint_mb=620.0,
    ))
    cloud_model: CloudModelSpec = field(default_factory=lambda: CloudModelSpec(
        name="CloudFull-DeepSeek-V3",
        param_count_b=671.0,
        capability_retention=1.0,
        avg_latency_ms=140.0,
    ))

    scenarios: List[ScenarioConfig] = field(default_factory=lambda: [
        ScenarioConfig(
            name="industrial_inspection",
            description="工业产线质量检测（高实时、强局部）",
            num_edge_nodes=4,
            samples=800,
        ),
        ScenarioConfig(
            name="smart_city_traffic",
            description="智慧城市交通监控（广域、跨节点协同）",
            num_edge_nodes=6,
            samples=900,
        ),
    ])


def get_default_config() -> SystemConfig:
    return SystemConfig()
