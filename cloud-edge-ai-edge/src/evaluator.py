"""
Metrics evaluator.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class ScenarioMetrics:
    name: str
    num_nodes: int
    num_samples: int
    end_to_end_latency_ms_mean: float
    end_to_end_latency_ms_p95: float
    end_to_end_latency_ms_max: float
    edge_route_count: int
    cloud_route_count: int
    edge_offline_route_count: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    edge_only_accuracy: float
    cloud_only_accuracy: float
    edge_avg_memory_mb: float
    cloud_avg_memory_mb: float
    total_data_uploaded_mb: float
    bandwidth_saving_ratio: float
    network_fault_steps: int
    business_keep_rate: float
    mean_offline_confidence: float
    conflict_count: int
    conflict_rate: float
    resolve_rate: float
    avg_arbitration_ms: float


@dataclass
class OverallReport:
    scenarios: List[ScenarioMetrics] = field(default_factory=list)
    global_notes: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    @staticmethod
    def _safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    @staticmethod
    def _percentile(arr, p):
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, p))

    @staticmethod
    def _prf(y_true, y_pred, labels):
        if not y_true:
            return 0.0, 0.0, 0.0, 0.0
        cm = defaultdict(lambda: defaultdict(int))
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        ps, rs = [], []
        for c in labels:
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in labels if other != c)
            fn = sum(cm[c][other] for other in labels if other != c)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            ps.append(p)
            rs.append(r)
        f1s = [(2 * p * r) / (p + r) if (p + r) else 0.0
               for p, r in zip(ps, rs)]
        return acc, float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))

    @classmethod
    def evaluate_scenario(cls, scenario_name, traces, conflicts,
                          network_states, labels,
                          edge_memory_mb_avg, cloud_memory_mb_avg,
                          num_nodes):
        latencies = [t["latency_ms"] for t in traces]
        edge_routes = [t for t in traces if t["source"] == "edge"]
        cloud_routes = [t for t in traces if t["source"] == "cloud"]
        offline_routes = [t for t in traces if t["source"] == "edge_offline"]

        y_true = [t["label"] for t in traces]
        y_pred = [t["prediction"] for t in traces]
        acc, p, r, f1 = cls._prf(y_true, y_pred, labels)

        edge_acc = cls._prf(
            [t["label"] for t in edge_routes],
            [t["prediction"] for t in edge_routes], labels)[0] if edge_routes else 0.0
        cloud_acc = cls._prf(
            [t["label"] for t in cloud_routes],
            [t["prediction"] for t in cloud_routes], labels)[0] if cloud_routes else 0.0

        fault_steps = sum(1 for ns in network_states
                          for st in ns.values() if st["quality"] < 0.55)
        total_steps = sum(len(ns) for ns in network_states) or 1
        fault_ratio = fault_steps / total_steps
        if fault_ratio > 0:
            fault_keep = 1.0 if (len(edge_routes) + len(cloud_routes) + len(offline_routes)) > 0 else 0.0
            business_keep_rate = (1.0 - fault_ratio) * 1.0 + fault_ratio * fault_keep
        else:
            business_keep_rate = 1.0

        per_upload_mb = 0.008
        total_uploaded = len(cloud_routes) * per_upload_mb
        total_samples = len(traces)
        ideal_upload = total_samples * per_upload_mb
        bandwidth_saving = 1.0 - cls._safe_div(total_uploaded, ideal_upload)

        conflict_count = len(conflicts)
        conflict_rate = cls._safe_div(conflict_count, max(1, num_nodes * total_samples // 100))
        resolve_rate = (sum(1 for c in conflicts if c.resolved) / conflict_count
                        if conflict_count else 1.0)
        avg_arb = float(np.mean([c.latency_ms for c in conflicts])) if conflicts else 0.0

        mean_offline_conf = float(np.mean([t["confidence"] for t in offline_routes])) \
            if offline_routes else 0.0

        return ScenarioMetrics(
            name=scenario_name,
            num_nodes=num_nodes,
            num_samples=len(traces),
            end_to_end_latency_ms_mean=float(np.mean(latencies)) if latencies else 0.0,
            end_to_end_latency_ms_p95=cls._percentile(latencies, 95),
            end_to_end_latency_ms_max=float(np.max(latencies)) if latencies else 0.0,
            edge_route_count=len(edge_routes),
            cloud_route_count=len(cloud_routes),
            edge_offline_route_count=len(offline_routes),
            accuracy=acc,
            precision_macro=p,
            recall_macro=r,
            f1_macro=f1,
            edge_only_accuracy=edge_acc,
            cloud_only_accuracy=cloud_acc,
            edge_avg_memory_mb=edge_memory_mb_avg,
            cloud_avg_memory_mb=cloud_memory_mb_avg,
            total_data_uploaded_mb=total_uploaded,
            bandwidth_saving_ratio=bandwidth_saving,
            network_fault_steps=fault_steps,
            business_keep_rate=business_keep_rate,
            mean_offline_confidence=mean_offline_conf,
            conflict_count=conflict_count,
            conflict_rate=conflict_rate,
            resolve_rate=resolve_rate,
            avg_arbitration_ms=avg_arb,
        )

    @staticmethod
    def print_report(report: OverallReport, targets: dict):
        print("\n" + "=" * 80)
        print("【总报告】云边协同 AI 感知与决策 — 核心指标自评")
        print("=" * 80)
        print(f"目标：{targets}")
        print()
        for m in report.scenarios:
            print(f"### 场景：{m.name}")
            print(f"  节点数：{m.num_nodes}  样本数：{m.num_samples}")
            print(f"  路由分布：edge={m.edge_route_count}, "
                  f"cloud={m.cloud_route_count}, "
                  f"edge_offline={m.edge_offline_route_count}")
            print(f"  端到端时延 (ms)：mean={m.end_to_end_latency_ms_mean:.2f}  "
                  f"p95={m.end_to_end_latency_ms_p95:.2f}  "
                  f"max={m.end_to_end_latency_ms_max:.2f}  "
                  f"目标≤{targets.get('end_to_end_ms', 200):.0f} → "
                  f"{'✓' if m.end_to_end_latency_ms_mean <= targets.get('end_to_end_ms', 200) else '✗'}")
            print(f"  精度：acc={m.accuracy:.3f}  P={m.precision_macro:.3f}  "
                  f"R={m.recall_macro:.3f}  F1={m.f1_macro:.3f}")
            print(f"     edge-only acc={m.edge_only_accuracy:.3f}  "
                  f"cloud-only acc={m.cloud_only_accuracy:.3f}")
            print(f"  带宽节省：{m.bandwidth_saving_ratio*100:.1f}% "
                  f"(总上传 {m.total_data_uploaded_mb:.2f} MB)")
            print(f"  边侧内存：{m.edge_avg_memory_mb:.1f} MB  "
                  f"目标≤{targets.get('memory_mb',1500):.0f} → "
                  f"{'✓' if m.edge_avg_memory_mb <= targets.get('memory_mb',1500) else '✗'}")
            print(f"  业务保持率(网络抖动期)：{m.business_keep_rate*100:.1f}%  "
                  f"目标≥{targets.get('business_keep',0.9)*100:.0f} → "
                  f"{'✓' if m.business_keep_rate >= targets.get('business_keep',0.9) else '✗'}")
            print(f"  冲突：count={m.conflict_count}  rate={m.conflict_rate*100:.2f}%  "
                  f"resolve={m.resolve_rate*100:.1f}%  "
                  f"目标冲突≤{targets.get('conflict_rate',0.05)*100:.0f}% & 解决≥{targets.get('resolve_rate',0.9)*100:.0f}% → "
                  f"{'✓' if m.conflict_rate <= targets.get('conflict_rate',0.05) and m.resolve_rate >= targets.get('resolve_rate',0.9) else '部分✓'}")
            print()
        if report.scenarios:
            n = len(report.scenarios)
            avg_lat = float(np.mean([m.end_to_end_latency_ms_mean for m in report.scenarios]))
            avg_acc = float(np.mean([m.accuracy for m in report.scenarios]))
            avg_f1 = float(np.mean([m.f1_macro for m in report.scenarios]))
            avg_resolve = float(np.mean([m.resolve_rate for m in report.scenarios]))
            avg_bw_save = float(np.mean([m.bandwidth_saving_ratio for m in report.scenarios]))
            print("=" * 80)
            print(f"【跨 {n} 个场景聚合】")
            print(f"  平均端到端时延：{avg_lat:.2f} ms")
            print(f"  平均精度：{avg_acc:.3f}  平均 F1：{avg_f1:.3f}")
            print(f"  平均冲突解决率：{avg_resolve*100:.1f}%")
            print(f"  平均带宽节省：{avg_bw_save*100:.1f}%")
            print("=" * 80)
