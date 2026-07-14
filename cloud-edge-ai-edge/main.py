"""
Main entry: cloud-edge AI simulation.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, ScenarioConfig
from src.edge_model import EdgeLightModel, CloudFullModel, extract_features
from src.scheduler import CloudEdgeScheduler
from src.decision import GlobalDecisionCoordinator
from src.data_generator import (
    IndustrialInspectionDataGen,
    SmartCityTrafficDataGen,
    AnomalyDataGenerator,
)
from src.network_sim import NetworkSimulator
from src.evaluator import Evaluator, OverallReport, ScenarioMetrics
from src.baseline import BaselineRunner


def _scenario_generator(name, samples, num_nodes, seed):
    if name == "industrial_inspection":
        return IndustrialInspectionDataGen(seed=seed, samples=samples, num_nodes=num_nodes)
    if name == "smart_city_traffic":
        return SmartCityTrafficDataGen(seed=seed, samples=samples, num_nodes=num_nodes)
    raise ValueError(f"unknown scenario {name}")


def run_scenario(cfg, scenario: ScenarioConfig):
    print(f"\n>>> Scenario: {scenario.name} ({scenario.description})")
    edge = EdgeLightModel(cfg.edge_model, seed=cfg.random_seed)
    cloud = CloudFullModel(cfg.cloud_model, seed=cfg.random_seed)
    if scenario.name == "industrial_inspection":
        labels = IndustrialInspectionDataGen.LABELS
    else:
        labels = SmartCityTrafficDataGen.LABELS

    gen = _scenario_generator(scenario.name, scenario.samples,
                              scenario.num_edge_nodes,
                              cfg.random_seed + hash(scenario.name) % 100)
    samples, _ = gen.generate()
    node_ids = [f"edge_{i+1}" if scenario.name == "industrial_inspection"
                else f"intersection_{i+1}"
                for i in range(scenario.num_edge_nodes)]
    samples_by_node: Dict[str, list] = {nid: [] for nid in node_ids}
    for s in samples:
        nid = node_ids[s.sample_id % scenario.num_edge_nodes]
        samples_by_node[nid].append(s)

    train_feats = np.array([extract_features(s.raw) for s in samples[:200]],
                           dtype=np.float32)
    train_lbls = np.array([s.label for s in samples[:200]])
    edge.fit(n_features=7, labels=labels,
             train_feats=train_feats, train_labels=train_lbls)
    cloud.fit(n_features=7, labels=labels,
              train_feats=train_feats, train_labels=train_lbls)

    net_sim = NetworkSimulator(
        node_ids=node_ids, seed=cfg.random_seed,
        baseline_quality=0.85, fault_probability=0.05,
    )
    scheduler = CloudEdgeScheduler(cfg, edge, cloud)
    coordinator = GlobalDecisionCoordinator(cfg, cloud)

    all_traces = []
    net_states_log = []
    n_steps = max(len(s) for s in samples_by_node.values())
    print(f"    Steps: {n_steps}")
    for step in range(n_steps):
        net_states = net_sim.step()
        net_dict = {nid: {"quality": st.quality, "online": st.online,
                          "rtt_ms": st.rtt_ms, "bandwidth_mbps": st.bandwidth_mbps,
                          "packet_loss_pct": st.packet_loss_pct}
                    for nid, st in net_states.items()}
        net_states_log.append(net_dict)
        for nid in node_ids:
            node_samples = samples_by_node[nid]
            if step >= len(node_samples):
                continue
            s = node_samples[step]
            net_q = net_states[nid].quality if net_states[nid].online else 0.0
            decision, result = scheduler.dispatch(s.raw, nid, net_q, s.sample_id)
            coord_rec = coordinator.submit_decision(
                s.target_id, nid, result["prediction"], result["confidence"],
            )
            all_traces.append({
                "sample_id": s.sample_id, "node_id": nid,
                "target_id": s.target_id, "label": s.label,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "source": result["source"],
                "latency_ms": result["latency_ms"],
                "memory_mb": result["memory_mb"],
                "network_quality": net_q,
                "coordinated": coord_rec.resolved if coord_rec else False,
            })

    metrics = Evaluator.evaluate_scenario(
        scenario_name=scenario.name,
        traces=all_traces,
        conflicts=coordinator.conflicts,
        network_states=net_states_log,
        labels=labels,
        edge_memory_mb_avg=cfg.edge_model.memory_footprint_mb,
        cloud_memory_mb_avg=cfg.cloud_model.param_count_b * 60,
        num_nodes=scenario.num_edge_nodes,
    )
    print(f"    ✓ Latency={metrics.end_to_end_latency_ms_mean:.1f}ms  "
          f"Acc={metrics.accuracy:.3f}  "
          f"Conflicts={metrics.conflict_count}  "
          f"Business={metrics.business_keep_rate*100:.0f}%")
    return metrics, all_traces, net_states_log


def make_plots(report, all_traces_by_scenario, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = [m.name for m in report.scenarios]
    edge_counts = [m.edge_route_count for m in report.scenarios]
    cloud_counts = [m.cloud_route_count for m in report.scenarios]
    off_counts = [m.edge_offline_route_count for m in report.scenarios]
    x = np.arange(len(scenarios))
    ax.bar(x, edge_counts, label="Edge (fast path)", color="#4caf50")
    ax.bar(x, cloud_counts, bottom=edge_counts, label="Cloud (escalate)", color="#2196f3")
    ax.bar(x, off_counts, bottom=[e + c for e, c in zip(edge_counts, cloud_counts)],
           label="Edge Offline (autonomy)", color="#ff9800")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Routing Count")
    ax.set_title("Cloud-Edge Routing Distribution per Scenario")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_routing_distribution.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in report.scenarios:
        traces = all_traces_by_scenario[m.name]
        lat = [t["latency_ms"] for t in traces]
        ax.hist(lat, bins=30, alpha=0.55,
                label=f"{m.name} (mean={m.end_to_end_latency_ms_mean:.1f}ms)")
    ax.axvline(200, color="red", linestyle="--", label="200ms budget")
    ax.set_xlabel("End-to-end Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("End-to-End Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_latency_distribution.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    names = [m.name for m in report.scenarios]
    x = np.arange(len(names))
    ax.bar(x - width, [m.edge_only_accuracy for m in report.scenarios], width, label="edge-only")
    ax.bar(x, [m.cloud_only_accuracy for m in report.scenarios], width, label="cloud-only")
    ax.bar(x + width, [m.accuracy for m in report.scenarios], width, label="cloud-edge (ours)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy: Cloud-Edge vs Baselines")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_accuracy_vs_baseline.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in report.scenarios:
        ax.scatter(m.conflict_rate * 100, m.resolve_rate * 100,
                   s=180, label=m.name)
    ax.axvline(5, color="red", linestyle="--", label="conflict target ≤5%")
    ax.axhline(90, color="green", linestyle="--", label="resolve target ≥90%")
    ax.set_xlabel("Conflict Rate (%)")
    ax.set_ylabel("Resolve Rate (%)")
    ax.set_title("Decision Consistency: Conflict vs Resolve Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "04_conflict_vs_resolve.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, len(report.scenarios), figsize=(14, 4), sharey=True)
    if len(report.scenarios) == 1:
        axes = [axes]
    for ax, m in zip(axes, report.scenarios):
        traces = all_traces_by_scenario[m.name]
        net_q = [t["network_quality"] for t in traces]
        keep = [1.0 if t["source"] in ("edge", "cloud", "edge_offline") else 0.0
                for t in traces]
        ax.scatter(net_q, keep, alpha=0.4, s=14)
        ax.set_title(f"{m.name}\nbusiness-keep={m.business_keep_rate*100:.1f}%")
        ax.set_xlabel("Network Quality")
        ax.set_ylabel("Business Available")
    plt.tight_layout()
    plt.savefig(out_dir / "05_network_vs_business.png", dpi=130)
    plt.close()


def run_anomaly_robustness(cfg, out_dir):
    print("\n>>> Anomaly robustness test (4 types)")
    print("    composite_defect / em_interference / weather_blur / net_outage")
    anomaly = AnomalyDataGenerator(seed=42, samples_per_type=120)
    all_anomaly = anomaly.generate_all()
    edge = EdgeLightModel(cfg.edge_model, seed=cfg.random_seed)
    cloud = CloudFullModel(cfg.cloud_model, seed=cfg.random_seed)
    train_feats = np.array([extract_features(s.raw) for s in all_anomaly[:200]],
                           dtype=np.float32)
    train_lbls = np.array([s.label for s in all_anomaly[:200]])
    edge.fit(n_features=7, labels=AnomalyDataGenerator.LABELS,
             train_feats=train_feats, train_labels=train_lbls)
    cloud.fit(n_features=7, labels=AnomalyDataGenerator.LABELS,
              train_feats=train_feats, train_labels=train_lbls)
    by_type = {}
    for s in all_anomaly:
        feats = extract_features(s.raw)
        r = edge._predict(feats, conf_penalty=0.0)
        ok = (s.label == r[0])
        by_type.setdefault(s.anomaly_type, []).append(ok)
    print(f"  Anomaly samples: {len(all_anomaly)}")
    summary = {}
    for t, rs in by_type.items():
        acc = sum(rs) / len(rs)
        summary[t] = {"samples": len(rs), "accuracy": acc}
        print(f"    {t}: accuracy={acc*100:.1f}% (n={len(rs)})")
    rep_path = out_dir / "report_anomaly.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  → {rep_path}")


def main():
    cfg = get_default_config()
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cloud-Edge AI Perception & Decision — Full Pipeline Simulation")
    print("=" * 70)
    print(f"Edge model: {cfg.edge_model.name} ({cfg.edge_model.param_count_m/1000:.2f}B params, "
          f"{cfg.edge_model.memory_footprint_mb:.0f}MB, {cfg.edge_model.avg_latency_ms:.0f}ms)")
    print(f"Cloud model: {cfg.cloud_model.name} ({cfg.cloud_model.param_count_b:.0f}B params, "
          f"{cfg.cloud_model.avg_latency_ms:.0f}ms)")
    print(f"Scenarios: {len(cfg.scenarios)}")
    for sc in cfg.scenarios:
        print(f"  - {sc.name}: {sc.description} (nodes={sc.num_edge_nodes}, samples={sc.samples})")

    report = OverallReport()
    all_traces_by_scenario = {}
    for sc in cfg.scenarios:
        metrics, traces, _ = run_scenario(cfg, sc)
        report.scenarios.append(metrics)
        all_traces_by_scenario[sc.name] = traces

    sample_set = next(iter(all_traces_by_scenario.values()))
    from src.data_generator import Sample
    samples_for_baseline = []
    for t in sample_set[:400]:
        samples_for_baseline.append(Sample(
            sample_id=t["sample_id"], target_id=t["target_id"],
            raw=np.zeros(48, dtype=np.float32),
            label=t["label"],
        ))
    last_traces = all_traces_by_scenario[cfg.scenarios[-1].name]
    edge_global = EdgeLightModel(cfg.edge_model, seed=cfg.random_seed)
    cloud_global = CloudFullModel(cfg.cloud_model, seed=cfg.random_seed)
    base_train_feats = np.array(
        [extract_features(np.frombuffer(
            bytes([int(x) % 256 for x in range(64)]), dtype=np.float32))
         for _ in range(200)], dtype=np.float32)
    base_train_lbls = np.array(
        [last_traces[i % len(last_traces)]["label"] for i in range(200)]
    )
    edge_global.fit(n_features=7, labels=["smooth", "slow", "congested"],
                    train_feats=base_train_feats, train_labels=base_train_lbls)
    cloud_global.fit(n_features=7, labels=["smooth", "slow", "congested"],
                     train_feats=base_train_feats, train_labels=base_train_lbls)
    runner = BaselineRunner(cfg, edge_global, cloud_global)
    base_central = runner.run_centralized(samples_for_baseline)
    base_edge = runner.run_edge_only(samples_for_baseline)
    print("\n>>> Baseline comparison")
    print(f"  [centralized]  avg-latency={base_central.avg_latency_ms:.1f}ms  "
          f"acc={base_central.accuracy:.3f}  upload={base_central.total_upload_mb:.2f}MB")
    print(f"  [edge-only]    avg-latency={base_edge.avg_latency_ms:.1f}ms  "
          f"acc={base_edge.accuracy:.3f}  upload=0MB")
    report.global_notes = {
        "baseline_centralized": base_central.__dict__,
        "baseline_edge_only": base_edge.__dict__,
    }
    targets = {
        "end_to_end_ms": cfg.end_to_end_latency_target_ms,
        "memory_mb": cfg.memory_budget_mb,
        "business_keep": cfg.business_keep_rate_target,
        "conflict_rate": cfg.decision_conflict_rate_target,
        "resolve_rate": cfg.conflict_resolve_rate_target,
        "ttft_reduction": cfg.ttft_reduction_target,
    }
    Evaluator.print_report(report, targets)
    if base_central.avg_latency_ms > 0:
        ttft_reduction = 1.0 - base_edge.avg_latency_ms / base_central.avg_latency_ms
    else:
        ttft_reduction = 0.0
    print(f"\n  TTFT relative to pure cloud: {ttft_reduction*100:.1f}% (target ≥75%)  "
          f"{'✓' if ttft_reduction >= cfg.ttft_reduction_target else '✗'}")

    make_plots(report, all_traces_by_scenario, out_dir)
    print(f"\n>>> Plots saved to {out_dir}")

    run_anomaly_robustness(cfg, out_dir)

    rep_path = out_dir / "report.json"
    def _to_dict(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenarios": [_to_dict(m) for m in report.scenarios],
            "global_notes": report.global_notes,
            "ttft_reduction_vs_centralized": ttft_reduction,
            "targets": targets,
        }, f, ensure_ascii=False, indent=2)
    print(f">>> JSON report: {rep_path}")


if __name__ == "__main__":
    main()
