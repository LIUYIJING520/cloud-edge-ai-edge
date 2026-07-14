"""Concurrent simulation: 100+ edge nodes."""
from __future__ import annotations
import multiprocessing as mp
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config
from src.edge_model import EdgeLightModel, CloudFullModel, extract_features
from src.scheduler import CloudEdgeScheduler


def run_one_node(args) -> List[Dict[str, Any]]:
    node_id, samples_subset, labels, net_q_seed = args
    cfg = get_default_config()
    edge = EdgeLightModel(cfg.edge_model, seed=cfg.random_seed + hash(node_id) % 1000)
    cloud = CloudFullModel(cfg.cloud_model, seed=cfg.random_seed + 23)
    train_feats = np.random.randn(50, 7).astype(np.float32)
    train_lbls = np.array([labels[i % 3] for i in range(50)])
    edge.fit(n_features=7, labels=labels, train_feats=train_feats, train_labels=train_lbls)
    cloud.fit(n_features=7, labels=labels, train_feats=train_feats, train_labels=train_lbls)
    scheduler = CloudEdgeScheduler(cfg, edge, cloud)
    traces = []
    rng = np.random.default_rng(net_q_seed)
    for s in samples_subset:
        net_q = float(np.clip(0.85 + rng.normal(0, 0.10), 0.05, 0.99))
        decision, result = scheduler.dispatch(s.raw, node_id, net_q, s.sample_id)
        traces.append({
            "sample_id": s.sample_id, "node_id": node_id,
            "label": s.label, "prediction": result["prediction"],
            "confidence": result["confidence"], "source": result["source"],
            "latency_ms": result["latency_ms"], "memory_mb": result["memory_mb"],
            "network_quality": net_q,
        })
    return traces


def main():
    cfg = get_default_config()
    scenario = cfg.scenarios[0]
    samples_per_node = 200
    num_nodes = 100
    num_processes = 4

    print(f"并发仿真: {num_nodes} 节点 × {samples_per_node} 样本 = {num_nodes * samples_per_node} 总任务")
    print(f"进程数: {num_processes}")

    from src.data_generator import IndustrialInspectionDataGen
    gen = IndustrialInspectionDataGen(seed=cfg.random_seed, samples=samples_per_node, num_nodes=num_nodes)
    all_samples, _ = gen.generate()

    node_args = []
    for i in range(num_nodes):
        nid = f"node_{i}"
        subset = [s for s in all_samples if s.sample_id % num_nodes == i][:samples_per_node]
        node_args.append((nid, subset, ["normal", "wear", "defect"], cfg.random_seed + i))

    t0 = time.time()
    if num_processes > 1:
        with mp.Pool(num_processes) as pool:
            all_traces_list = pool.map(run_one_node, node_args)
    else:
        all_traces_list = [run_one_node(a) for a in node_args]
    elapsed = time.time() - t0
    all_traces = [t for traces in all_traces_list for t in traces]

    print(f"\n并发仿真结果:")
    print(f"  节点数: {num_nodes}")
    print(f"  总样本: {len(all_traces)}")
    print(f"  耗时: {elapsed:.2f}s （平均 {len(all_traces)/elapsed:.0f} 样本/秒）")
    edge_count = sum(1 for t in all_traces if t["source"] == "edge")
    cloud_count = sum(1 for t in all_traces if t["source"] == "cloud")
    offline_count = sum(1 for t in all_traces if t["source"] == "edge_offline")
    print(f"  路由分布: edge={edge_count}, cloud={cloud_count}, edge_offline={offline_count}")
    correct = sum(1 for t in all_traces if t["label"] == t["prediction"])
    acc = correct / len(all_traces)
    print(f"  总体精度: {acc*100:.2f}%")

    report = {
        "concurrent_nodes": num_nodes,
        "concurrent_processes": num_processes,
        "total_samples": len(all_traces),
        "elapsed_seconds": elapsed,
        "throughput_samples_per_sec": len(all_traces) / elapsed,
        "routing_dist": {"edge": edge_count, "cloud": cloud_count, "edge_offline": offline_count},
        "accuracy": acc,
    }
    out_path = PROJECT_ROOT / "outputs" / "report_concurrent.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {out_path}")


if __name__ == "__main__":
    main()
