"""PEMS-BAY (traffic) + Tetouan (power) real data loaders."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd


@dataclass
class TrafficSample:
    sample_id: int
    target_id: str
    raw: np.ndarray
    label: str


@dataclass
class PowerSample:
    sample_id: int
    target_id: str
    raw: np.ndarray
    label: str


class PEMSBayLoader:
    LABELS = ["smooth", "slow", "congested"]

    def __init__(self, h5_path: str = None, samples: int = 800,
                 num_nodes: int = 6, window: int = 48, seed: int = 0):
        if h5_path is None:
            h5_path = str(Path(__file__).resolve().parent.parent
                           / "data" / "datasets" / "pems_bay" / "PEMS-BAY.h5")
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"PEMS-BAY not found: {self.h5_path}")
        self.samples = samples
        self.num_nodes = num_nodes
        self.window = window
        self.rng = np.random.default_rng(seed)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            arr = f["speed/block0_values"][:]
        sensor_ids = list(range(min(self.num_nodes * 3, arr.shape[1])))
        df = arr[:, sensor_ids].astype(np.float32)
        n_segments = (len(df) - self.window) // self.window
        all_segs = []
        all_labels = []
        seg_means = np.array([df[i * self.window: (i + 1) * self.window].mean()
                              for i in range(n_segments)])
        p33 = np.percentile(seg_means, 33)
        p66 = np.percentile(seg_means, 66)
        for i in range(n_segments):
            seg = df[i * self.window: (i + 1) * self.window].mean(axis=1)
            if len(seg) == self.window:
                all_segs.append(seg)
                m = seg.mean()
                if m < p33:
                    lbl = "smooth"
                elif m < p66:
                    lbl = "slow"
                else:
                    lbl = "congested"
                all_labels.append(lbl)
        all_segs = np.array(all_segs, dtype=np.float32)
        all_labels = np.array(all_labels)
        if self.samples and self.samples < len(all_segs):
            idx = self.rng.choice(len(all_segs), self.samples, replace=False)
            all_segs = all_segs[idx]
            all_labels = all_labels[idx]
        return all_segs, all_labels

    def generate(self) -> Tuple[list, list]:
        segs, labels = self.load()
        per_node = []
        for i, (seg, lbl) in enumerate(zip(segs, labels)):
            if i % 30 == 0 and i >= self.num_nodes:
                target_id = f"vehicle_{i // 30}"
            else:
                target_id = f"intersection_{(i % self.num_nodes) + 1}_flow_{i}"
            per_node.append(TrafficSample(
                sample_id=i, target_id=target_id, raw=seg, label=str(lbl),
            ))
        shared = [s for s in per_node if s.target_id.startswith("vehicle_")]
        expanded = list(per_node)
        for s in shared:
            for offset in range(1, 2):
                expanded.append(TrafficSample(
                    sample_id=s.sample_id + 50_000 * offset,
                    target_id=s.target_id, raw=s.raw.copy(), label=s.label,
                ))
        return expanded, shared


class PowerConsumptionLoader:
    LABELS = ["normal", "warning", "fault"]

    def __init__(self, csv_path: str = None, samples: int = 600,
                 num_nodes: int = 4, window: int = 48, seed: int = 0):
        if csv_path is None:
            csv_path = str(Path(__file__).resolve().parent.parent
                           / "data" / "datasets" / "power_consumption" / "train.csv")
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"power_consumption not found: {self.csv_path}")
        self.samples = samples
        self.num_nodes = num_nodes
        self.window = window
        self.rng = np.random.default_rng(seed)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.csv_path)
        power_cols = [c for c in df.columns if "Power Consumption" in c or "power" in c.lower()]
        if not power_cols:
            raise ValueError(f"no power column found: {df.columns.tolist()}")
        series = df[power_cols[0]].dropna().values.astype(np.float32)
        n_segments = (len(series) - self.window) // self.window
        all_segs = []
        all_labels = []
        seg_stats = []
        for i in range(n_segments):
            seg = series[i * self.window: (i + 1) * self.window]
            if len(seg) == self.window:
                cv = float(seg.std() / (seg.mean() + 1e-6))
                seg_stats.append((seg, cv))
        if not seg_stats:
            return np.array([]), np.array([])
        cvs = np.array([s[1] for s in seg_stats])
        p33 = np.percentile(cvs, 33)
        p66 = np.percentile(cvs, 66)
        for seg, cv in seg_stats:
            all_segs.append(seg)
            if cv < p33:
                lbl = "normal"
            elif cv < p66:
                lbl = "warning"
            else:
                lbl = "fault"
            all_labels.append(lbl)
        all_segs = np.array(all_segs, dtype=np.float32)
        all_labels = np.array(all_labels)
        if self.samples and self.samples < len(all_segs):
            idx = self.rng.choice(len(all_segs), self.samples, replace=False)
            all_segs = all_segs[idx]
            all_labels = all_labels[idx]
        return all_segs, all_labels

    def generate(self) -> Tuple[list, list]:
        segs, labels = self.load()
        per_node = []
        for i, (seg, lbl) in enumerate(zip(segs, labels)):
            if i % 20 == 0 and i >= self.num_nodes:
                target_id = f"shared_grid_{i // 20}"
            else:
                target_id = f"station_{(i % self.num_nodes) + 1}_read_{i}"
            per_node.append(PowerSample(
                sample_id=i, target_id=target_id, raw=seg, label=str(lbl),
            ))
        shared = [s for s in per_node if s.target_id.startswith("shared_grid_")]
        expanded = list(per_node)
        for s in shared:
            for offset in range(1, 2):
                expanded.append(PowerSample(
                    sample_id=s.sample_id + 30_000 * offset,
                    target_id=s.target_id, raw=s.raw.copy(), label=s.label,
                ))
        return expanded, shared
