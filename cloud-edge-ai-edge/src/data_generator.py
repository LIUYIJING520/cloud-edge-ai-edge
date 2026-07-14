"""
仿真数据生成器 + 4 大异常（复合缺陷 / 电磁干扰 / 雨雪模糊 / 瞬时断网）。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Sample:
    sample_id: int
    target_id: str
    raw: np.ndarray
    label: str
    anomaly_type: str = "none"


class IndustrialInspectionDataGen:
    LABELS = ["normal", "wear", "defect"]

    def __init__(self, seed: int = 0, samples: int = 800, num_nodes: int = 4):
        self.rng = np.random.default_rng(seed)
        self.samples = samples
        self.num_nodes = num_nodes

    def generate(self) -> Tuple[list, list]:
        per_node = []
        shared = []
        for node_idx in range(self.num_nodes):
            node_id = f"edge_{node_idx+1}"
            base_freq = 5.0 + node_idx * 0.7
            for s in range(self.samples):
                cls = self.rng.choice(self.LABELS, p=[0.7, 0.2, 0.1])
                t = np.linspace(0, 1, 64)
                if cls == "normal":
                    sig = 0.5 * np.sin(2 * np.pi * base_freq * t) + self.rng.normal(0, 0.05, size=t.shape)
                elif cls == "wear":
                    sig = 0.6 * np.sin(2 * np.pi * base_freq * t) + \
                          0.15 * np.sin(2 * np.pi * base_freq * 3 * t) + \
                          self.rng.normal(0, 0.10, size=t.shape)
                else:
                    sig = 0.9 * np.sin(2 * np.pi * base_freq * t + 0.5) + \
                          0.4 * np.sin(2 * np.pi * base_freq * 5 * t) + \
                          self.rng.normal(0, 0.20, size=t.shape)
                if s % 20 == 0 and node_idx > 0:
                    target_id = f"shared_dev_{s // 20}"
                else:
                    target_id = f"{node_id}_dev_{s}"
                per_node.append(Sample(
                    sample_id=len(per_node), target_id=target_id,
                    raw=sig.astype(np.float32), label=cls,
                ))
        shared = [s for s in per_node if s.target_id.startswith("shared_dev_")]
        return per_node, shared


class SmartCityTrafficDataGen:
    LABELS = ["smooth", "slow", "congested"]

    def __init__(self, seed: int = 0, samples: int = 900, num_nodes: int = 6):
        self.rng = np.random.default_rng(seed + 100)
        self.samples = samples
        self.num_nodes = num_nodes

    def generate(self) -> Tuple[list, list]:
        per_node = []
        shared = []
        for node_idx in range(self.num_nodes):
            node_id = f"intersection_{node_idx+1}"
            for s in range(self.samples):
                cls = self.rng.choice(self.LABELS, p=[0.55, 0.30, 0.15])
                t = np.linspace(0, 1, 48)
                if cls == "smooth":
                    sig = 5 + 2 * np.sin(2 * np.pi * 0.5 * t) + self.rng.normal(0, 0.8, size=t.shape)
                elif cls == "slow":
                    sig = 12 + 4 * np.sin(2 * np.pi * 1.0 * t + 1) + self.rng.normal(0, 1.5, size=t.shape)
                else:
                    sig = 25 + 5 * np.sin(2 * np.pi * 1.5 * t + 2) + self.rng.normal(0, 2.5, size=t.shape)
                if s % 5 == 0 and node_idx < self.num_nodes - 1:
                    target_id = f"vehicle_{s // 5}"
                else:
                    target_id = f"{node_id}_flow_{s}"
                per_node.append(Sample(
                    sample_id=len(per_node), target_id=target_id,
                    raw=sig.astype(np.float32), label=cls,
                ))
        shared = [s for s in per_node if s.target_id.startswith("vehicle_")]
        return per_node, shared


class AnomalyDataGenerator:
    LABELS = ["normal", "wear", "defect"]

    def __init__(self, seed: int = 42, samples_per_type: int = 200):
        self.rng = np.random.default_rng(seed)
        self.samples_per_type = samples_per_type

    def composite_defect(self) -> list:
        out = []
        for _ in range(self.samples_per_type):
            t = np.linspace(0, 1, 64)
            n_harmonics = int(self.rng.integers(2, 6))
            sig = np.zeros_like(t)
            for _ in range(n_harmonics):
                f = float(self.rng.uniform(2, 15))
                amp = float(self.rng.uniform(0.1, 0.5))
                phase = float(self.rng.uniform(0, 2 * np.pi))
                sig += amp * np.sin(2 * np.pi * f * t + phase)
            sig += self.rng.normal(0, 0.15, size=t.shape)
            out.append(Sample(
                sample_id=len(out), target_id=f"composite_{len(out)}",
                raw=sig.astype(np.float32), label="defect",
                anomaly_type="composite_defect",
            ))
        return out

    def em_interference(self) -> list:
        out = []
        for _ in range(self.samples_per_type):
            t = np.linspace(0, 1, 64)
            base = 0.5 * np.sin(2 * np.pi * 5 * t) + self.rng.normal(0, 0.05, size=t.shape)
            loss_mask = self.rng.random(64) < 0.15
            base[loss_mask] = 0.0
            rf = 0.3 * np.sin(2 * np.pi * 50 * t) * (self.rng.random() > 0.3)
            sig = base + rf
            out.append(Sample(
                sample_id=len(out), target_id=f"em_{len(out)}",
                raw=sig.astype(np.float32), label="wear",
                anomaly_type="em_interference",
            ))
        return out

    def weather_blur(self) -> list:
        out = []
        for _ in range(self.samples_per_type):
            t = np.linspace(0, 1, 64)
            attenuation = float(self.rng.uniform(0.3, 0.7))
            drift = float(self.rng.uniform(-0.5, 0.5))
            sig = attenuation * np.sin(2 * np.pi * 5 * t) + \
                  drift * np.sin(2 * np.pi * 0.3 * t) + \
                  self.rng.normal(0, 0.08, size=t.shape)
            out.append(Sample(
                sample_id=len(out), target_id=f"weather_{len(out)}",
                raw=sig.astype(np.float32), label="normal",
                anomaly_type="weather_blur",
            ))
        return out

    def net_outage(self) -> list:
        out = []
        for _ in range(self.samples_per_type):
            t = np.linspace(0, 1, 64)
            sig = 0.5 * np.sin(2 * np.pi * 5 * t) + self.rng.normal(0, 0.05, size=t.shape)
            gap_start = int(self.rng.integers(10, 50))
            gap_len = int(self.rng.integers(10, 30))
            sig[gap_start: gap_start + gap_len] = 0
            out.append(Sample(
                sample_id=len(out), target_id=f"outage_{len(out)}",
                raw=sig.astype(np.float32), label="defect",
                anomaly_type="net_outage",
            ))
        return out

    def generate_all(self) -> list:
        return (
            self.composite_defect()
            + self.em_interference()
            + self.weather_blur()
            + self.net_outage()
        )
