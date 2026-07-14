"""CWRU real data loader (skeleton; data downloaded separately)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class CWRUSample:
    sample_id: int
    target_id: str
    raw: np.ndarray
    label: str


CWRU_3CLASS = {
    0: "normal",
    1: "wear", 2: "wear", 3: "defect",
    4: "wear", 5: "defect", 6: "defect",
    7: "wear", 8: "defect", 9: "defect",
}


class CWRUBearingDataLoader:
    LABELS = ["normal", "wear", "defect"]

    def __init__(self, csv_path: str = None, samples: int = 1000,
                 num_nodes: int = 4, segment_len: int = 256, seed: int = 0):
        if csv_path is None:
            csv_path = str(Path(__file__).resolve().parent.parent
                           / "data" / "datasets" / "cwru_bearing" / "CWRU1024" / "12kDriveEnd.csv")
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CWRU dataset not found: {self.csv_path}\n"
                f"Download via scripts/download_datasets.py first."
            )
        self.samples = samples
        self.num_nodes = num_nodes
        self.segment_len = segment_len
        self.rng = np.random.default_rng(seed)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.csv_path, header=None)
        signals = df.iloc[:, :-1].values.astype(np.float32)
        raw_labels = df.iloc[:, -1].values.astype(int)
        mask = (raw_labels >= 0) & (raw_labels <= 9)
        signals = signals[mask]
        raw_labels = raw_labels[mask]
        n_segments_per_signal = 1024 // self.segment_len
        all_segs = []
        all_labels = []
        for sig, lbl in zip(signals, raw_labels):
            for i in range(n_segments_per_signal):
                seg = sig[i * self.segment_len: (i + 1) * self.segment_len]
                if len(seg) == self.segment_len:
                    all_segs.append(seg)
                    all_labels.append(CWRU_3CLASS[int(lbl)])
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
            if i % 25 == 0 and i >= self.num_nodes:
                target_id = f"shared_dev_{i // 25}"
            else:
                target_id = f"edge_{(i % self.num_nodes) + 1}_dev_{i}"
            per_node.append(CWRUSample(
                sample_id=i, target_id=target_id,
                raw=seg, label=str(lbl),
            ))
        shared = [s for s in per_node if s.target_id.startswith("shared_dev_")]
        expanded = list(per_node)
        for s in shared:
            for offset in range(1, 2):
                expanded.append(CWRUSample(
                    sample_id=s.sample_id + 10_000 * offset,
                    target_id=s.target_id, raw=s.raw.copy(), label=s.label,
                ))
        return expanded, shared
