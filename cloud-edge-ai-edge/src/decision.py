"""Global decision coordinator: cross-node arbitration."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .edge_model import CloudFullModel


@dataclass
class ConflictRecord:
    conflict_id: int
    target_id: str
    edge_decisions: Dict[str, Tuple[Any, float]]
    cloud_decision: Optional[Any] = None
    resolved: bool = False
    resolve_strategy: str = ""
    latency_ms: float = 0.0


class GlobalDecisionCoordinator:
    """Detect cross-node conflicts and arbitrate with weighted voting."""

    def __init__(self, cfg: SystemConfig, cloud: CloudFullModel):
        self.cfg = cfg
        self.cloud = cloud
        self.rng = np.random.default_rng(cfg.random_seed + 31)
        self.node_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.conflicts: List[ConflictRecord] = []
        self._conflict_seq = 0
        self._decision_buffer: Dict[str, Dict[str, Tuple[Any, float]]] = defaultdict(dict)
        self._buffer_window_steps: int = 6
        self._step_counter = 0
        self._last_seen_step: Dict[str, int] = {}

    def _detect_conflict(self, target_id, node_id, prediction, confidence):
        self._step_counter += 1
        expired = [tid for tid, st in self._last_seen_step.items()
                   if self._step_counter - st > self._buffer_window_steps]
        for tid in expired:
            self._decision_buffer.pop(tid, None)
        buf = self._decision_buffer[target_id]
        buf[node_id] = (prediction, confidence)
        self._last_seen_step[target_id] = self._step_counter
        preds = {nid: p for nid, (p, _) in buf.items()}
        if len(preds) >= 2 and len(set(preds.values())) > 1:
            self._conflict_seq += 1
            rec = ConflictRecord(
                conflict_id=self._conflict_seq,
                target_id=target_id,
                edge_decisions=dict(buf),
            )
            self.conflicts.append(rec)
            self._decision_buffer.pop(target_id, None)
            self._last_seen_step.pop(target_id, None)
            return rec
        return None

    def arbitrate(self, rec: ConflictRecord) -> ConflictRecord:
        edge_dec = rec.edge_decisions
        best_node = max(
            edge_dec.keys(),
            key=lambda nid: edge_dec[nid][1] * self.node_weights[nid],
        )
        tentative = edge_dec[best_node][0]
        if self.node_weights[best_node] >= 1.0:
            final = tentative
            strategy = "weighted_vote"
        else:
            vote: Dict[Any, float] = defaultdict(float)
            for nid, (p, c) in edge_dec.items():
                vote[p] += c * self.node_weights[nid]
            final = max(vote.keys(), key=lambda k: vote[k])
            strategy = "reweighted_vote"
        rec.cloud_decision = final
        rec.resolved = True
        rec.resolve_strategy = strategy
        rec.latency_ms = self.cfg.cloud_model.avg_latency_ms
        for nid, (p, c) in edge_dec.items():
            if p == final:
                self.node_weights[nid] = min(1.5, self.node_weights[nid] + 0.05)
            else:
                self.node_weights[nid] = max(0.5, self.node_weights[nid] - 0.05)
        return rec

    def submit_decision(self, target_id, node_id, prediction, confidence):
        rec = self._detect_conflict(target_id, node_id, prediction, confidence)
        if rec is not None:
            return self.arbitrate(rec)
        return None

    def broadcast_weights(self) -> Dict[str, float]:
        return dict(self.node_weights)
