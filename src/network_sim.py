"""
Network fluctuation simulator.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkCondition:
    node_id: str
    quality: float
    online: bool = True
    bandwidth_mbps: float = 100.0
    rtt_ms: float = 30.0
    packet_loss_pct: float = 0.0


class NetworkSimulator:
    def __init__(self, node_ids, seed: int = 0,
                 baseline_quality: float = 0.85,
                 fault_probability: float = 0.04):
        self.rng = np.random.default_rng(seed)
        self.node_ids = list(node_ids)
        self.baseline_quality = baseline_quality
        self.fault_probability = fault_probability
        self._state = {nid: "normal" for nid in self.node_ids}
        self._remaining = {nid: 0 for nid in self.node_ids}
        self._t = 0

    def step(self) -> dict:
        self._t += 1
        states = {}
        for nid in self.node_ids:
            osc = 0.10 * np.sin(self._t * 0.15 + hash(nid) % 7)
            noise = float(self.rng.normal(0, 0.03))
            q = float(np.clip(self.baseline_quality + osc + noise, 0.0, 1.0))

            if self._remaining[nid] > 0:
                self._remaining[nid] -= 1
                if self._state[nid] == "weak_instant":
                    q = float(self.rng.uniform(0.30, 0.55))
                    packet_loss = 10.0
                elif self._state[nid] == "long_outage":
                    q = float(self.rng.uniform(0.05, 0.20))
                    packet_loss = 50.0
                elif self._state[nid] == "em_packetloss":
                    q = float(self.rng.uniform(0.45, 0.65))
                    packet_loss = float(self.rng.uniform(30.0, 50.0))
                else:
                    packet_loss = 0.0
                if self._remaining[nid] == 0:
                    self._state[nid] = "normal"
            else:
                packet_loss = 0.0
                r = float(self.rng.random())
                if r < self.fault_probability * 0.5:
                    self._state[nid] = "weak_instant"
                    self._remaining[nid] = int(self.rng.integers(1, 4))
                    q = float(self.rng.uniform(0.30, 0.55))
                    packet_loss = 10.0
                elif r < self.fault_probability:
                    state_roll = float(self.rng.random())
                    if state_roll < 0.7:
                        self._state[nid] = "long_outage"
                        self._remaining[nid] = int(self.rng.integers(8, 31))
                        q = float(self.rng.uniform(0.05, 0.20))
                        packet_loss = 50.0
                    else:
                        self._state[nid] = "em_packetloss"
                        self._remaining[nid] = int(self.rng.integers(5, 16))
                        q = float(self.rng.uniform(0.45, 0.65))
                        packet_loss = float(self.rng.uniform(30.0, 50.0))

            online = q > 0.10
            states[nid] = NetworkCondition(
                node_id=nid, quality=q, online=online,
                bandwidth_mbps=100.0 * q, rtt_ms=120.0 * (1.0 - q) + 10.0,
                packet_loss_pct=packet_loss,
            )
        return states
