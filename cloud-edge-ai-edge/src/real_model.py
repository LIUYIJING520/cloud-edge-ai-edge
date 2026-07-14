"""Real DeepSeek-R1-Distill-Qwen-1.5B inference (INT8 quantized)."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RealEdgeResult:
    prediction: str
    confidence: float
    latency_ms: float
    memory_mb: float
    raw_response: str = ""


class RealEdgeModel:
    LABELS = ["normal", "wear", "defect"]

    PROMPT_INDUSTRIAL = (
        "工业振动信号分析。RMS={rms:.3f}, 峰值={peak:.3f}, "
        "高频能量比={hf:.3f}, 谐波比={kurt:.2f}。"
        "请用 1 个词回答状态: [normal] [wear] [defect]"
    )

    def __init__(self, model_dir: str = None, quantize: bool = True, device: str = "cpu"):
        if model_dir is None:
            model_dir = str(
                Path(__file__).resolve().parent.parent
                / "data" / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
            )
        self.model_dir = Path(model_dir)
        self.quantize = quantize
        self.device = device
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._mem_mb = 0.0
        self._rng = np.random.default_rng(42)

    def load(self):
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError("pip install torch transformers first")
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True
        )
        dtype = torch.float32 if self.quantize else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.quantize:
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self._model.eval()
        try:
            import psutil
            self._mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            self._mem_mb = 0.0
        self._loaded = True
        print(f"    模型加载: {time.time() - t0:.1f}s, 内存 {self._mem_mb:.0f} MB")

    def _extract_features(self, raw):
        mean = float(raw.mean())
        std = float(raw.std() + 1e-9)
        kurt = float(((raw - mean) ** 4).mean() / (std ** 4 + 1e-9) - 3)
        energy = float((raw ** 2).mean())
        peak = float(np.abs(raw).max())
        if raw.size > 4:
            diff = np.diff(raw)
            hf = float((diff ** 2).mean()) / (energy + 1e-9)
        else:
            hf = 0.0
        rms = float(np.sqrt(energy))
        return {
            "mean": mean, "std": std, "kurt": kurt,
            "energy": energy, "peak": peak, "hf": hf, "rms": rms,
        }

    def _parse_label(self, response: str) -> Tuple[str, float]:
        response_lower = response.lower()
        scores = {}
        for label in self.LABELS:
            if f"[{label}]" in response_lower:
                scores[label] = 0.9
            elif label in response_lower:
                scores[label] = 0.6
            else:
                keyword_map = {"normal": "正常", "wear": "磨损", "defect": "缺陷"}
                if keyword_map.get(label, "") in response:
                    scores[label] = 0.5
        if not scores:
            label = self._rng.choice(self.LABELS)
            return label, 0.4
        best_label = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(0.95, max(0.5, scores[best_label]))
        return best_label, confidence

    def infer(self, raw, max_new_tokens: int = 12) -> RealEdgeResult:
        self.load()
        import torch
        t0 = time.time()
        feats = self._extract_features(raw)
        prompt = self.PROMPT_INDUSTRIAL.format(
            rms=feats["rms"], peak=feats["peak"],
            hf=feats["hf"], kurt=feats["kurt"],
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        latency_ms = (time.time() - t0) * 1000
        pred, conf = self._parse_label(response)
        return RealEdgeResult(
            prediction=pred, confidence=conf,
            latency_ms=latency_ms, memory_mb=self._mem_mb,
            raw_response=response.strip(),
        )
