from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelAdapter:
    model: Any
    framework: str
    channel_first: bool
    batch_size: int = 128

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        if self.framework == "tensorflow":
            return np.asarray(
                self.model.predict(images, batch_size=self.batch_size, verbose=0),
                dtype=np.float32,
            )
        if self.framework == "torch":
            import torch

            self.model.eval()
            device = next(self.model.parameters()).device
            outputs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(images), self.batch_size):
                    batch = torch.from_numpy(images[start : start + self.batch_size]).float().to(device)
                    logits = self.model(batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.softmax(logits, dim=1)
                    outputs.append(probs.detach().cpu().numpy())
            return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        raise ValueError(f"Unknown framework: {self.framework}")
