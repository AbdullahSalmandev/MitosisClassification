from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from scripts.checkpoint_tools import (
    detect_payload_type,
    extract_state_dict,
    repack_extracted_checkpoint,
    summarize_architecture_signals,
)


@dataclass
class RuntimeConfig:
    extracted_checkpoint_dir: Path = Path("inception_balanced_trained_weights_1")
    repacked_checkpoint_file: Path = Path("artifacts/recovered_checkpoint.pth")
    labels_file: Path = Path("artifacts/labels.json")
    default_arch: str = "inception_v3"
    image_size: int = 299
    num_classes: int = 2
    class_names: tuple[str, str] = ("mitotic", "non_mitotic")


class ModelRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.model: torch.nn.Module | None = None
        self.payload_type: str | None = None
        self.architecture_signals: dict[str, Any] = {}
        self.labels: list[str] = []
        self.device = torch.device("cpu")
        self.transform = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_compatible_state_dict(
        self, model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        model_state = model.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
            else:
                dropped.append(key)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if dropped:
            print(f"[runtime] Dropped incompatible keys: {len(dropped)}")
        return list(missing), list(unexpected)

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
        if self.config.default_arch == "inception_v3":
            model = models.inception_v3(num_classes=self.config.num_classes, aux_logits=True)
        else:
            raise ValueError(f"Unsupported default_arch: {self.config.default_arch}")
        missing, unexpected = self._load_compatible_state_dict(model, state_dict)
        if missing:
            print(f"[runtime] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[runtime] Unexpected keys: {len(unexpected)}")
        return model

    def load(self) -> None:
        repacked = repack_extracted_checkpoint(
            self.config.extracted_checkpoint_dir, self.config.repacked_checkpoint_file
        )
        payload = torch.load(repacked, map_location="cpu", weights_only=False)
        self.payload_type = detect_payload_type(payload)

        state_dict = extract_state_dict(payload) if not isinstance(payload, torch.nn.Module) else payload.state_dict()
        model = self._build_model_from_state_dict(state_dict)

        self.architecture_signals = summarize_architecture_signals(state_dict)
        self.model = model.to(self.device).eval()

        if self.config.labels_file.exists():
            self.labels = json.loads(self.config.labels_file.read_text(encoding="utf-8"))
        else:
            self.labels = list(self.config.class_names)

    @torch.inference_mode()
    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        vals, idxs = torch.topk(probs, k=min(top_k, probs.numel()))
        results = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append({"index": idx, "label": label, "score": float(score)})
        return results
