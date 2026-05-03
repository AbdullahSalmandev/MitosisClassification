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
    summarize_architecture_signals,
)


@dataclass
class RuntimeConfig:
    checkpoint_file: Path = Path("inception_balanced_trained_weights_1.pth")
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
        self.binary_single_logit = False
        self.device = torch.device("cpu")

        self.transform = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_compatible_state_dict(
        self,
        model: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
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

    def _build_model_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> torch.nn.Module:

        if self.config.default_arch != "inception_v3":
            raise ValueError(
                f"Unsupported default_arch: {self.config.default_arch}"
            )

        # Detect custom binary classifier head
        if "fc.3.weight" in state_dict and torch.is_tensor(state_dict["fc.3.weight"]):
            print("[runtime] Detected custom binary head")

            model = models.inception_v3(
                num_classes=1000,
                aux_logits=True,
            )

            model.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(256, 1),
            )

            self.binary_single_logit = True

        else:
            print("[runtime] Using standard multi-class head")

            model = models.inception_v3(
                num_classes=self.config.num_classes,
                aux_logits=True,
            )

            self.binary_single_logit = False

        missing, unexpected = self._load_compatible_state_dict(
            model,
            state_dict,
        )

        if missing:
            print(f"[runtime] Missing keys: {len(missing)}")

        if unexpected:
            print(f"[runtime] Unexpected keys: {len(unexpected)}")

        return model

    def load(self) -> None:
        checkpoint_path = self.config.checkpoint_file

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found at '{checkpoint_path}'"
            )

        print(f"[runtime] Loading checkpoint: {checkpoint_path}")

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        self.payload_type = detect_payload_type(payload)

        if isinstance(payload, torch.nn.Module):
            state_dict = payload.state_dict()
        else:
            state_dict = extract_state_dict(payload)

        model = self._build_model_from_state_dict(state_dict)

        self.architecture_signals = summarize_architecture_signals(
            state_dict
        )

        self.model = model.to(self.device).eval()

        if self.config.labels_file.exists():
            self.labels = json.loads(
                self.config.labels_file.read_text(
                    encoding="utf-8"
                )
            )
        else:
            self.labels = list(self.config.class_names)

        print(f"[runtime] Labels loaded: {self.labels}")

    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if self.model is None:
            raise RuntimeError("Model is not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)

        if isinstance(logits, tuple):
            logits = logits[0]

        # FIXED binary mapping logic
        if self.binary_single_logit and logits.ndim == 2 and logits.shape[1] == 1:
            score = float(torch.sigmoid(logits)[0, 0].item())

            # IMPORTANT:
            # Observed behavior suggests:
            # sigmoid high = non_mitotic
            # sigmoid low  = mitotic

            mitotic_score = float(1.0 - score)
            non_mitotic_score = float(score)

            labels = (
                self.labels
                if len(self.labels) >= 2
                else list(self.config.class_names)
            )

            ranked = sorted(
                [
                    {
                        "index": 0,
                        "label": labels[0],  # mitotic
                        "score": mitotic_score,
                    },
                    {
                        "index": 1,
                        "label": labels[1],  # non_mitotic
                        "score": non_mitotic_score,
                    },
                ],
                key=lambda item: item["score"],
                reverse=True,
            )

            return ranked[: min(top_k, len(ranked))]

        # Standard multi-class handling
        probs = torch.softmax(logits, dim=1)[0]

        vals, idxs = torch.topk(
            probs,
            k=min(top_k, probs.numel()),
        )

        results = []

        for score, idx in zip(vals.tolist(), idxs.tolist()):
            label = (
                self.labels[idx]
                if idx < len(self.labels)
                else f"class_{idx}"
            )

            results.append(
                {
                    "index": idx,
                    "label": label,
                    "score": float(score),
                }
            )

        return results