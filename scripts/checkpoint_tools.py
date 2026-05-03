from __future__ import annotations

import json
import pickle
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    device: str
    numel: int
    requires_grad: bool


def repack_extracted_checkpoint(extracted_dir: Path, output_file: Path) -> Path:
    """
    Rebuild a loadable .pth/.pt file from an extracted torch zip folder.
    """
    required = ["data.pkl", "version", "byteorder", "data"]
    missing = [name for name in required if not (extracted_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required checkpoint members: {missing}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_file,
        "w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=False,
    ) as zf:
        for path in sorted(extracted_dir.rglob("*")):
            if not path.is_file():
                continue
            # Torch's zip reader expects checkpoint members under a shared archive prefix.
            arcname = f"archive/{path.relative_to(extracted_dir).as_posix()}"
            zf.write(path, arcname=arcname)
    return output_file


def _is_state_dict(candidate: Any) -> bool:
    if not isinstance(candidate, dict) or not candidate:
        return False
    values = list(candidate.values())
    return any(torch.is_tensor(v) for v in values)


def detect_payload_type(payload: Any) -> str:
    if isinstance(payload, torch.nn.Module):
        return "full_model"
    if _is_state_dict(payload):
        return "state_dict"
    if isinstance(payload, dict):
        if "state_dict" in payload and _is_state_dict(payload["state_dict"]):
            return "checkpoint_dict_with_state_dict"
        if "model_state_dict" in payload and _is_state_dict(payload["model_state_dict"]):
            return "checkpoint_dict_with_model_state_dict"
        return "generic_dict"
    return type(payload).__name__


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if _is_state_dict(payload):
        return payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            value = payload.get(key)
            if _is_state_dict(value):
                return value
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    raise ValueError("Unable to locate state_dict in payload")


def tensor_inventory(state_dict: dict[str, torch.Tensor]) -> list[TensorInfo]:
    rows: list[TensorInfo] = []
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        rows.append(
            TensorInfo(
                name=name,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                numel=int(tensor.numel()),
                requires_grad=bool(tensor.requires_grad),
            )
        )
    return rows


def summarize_architecture_signals(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = list(state_dict.keys())
    prefix_counts = Counter(name.split(".")[0] for name in names if "." in name)
    conv_like = [n for n in names if "conv" in n.lower()]
    bn_like = [n for n in names if "bn" in n.lower() or "batchnorm" in n.lower()]
    fc_like = [n for n in names if "fc" in n.lower() or "classifier" in n.lower() or "head" in n.lower()]
    inception_like = [n for n in names if "inception" in n.lower() or "mixed" in n.lower()]
    return {
        "num_parameters_tensors": len(names),
        "top_level_prefixes": prefix_counts.most_common(20),
        "signals": {
            "conv_layers_detected": len(conv_like),
            "batchnorm_layers_detected": len(bn_like),
            "fc_or_classifier_layers_detected": len(fc_like),
            "inception_signals_detected": len(inception_like),
        },
        "sample_layer_names": names[:40],
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def dataclass_list_to_dict(items: list[TensorInfo]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
