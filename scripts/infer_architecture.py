from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer likely architecture from tensor metadata")
    parser.add_argument("--report-json", type=Path, default=Path("artifacts/checkpoint_report.json"))
    parser.add_argument("--tensors-json", type=Path, default=Path("artifacts/tensors.json"))
    return parser.parse_args()


def infer(tensors: list[dict], report: dict) -> list[str]:
    names = [t["name"] for t in tensors]
    max_rank = max((len(t["shape"]) for t in tensors), default=0)
    signals = report.get("architecture_signals", {}).get("signals", {})
    hints: list[str] = []

    if any("mixed" in n.lower() for n in names) or signals.get("inception_signals_detected", 0) > 0:
        hints.append("Likely Inception-style backbone (layer names include 'mixed' or 'inception').")
    if any("aux" in n.lower() for n in names):
        hints.append("Auxiliary classifier seems present (common in Inception-v3 training).")
    if any("fc" in n.lower() for n in names):
        last_fc = [t for t in tensors if "fc" in t["name"].lower() and len(t["shape"]) == 2]
        if last_fc:
            classes = last_fc[-1]["shape"][0]
            hints.append(f"Classifier head indicates approximately {classes} output classes.")
    if max_rank >= 4:
        hints.append("Contains 4D tensors, so this is a CNN vision model.")
    if not hints:
        hints.append("No strong naming signal found; reconstruct by mapping tensor shapes to candidate backbones.")
    return hints


def main() -> None:
    args = parse_args()
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    tensors = json.loads(args.tensors_json.read_text(encoding="utf-8"))
    hints = infer(tensors, report)
    print("[+] Architecture reconstruction hints:")
    for i, hint in enumerate(hints, start=1):
        print(f"{i}. {hint}")


if __name__ == "__main__":
    main()
