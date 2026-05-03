from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import torch

from checkpoint_tools import (
    dataclass_list_to_dict,
    detect_payload_type,
    extract_state_dict,
    repack_extracted_checkpoint,
    save_json,
    summarize_architecture_signals,
    tensor_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect extracted torch checkpoint folder")
    parser.add_argument("--extracted-dir", type=Path, required=True, help="Path containing data.pkl/data/version")
    parser.add_argument("--repacked-file", type=Path, default=Path("artifacts/recovered_checkpoint.pth"))
    parser.add_argument("--report-json", type=Path, default=Path("artifacts/checkpoint_report.json"))
    parser.add_argument("--tensors-json", type=Path, default=Path("artifacts/tensors.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repacked = repack_extracted_checkpoint(args.extracted_dir, args.repacked_file)
    payload = torch.load(repacked, map_location="cpu", weights_only=False)

    payload_type = detect_payload_type(payload)
    print(f"[+] Payload type: {payload_type}")

    state_dict = extract_state_dict(payload)
    tensors = tensor_inventory(state_dict)
    architecture = summarize_architecture_signals(state_dict)

    report = {
        "extracted_dir": str(args.extracted_dir),
        "repacked_file": str(repacked),
        "payload_type": payload_type,
        "num_state_dict_entries": len(state_dict),
        "architecture_signals": architecture,
        "checkpoint_keys": list(payload.keys())[:40] if isinstance(payload, dict) else [],
    }
    save_json(args.report_json, report)
    save_json(args.tensors_json, dataclass_list_to_dict(tensors))

    print("[+] Wrote report:", args.report_json)
    print("[+] Wrote tensor inventory:", args.tensors_json)
    print("[+] Top architecture signals:")
    pprint(architecture["signals"])


if __name__ == "__main__":
    main()
