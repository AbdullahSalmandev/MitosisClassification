from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.model_runtime import ModelRuntime, RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-off local inference")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = ModelRuntime(RuntimeConfig())
    runtime.load()
    preds = runtime.predict(args.image.read_bytes(), top_k=args.top_k)
    for p in preds:
        print(f"{p['label']:20s} score={p['score']:.6f} idx={p['index']}")


if __name__ == "__main__":
    main()
