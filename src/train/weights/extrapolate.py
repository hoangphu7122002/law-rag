"""Weight-space dial along a tuned direction: theta(alpha) = base + alpha * (tuned - base). alpha 0 = base, 1 = tuned, >1 extrapolates (more of what the tuning changed),
< 1 interpolates. Float tensors only (computed in float32, stored in the base dtype); other tensors are copied from the base.

    python -m train.weights.extrapolate BASE_DIR TUNED_DIR OUT_DIR --alpha 1.5
"""
import argparse
import shutil
from pathlib import Path

import torch


def extrapolate(base: dict, tuned: dict, alpha: float) -> dict:
    out = {}
    for k, b in base.items():
        if b.is_floating_point() and k in tuned:
            out[k] = (b.float() + alpha * (tuned[k].float() - b.float())).to(b.dtype)
        else:
            out[k] = b
    return out


def main() -> None:
    from safetensors.torch import load_file, save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tuned")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, required=True)
    a = ap.parse_args()
    base, tuned = load_file(f"{a.base}/model.safetensors"), load_file(f"{a.tuned}/model.safetensors")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(a.tuned).iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, out / f.name)
    save_file(extrapolate(base, tuned, a.alpha), str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"alpha={a.alpha} -> {out}")


if __name__ == "__main__":
    main()
