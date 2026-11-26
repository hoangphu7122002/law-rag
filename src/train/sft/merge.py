"""Training-free merging of full model checkpoints of one architecture (soup, task arithmetic).

Pure functions work on state dicts (name -> array-like); `main()` does the tensor-by-tensor safetensors I/O.
torch / safetensors are imported inside functions only, so unit tests run with numpy dicts."""


def _check_keys(models: list[dict]) -> None:
    for m in models[1:]:
        if m.keys() != models[0].keys():
            raise ValueError("weights differ: " + str(sorted(set(m) ^ set(models[0]))[:3]))


def _is_float(x) -> bool:
    if hasattr(x, "is_floating_point"):
        return bool(x.is_floating_point())
    return getattr(getattr(x, "dtype", None), "kind", None) == "f"


def _f32(x):
    return x.astype("float32") if hasattr(x, "astype") else x.float()


def _back(y, like):
    return y.astype(like.dtype) if hasattr(y, "astype") else y.to(like.dtype)


def soup(models: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of state dicts (weights normalised, uniform by default); non-float entries come from the first model."""
    _check_keys(models)
    w = [1.0] * len(models) if weights is None else list(weights)
    if len(w) != len(models):
        raise ValueError("need one weight per model")
    total = float(sum(w))
    out = {}
    for k, first in models[0].items():
        if not _is_float(first):
            out[k] = first
            continue
        acc = _f32(first) * (w[0] / total)
        for m, wi in zip(models[1:], w[1:]):
            acc = acc + _f32(m[k]) * (wi / total)
        out[k] = _back(acc, first)
    return out


def task_arithmetic(base: dict, deltas: list[tuple[dict, dict, float]]) -> dict:
    """base + sum(alpha * (tuned - origin)) over (tuned, origin, alpha) deltas; non-float entries come from `base`."""
    _check_keys([base] + [d for t, o, _ in deltas for d in (t, o)])
    out = {}
    for k, b in base.items():
        if not _is_float(b):
            out[k] = b
            continue
        acc = _f32(b)
        for tuned, origin, alpha in deltas:
            acc = acc + alpha * (_f32(tuned[k]) - _f32(origin[k]))
        out[k] = _back(acc, b)
    return out


SHARD_BYTES = 5 * 10**9


def _index(model_dir) -> dict[str, str]:
    """tensor name -> shard file path for a checkpoint dir (single file or sharded index json)."""
    import json
    from pathlib import Path

    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return {k: str(d / f) for k, f in json.loads(idx.read_text())["weight_map"].items()}
    from safetensors import safe_open

    files = sorted(d.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {d}")
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            out |= {k: str(f) for k in h.keys()}
    return out


class _Reader:
    """Lazy per-tensor reader over one checkpoint dir; keeps at most one shard handle open."""

    def __init__(self, model_dir):
        self.map = _index(model_dir)
        self._path, self._h = None, None

    def get(self, name):
        from safetensors import safe_open

        p = self.map[name]
        if p != self._path:
            self._h = safe_open(p, framework="pt")
            self._path = p
        return self._h.get_tensor(name)


def _copy_non_weights(src, out) -> None:
    import shutil
    from pathlib import Path

    for f in Path(src).iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.endswith(".index.json"):
            shutil.copy(f, Path(out) / f.name)


def _write_sharded(names, produce, out) -> None:
    """Save `produce(name)` tensors in shards <= SHARD_BYTES plus a regenerated model.safetensors.index.json."""
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for n in names:
        t = produce(n).contiguous()
        nbytes = t.numel() * t.element_size()
        if cur and size + nbytes > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[n] = t
        size += nbytes
    if cur:
        shards.append(cur)
    wm, total = {}, 0
    for i, sh in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(out / fn), metadata={"format": "pt"})
        for k, t in sh.items():
            wm[k] = fn
            total += t.numel() * t.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=1))
    shards.clear()


def _check_same(readers) -> list[str]:
    _check_keys([dict.fromkeys(r.map) for r in readers])
    return list(readers[0].map)


def soup_dirs(models: list[str], out: str, weights: list[float] | None = None) -> dict:
    readers = [_Reader(m) for m in models]
    names = _check_same(readers)
    _write_sharded(names, lambda n: soup([{n: r.get(n)} for r in readers], weights)[n], out)
    _copy_non_weights(models[0], out)
    return {"tensors": len(names), "models": len(models)}


def arith_dirs(base: str, tuned: str, origin: str, alpha: float, out: str) -> dict:
    rb, rt, ro = _Reader(base), _Reader(tuned), _Reader(origin)
    names = _check_same([rb, rt, ro])
    peak = [0.0]

    def produce(n):
        b, t, o = rb.get(n), rt.get(n), ro.get(n)
        if t.is_floating_point():
            peak[0] = max(peak[0], float((t.float() - o.float()).abs().max()))
        return task_arithmetic({n: b}, [({n: t}, {n: o}, alpha)])[n]

    _write_sharded(names, produce, out)
    _copy_non_weights(base, out)
    return {"tensors": len(names), "alpha": alpha, "max_abs_diff_tuned_vs_origin": peak[0]}


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("soup", help="weighted average of N checkpoints")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--weights", nargs="+", type=float)
    s.add_argument("--out", required=True)
    a = sub.add_parser("arith", help="base + alpha * (tuned - origin)")
    a.add_argument("--base", required=True)
    a.add_argument("--tuned", required=True)
    a.add_argument("--origin", required=True)
    a.add_argument("--alpha", type=float, default=1.0)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "soup":
        rep = soup_dirs(args.models, args.out, args.weights)
    else:
        rep = arith_dirs(args.base, args.tuned, args.origin, args.alpha, args.out)
    print(json.dumps(rep | {"out": args.out}))


if __name__ == "__main__":
    main()
