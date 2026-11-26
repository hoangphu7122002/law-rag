"""Fine-tune the embedder (AITeamVN/Vietnamese_Embedding) on the rerank-data groups, then score it with EmbedBench.

Groups = data/train/rr_v0/train.jsonl rows {"query", "pos": [text], "neg": [text x7]} (course train + Zalo, decontaminated vs test).
Loss = MultipleNegativesRanking (in-batch + the group's hard negatives), via CachedMNRL so the effective batch is large on one GPU.
`VARIANTS` = the grid we compare; each is LoRA (peft, merged after training) or full fine-tune.
Runs inside Modal (src/ops/modal_embed_ft.py) or anywhere with a GPU:  python src/train/embed_ft.py --variant lora32 --data ... --pool ...
"""
import json
import random
from pathlib import Path

BASE = "AITeamVN/Vietnamese_Embedding"
VARIANTS: dict[str, dict] = {
    "lora32": {"mode": "lora", "r": 32, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 20.0, "alphas": [0.25, 0.5, 0.75]},
    "lora8": {"mode": "lora", "r": 8, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 20.0},
    "lora64": {"mode": "lora", "r": 64, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 20.0},
    "lora32_inbatch": {"mode": "lora", "r": 32, "lr": 1e-4, "epochs": 1, "negatives": 0, "scale": 20.0},       # control: no hard negatives
    "lora32_t02": {"mode": "lora", "r": 32, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 50.0},           # temperature .02
    "lora32_e3": {"mode": "lora", "r": 32, "lr": 1e-4, "epochs": 3, "negatives": 7, "scale": 20.0},            # longer
    "full": {"mode": "full", "r": 0, "lr": 1e-5, "epochs": 1, "negatives": 7, "scale": 20.0, "alphas": [0.25, 0.5]},
    # conservative kinds (the first grid was worse than the base: AITeamVN saw ~1M triplets, we have 4k groups)
    "lora32_low": {"mode": "lora", "r": 32, "lr": 2e-5, "epochs": 1, "negatives": 7, "scale": 20.0, "alphas": [0.5]},
    "lora32_vlow": {"mode": "lora", "r": 32, "lr": 5e-6, "epochs": 1, "negatives": 7, "scale": 20.0},
    "full_low": {"mode": "full", "r": 0, "lr": 2e-6, "epochs": 1, "negatives": 7, "scale": 20.0, "alphas": [0.5]},
    # temperature .02 (scale 50) was the one grid cell above the base (Điều@1 .564 vs .534, no per-question test yet): repeat it and its low-lr / low-rank cousins
    "lora32_t02_low": {"mode": "lora", "r": 32, "lr": 2e-5, "epochs": 1, "negatives": 7, "scale": 50.0, "alphas": [0.5]},
    "lora8_t02": {"mode": "lora", "r": 8, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 50.0},
    "lora32_t02_s2": {"mode": "lora", "r": 32, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 50.0, "seed": 7},   # seed repeat: is .564 luck?
    "query32": {"mode": "query", "r": 32, "lr": 1e-4, "epochs": 1, "negatives": 7, "scale": 20.0},   # query tower only; documents stay = index stays valid
}


def variant_args(name: str) -> dict:
    return dict(VARIANTS[name])


def interpolate(base: dict, tuned: dict, alpha: float) -> dict:
    """Weight-space interpolation (WiSE-FT): base + alpha * (tuned - base) for every tensor; non-float entries come from `tuned`."""
    if base.keys() != tuned.keys():
        raise ValueError("weights differ: " + str(sorted(set(base) ^ set(tuned))[:3]))
    out = {}
    for k, b in base.items():
        t = tuned[k]
        floaty = getattr(b, "dtype", None) is not None and getattr(b.dtype, "kind", "f") == "f" if hasattr(b, "dtype") else isinstance(b, float)
        if hasattr(b, "is_floating_point"):
            floaty = b.is_floating_point()
        out[k] = b + alpha * (t - b) if floaty else t
    return out


def load_groups(path, negatives: int, limit: int = 0, seed: int = 20261031) -> list[dict]:
    """Rows for MNRL: {"anchor", "positive", "negative_1".. "negative_n"}. Groups with fewer than `negatives` negatives are dropped
    (a trainer needs the same columns in every row). Seeded shuffle, then `limit`."""
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for r in rows:
        if not r["pos"] or len(r["neg"]) < negatives:
            continue
        ex = {"anchor": r["query"], "positive": r["pos"][0]}
        ex |= {f"negative_{i + 1}": n for i, n in enumerate(r["neg"][:negatives])}
        out.append(ex)
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def query_only_loss_class():
    import torch
    from torch import nn

    class QueryOnlyLoss(nn.Module):
        """InfoNCE where the question goes through the trainable tower and positives / negatives through the frozen one."""

        def __init__(self, model, frozen, scale: float):
            super().__init__()
            self.model, self.frozen, self.scale = model, frozen.to(model.device), scale

        def forward(self, sentence_features, labels=None):
            q = torch.nn.functional.normalize(self.model(sentence_features[0])["sentence_embedding"], dim=-1)
            with torch.no_grad():
                d = torch.cat([torch.nn.functional.normalize(self.frozen(f)["sentence_embedding"], dim=-1) for f in sentence_features[1:]])
            scores = self.scale * q @ d.T
            return torch.nn.functional.cross_entropy(scores, torch.arange(len(q), device=q.device))

    return QueryOnlyLoss


def interpolated_model(base: str, tuned_dir: str, alpha: float, out: str, max_len: int = 512) -> str:
    """Save the model whose weights are base + alpha * (tuned - base); returns its directory. Needs torch (not unit-tested)."""
    from sentence_transformers import SentenceTransformer

    b = SentenceTransformer(base, device="cpu", trust_remote_code=True)
    t = SentenceTransformer(tuned_dir, device="cpu", trust_remote_code=True)
    b.load_state_dict(interpolate(b.state_dict(), t.state_dict(), alpha))
    b.max_seq_length = max_len
    b.save(out)
    return out


def train(name: str, data: str, out: str, base: str = BASE, batch: int = 128, mini_batch: int = 16, max_len: int = 512,
          limit: int = 0, seed: int | None = None) -> str:
    """Train one variant; returns the model directory (LoRA already merged). Needs a GPU (not unit-tested)."""
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    QueryOnlyLoss = query_only_loss_class()
    v = variant_args(name)
    seed = seed if seed is not None else v.get("seed", 20261031)
    ds = Dataset.from_list(load_groups(data, v["negatives"], limit, seed))
    model = SentenceTransformer(base, device="cuda", trust_remote_code=True)
    model.max_seq_length = max_len
    frozen = None
    if v["mode"] == "query":   # document tower = frozen copy of the base (what the index holds); only the query tower learns
        import copy
        frozen = copy.deepcopy(model).eval().requires_grad_(False)
    if v["mode"] in ("lora", "query"):
        from peft import LoraConfig, get_peft_model
        peft_model = get_peft_model(model[0].auto_model, LoraConfig(r=v["r"], lora_alpha=v["r"], lora_dropout=0.05, bias="none",
                                                              target_modules=["query", "key", "value", "dense"]))
        model[0].auto_model = peft_model
    loss = (QueryOnlyLoss(model, frozen, v["scale"]) if frozen is not None
            else CachedMultipleNegativesRankingLoss(model, scale=v["scale"], mini_batch_size=mini_batch))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out) / "ckpt"), num_train_epochs=v["epochs"], per_device_train_batch_size=batch, learning_rate=v["lr"],
        warmup_ratio=0.1, lr_scheduler_type="cosine", bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
        logging_steps=5, report_to=[], seed=seed)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    if v["mode"] in ("lora", "query"):
        model[0].auto_model = peft_model.merge_and_unload()
    model.save(str(Path(out) / "model"))
    del model
    torch.cuda.empty_cache()
    return str(Path(out) / "model")
