"""LoRA SFT on prompt / completion rows (src/train/sft_data.py), loss on the completion only; then merge and save a
plain checkpoint that vLLM serves like the base model. Self-contained (torch, transformers, peft, trl, datasets) so
it runs in the box's training env:

    python src/train/sft_lora.py --data data/train/q35_v0 --base Qwen/Qwen3.5-2B --out data/models/q35_v0

Hyper-parameters follow datamate's SFT runs (r 32, alpha 32, dropout 0.05, all linear, lr 1e-4; 5 epochs, best val-loss checkpoint).
Multimodal bases (Qwen3.5): LoRA on the language model only; the vision tower is untouched and saved as is.
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def complete_config(base: str, out: Path) -> None:
    """Copy processor / chat-template files vLLM expects next to the checkpoint — never weights or a weight index
    (2026-11-19 bug: the base's index.json + shard were copied and would have made vLLM serve the BASE weights)."""
    from huggingface_hub import snapshot_download

    base_dir = Path(snapshot_download(base, allow_patterns=["*.json", "*.jinja"]))
    for f in base_dir.iterdir():
        if f.suffix in (".json", ".jinja") and not f.name.endswith(".index.json") and not (out / f.name).exists():
            shutil.copy(f, out / f.name)


def clean_merged(base: str, out: Path) -> None:
    """Remove base weight shards / weight index wrongly copied into `out`: same name as a base snapshot file and the
    same bytes (size + first MiB). Our own merged weights always differ, so they are kept."""
    from huggingface_hub import snapshot_download

    base_dir = Path(snapshot_download(base, allow_patterns=["*.json"]))
    for f in list(out.iterdir()):
        b = (base_dir / f.name)
        if not (f.name.endswith(".index.json") or ".safetensors" in f.name) or not b.exists():
            continue
        b = b.resolve()
        with open(f, "rb") as x, open(b, "rb") as y:
            same = f.stat().st_size == b.stat().st_size and x.read(1 << 20) == y.read(1 << 20)
        if same:
            print(f"removing copied base file {f}")
            f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)   # small data; best val checkpoint guards overfit
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--eval-steps", type=int, default=30, help="val loss every N steps; best val checkpoint is kept")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows, a few steps")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--clean-only", action="store_true", help="only remove base files wrongly copied into --out")
    a = ap.parse_args()
    if a.clean_only:
        return clean_merged(a.base, Path(a.out))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0, data, out = time.time(), Path(a.data), Path(a.out)
    ds = load_dataset("json", data_files=str(data / "train.jsonl"), split="train").remove_columns(["qid"])
    val = (load_dataset("json", data_files=str(data / "val.jsonl"), split="train").remove_columns(["qid"])
           if (data / "val.jsonl").exists() else None)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
        val = val.select(range(min(8, len(val)))) if val is not None else None
    every = 2 if a.limit else a.eval_steps
    evals = dict(eval_strategy="steps", eval_steps=every, save_strategy="steps", save_steps=every, save_total_limit=2,
                 load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                 per_device_eval_batch_size=a.batch) if val is not None else dict(save_strategy="no")
    tok = AutoTokenizer.from_pretrained(a.base)
    multimodal = hasattr(AutoConfig.from_pretrained(a.base), "vision_config")
    Model = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = Model.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")

    lora = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear", exclude_modules=r".*(visual|vision|lm_head).*" if multimodal else None)
    args = SFTConfig(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, max_steps=4 if a.limit else -1,
                     per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, learning_rate=a.lr,
                     lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0, max_grad_norm=1.0, bf16=True,  # warmup < 1 = ratio
                     gradient_checkpointing=True, logging_steps=5, report_to=[], seed=a.seed, **evals,
                     max_length=a.max_length, completion_only_loss=True, dataset_num_proc=4)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, eval_dataset=val, processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    result = trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    complete_config(a.base, out)
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    hist = trainer.state.log_history
    rec = {"base": a.base, "data": str(data), "rows": len(ds), "val_rows": len(val) if val is not None else 0,
           "args": vars(a), "train_loss": result.training_loss, "steps": result.global_step,
           "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
           "eval": [{"step": r["step"], "eval_loss": r["eval_loss"]} for r in hist if "eval_loss" in r], "minutes": round((time.time() - t0) / 60, 1),
           "log": [r for r in hist if "loss" in r]}
    (out / "train_report.json").write_text(json.dumps(rec, indent=1))
    print(f"done: {out} loss {result.training_loss:.4f}, best val {rec['best_eval_loss']} at {rec['best_checkpoint']}, "
          f"{rec['minutes']} min")


if __name__ == "__main__":
    main()
