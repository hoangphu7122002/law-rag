"""DPO on top of an SFT checkpoint with the model's own best-vs-worst samples (src/train/pref_data.py pairs).

    python src/train/dpo.py --pairs data/train/pref_v3b/pairs.jsonl --base data/models/q35_v3b --out data/models/q35_dpo0 [--limit 16]

LoRA on the policy, the reference = the same weights with the adapter off (no second model in memory). Output = merged weights, same as grpo.py.
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=7168)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="weight of an added SFT (NLL of the chosen answer) loss next to the sigmoid DPO loss; 0 = plain DPO")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N pairs, 5 steps")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    t0, out = time.time(), Path(a.out)
    rows = [json.loads(l) for l in open(a.pairs)]
    if a.limit:
        rows = rows[:a.limit]
    end = "<|im_end|>"
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"] + end, "rejected": r["rejected"] + end} for r in rows])
    tok = AutoTokenizer.from_pretrained(a.base)
    multimodal = hasattr(AutoConfig.from_pretrained(a.base), "vision_config")
    Model = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = Model.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=a.r, lora_alpha=a.r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
                      exclude_modules=r".*(visual|vision|lm_head).*" if multimodal else None)
    args = DPOConfig(output_dir=str(out / "ckpt"), beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=5 if a.limit else -1,
                     per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, max_length=a.max_length,
                     **({"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, a.sft_weight]} if a.sft_weight else {}), bf16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
                     report_to=[], seed=a.seed, lr_scheduler_type="constant_with_warmup", warmup_steps=5)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    if Path(a.base).is_dir():
        for f in Path(a.base).iterdir():
            if f.suffix in (".json", ".jinja") and not f.name.endswith(".index.json") and not (out / f.name).exists():
                shutil.copy(f, out / f.name)
    (out / "train_report.json").write_text(json.dumps({"args": vars(a), "pairs": len(rows), "minutes": round((time.time() - t0) / 60, 1),
                                                        "log": [r for r in trainer.state.log_history if "loss" in r]}, indent=1))
    print(f"done: {out} in {round((time.time() - t0) / 60, 1)} min")


if __name__ == "__main__":
    main()
