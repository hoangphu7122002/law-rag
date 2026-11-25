"""Continued pretraining of the answer model on the law text (docs/research: 'pretrain Qwen3.5-2B on the laws our questions are about').

    torchrun --nproc_per_node 2 src/train/cpt.py --data data/cpt/laws.jsonl --base Qwen/Qwen3.5-2B --out data/models/q35_cpt [--lora-r 64] [--full]
Both GPUs, data parallel. Plain next-token loss on packed sequences (train.cpt_data); a small held-out slice reports the loss. The result is a
merged bf16 model, the starting point of the normal SFT (src/train/sft_lora.py --base <out>).
"""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="full-parameter CPT (lr should then be ~1e-5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N sequences, a few steps")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, Trainer, TrainingArguments, default_data_collator

    from train.sft_lora import complete_config

    seqs = [json.loads(line)["input_ids"] for line in Path(a.data).read_text().splitlines() if line.strip()]
    if a.limit:
        seqs = seqs[: a.limit]
    cut = max(1, len(seqs) // 50)
    rows = lambda xs: [{"input_ids": s, "labels": s, "attention_mask": [1] * len(s)} for s in xs]   # noqa: E731
    from datasets import Dataset

    train, val = Dataset.from_list(rows(seqs[cut:])), Dataset.from_list(rows(seqs[:cut]))
    multimodal = hasattr(AutoConfig.from_pretrained(a.base), "vision_config")
    Model = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = Model.from_pretrained(a.base, dtype=torch.float32 if a.full else torch.bfloat16, attn_implementation="sdpa")
    if not a.full:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
                                                 exclude_modules=r".*(visual|vision|lm_head).*" if multimodal else None))
    args = TrainingArguments(output_dir=str(Path(a.out) / "ckpt"), num_train_epochs=a.epochs, max_steps=4 if a.limit else -1, per_device_train_batch_size=a.batch,
                             per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
                             weight_decay=0.0, max_grad_norm=1.0, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                             ddp_find_unused_parameters=False, logging_steps=5, eval_strategy="epoch", save_strategy="no", report_to=[], seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=default_data_collator)
    t0 = time.time()
    result = trainer.train()
    if trainer.is_world_process_zero():
        merged = (model.to(torch.bfloat16) if a.full else trainer.model.merge_and_unload())
        merged.save_pretrained(a.out, safe_serialization=True)
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
        complete_config(a.base, Path(a.out))
        hist = trainer.state.log_history
        (Path(a.out) / "cpt_report.json").write_text(json.dumps({"args": vars(a), "train_loss": result.training_loss, "sequences": len(train),
                                                                    "minutes": round((time.time() - t0) / 60, 1), "eval": [r for r in hist if "eval_loss" in r]}, indent=1))
        print(f"done: {a.out} train loss {result.training_loss:.4f}, {round((time.time() - t0) / 60, 1)} min")


if __name__ == "__main__":
    main()
