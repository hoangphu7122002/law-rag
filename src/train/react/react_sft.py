"""LoRA SFT of the ReAct student on its own good trajectories (src/train/react/react_build.py, datamate S1), loss on the ASSISTANT turns only
(content + tool calls + the closing <|im_end|>); system, user and tool-result text is masked. Self-contained in the box's training env:

    PYTHONPATH=src python -m train.react.react_sft --data data/train/react_s1 --base Qwen/Qwen3.5-2B --out data/models/react_s1

Rows: {"qid", "messages" (OpenAI style, full conversation), "tools" (tool specs)}; rendered with the base's chat template, non-thinking mode.
"""
import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path


def assistant_spans(messages: list[dict], render) -> list[tuple[int, int]]:
    """Character spans (start, end) of every assistant turn inside render(messages). `render(msgs, add_generation_prompt=False) -> str` is the
    chat template. Turn i starts where the generation prompt of messages[:i] ends and ends where messages[:i+1] ends (trailing newline excluded)."""
    full = render(messages)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start, end = len(render(messages[:i], add_generation_prompt=True)), len(render(messages[: i + 1]).rstrip("\n"))
        if not (render(messages[:i], add_generation_prompt=True) == full[:start] and render(messages[: i + 1]).startswith(full[:end])):
            raise ValueError(f"chat template is not prefix-stable at message {i}")
        spans.append((start, end))
    return spans


def _template_messages(messages: list[dict]) -> list[dict]:
    """Copy with tool-call arguments parsed to dicts (the Qwen template iterates them as a mapping; react.py stores the JSON string)."""
    out = []
    for m in messages:
        calls = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"] or "{}")
                                    if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]}} for c in m.get("tool_calls") or []]
        out.append({**m, "tool_calls": calls} if calls else dict(m))
    return out


def build_examples(rows: list[dict], tokenizer, max_length: int | None = None) -> list[dict]:
    """rows -> {"input_ids", "attention_mask", "labels"}; labels are -100 outside assistant turns. Rows longer than `max_length` tokens (never truncated,
    a cut trajectory would teach a half answer), with a template that is not prefix-stable, or without any trainable token are skipped."""
    examples = []
    for r in rows:
        msgs, tools = _template_messages(r["messages"]), r.get("tools")

        def render(ms, add_generation_prompt=False):
            return tokenizer.apply_chat_template(ms, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)

        try:
            spans = assistant_spans(msgs, render)
        except ValueError:
            continue
        enc = tokenizer(render(msgs), add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        labels = [tid if any(s <= a < e for s, e in spans) else -100 for tid, (a, _) in zip(ids, enc["offset_mapping"])]
        if (max_length and len(ids) > max_length) or all(x == -100 for x in labels):
            continue
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return examples


def _sft_lora():
    """complete_config lives in train/sft/sft_lora.py; loaded by path so this works as a script and as a module."""
    spec = importlib.util.spec_from_file_location("sft_lora", Path(__file__).resolve().parents[1] / "sft" / "sft_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda key, v: torch.tensor([b[key] + [v] * (n - len(b[key])) for b in batch])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.jsonl from react_build")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows, a few steps")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainingArguments

    t0, out = time.time(), Path(a.out)
    rows = [json.loads(line) for line in (Path(a.data) / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: a.limit] if a.limit else rows
    tok = AutoTokenizer.from_pretrained(a.base)
    examples = build_examples(rows, tok, a.max_length)
    trained = sum(x != -100 for e in examples for x in e["labels"])
    print(f"react sft: {len(examples)}/{len(rows)} trajectories fit (<= {a.max_length} tokens, template stable); {trained} trained tokens")
    if not examples:
        raise SystemExit("no trainable examples")

    multimodal = hasattr(AutoConfig.from_pretrained(a.base), "vision_config")
    Model = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = Model.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear", exclude_modules=r".*(visual|vision|lm_head).*" if multimodal else None)
    model.enable_input_require_grads()   # gradient checkpointing with a frozen embedding layer
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, max_steps=4 if a.limit else -1,
                             per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, learning_rate=a.lr,
                             lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0, max_grad_norm=1.0, bf16=True,
                             gradient_checkpointing=True, logging_steps=5, report_to=[], seed=a.seed, save_strategy="no",
                             remove_unused_columns=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=lambda b: collate(b, pad_id))
    result = trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    _sft_lora().complete_config(a.base, out)
    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"):   # vLLM loads the image processor of the VL checkpoint
        if (Path(a.base) / f).exists():
            shutil.copy(Path(a.base) / f, out / f)
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    rec = {"base": a.base, "data": a.data, "rows": len(rows), "examples": len(examples), "trained_tokens": trained, "args": vars(a),
           "train_loss": result.training_loss, "steps": result.global_step, "minutes": round((time.time() - t0) / 60, 1),
           "log": [r for r in trainer.state.log_history if "loss" in r]}
    (out / "train_report.json").write_text(json.dumps(rec, indent=1))
    print(f"done: {out} loss {result.training_loss:.4f}, {rec['minutes']} min")


if __name__ == "__main__":
    main()
