"""GRPO on top of an SFT checkpoint, verifiable reward vs the course reference answer (no reward model).

    python src/train/grpo.py --data data/train/q35_v0 --base data/models/q35_v0 --out data/models/q35_rl0 [--limit 16]

Prompts = rows of src/train/sft_data.py (`prompt` string with chat template + `completion` = reference). Only rows the SFT
model answers badly carry gradient; `--hard` keeps the rows whose first rollout scores below that threshold (default: all).
Reward = .6 rougeL + .2 token overlap + .2 citation (same "Điều N" as the reference) - copy penalty - length penalty;
refusal-like answers get 0, unless the reference is itself a refusal (no-answer row): then refusal = 1, else 0. Metrics are the course ones (src/evaluate/metrics.py, loaded by path: numpy only).
"""
import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

REFUSAL = ("dựa trên các ngữ cảnh", "các ngữ cảnh được cung cấp không", "ngữ cảnh được cung cấp không", "trong các ngữ cảnh")
DIEU = re.compile(r"điều\s+(\d+)", re.I)


def load_metrics():
    p = Path(__file__).resolve().parents[1] / "evaluate" / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load_metrics()


def ngrams(text: str, n: int = 8) -> set:
    w = M.normalize_text(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(REFUSAL)


def filter_long(rows, count_tokens, max_tokens: int):
    """Drop rows whose 'prompt' is over max_tokens. Returns (kept_rows, dropped_count)."""
    kept = [r for r in rows if count_tokens(r["prompt"]) <= max_tokens]
    return kept, len(rows) - len(kept)


def score(pred: str, ref: str, prompt: str, length_pen: bool = True, cos: float | None = None, cite_weight: float = 0.2) -> float:
    """Reward of one completion. Pure function (unit-testable, no GPU). `cos` = cosine to the reference (config 3), else None."""
    p = pred.strip()
    if not p:
        return 0.0
    if is_refusal(ref):     # no-answer row: a refusal is the correct behaviour, anything else is wrong
        return 1.0 if is_refusal(p) else 0.0
    if is_refusal(p):
        return 0.0
    cite = 1.0
    want = set(DIEU.findall(ref))
    if want:
        cite = len(want & set(DIEU.findall(p))) / len(want)
    rest = (1.0 - cite_weight) / 0.8      # the other terms keep their proportions; cite_weight 0.2 is the original reward
    if cos is None:
        r = rest * (0.6 * M.rouge_l_score(p, ref) + 0.2 * M.token_overlap_score(p, ref)) + cite_weight * cite
    else:
        r = rest * (0.5 * M.rouge_l_score(p, ref) + 0.2 * cos + 0.1 * M.token_overlap_score(p, ref)) + cite_weight * cite
    g = ngrams(p)
    copy = len(g & ngrams(prompt)) / len(g) if g else 0.0       # share of 8-grams found in the prompt context
    ratio = len(M.normalize_text(p).split()) / max(1, len(M.normalize_text(ref).split()))
    # copy of > 90 % of the prompt's 8-grams; two-sided length band 0.6x..1.5x of the reference (a one-sided penalty pushes to evasive short answers)
    pen = 0.3 * max(0.0, copy - 0.9) / 0.1 + (0.2 * min(1.0, max(0.0, ratio - 1.5, 0.6 - ratio)) if length_pen else 0.0)
    return max(0.0, r - pen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8, help="completions per prompt (the group)")
    ap.add_argument("--batch", type=int, default=2, help="prompts per device step")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=6144)
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--beta", type=float, default=0.0, help="KL to the SFT policy (0 = no ref model; research: 0.04 hurt)")
    ap.add_argument("--no-length", action="store_true", help="config 2: no length penalty")
    ap.add_argument("--cosine", action="store_true", help="config 3: cosine (course sbert) in the reward")
    ap.add_argument("--use-vllm", action="store_true", help="colocated vLLM rollouts (needs vllm in this env)")
    ap.add_argument("--max-prompt-tokens", type=int, default=0, help="drop train rows whose prompt exceeds N tokens (0 = off)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows, 10 steps")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    t0, out = time.time(), Path(a.out)
    ds = load_dataset("json", data_files=str(Path(a.data) / "train.jsonl"), split="train")
    ds = ds.remove_columns(["qid"]).rename_column("completion", "reference")
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(a.base)
    dropped = 0
    if a.max_prompt_tokens > 0:
        rows = [{"prompt": r["prompt"], "i": i} for i, r in enumerate(ds)]
        kept, dropped = filter_long(rows, lambda p: len(tok(p).input_ids), a.max_prompt_tokens)
        print(f"--max-prompt-tokens {a.max_prompt_tokens}: dropped {dropped} of {len(ds)} rows")
        ds = ds.select([r["i"] for r in kept])

    emb = None
    if a.cosine:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("keepitreal/vietnamese-sbert", device="cuda")

    def reward(prompts, completions, reference, **_):
        cos = [None] * len(completions)
        if emb is not None:
            v = emb.encode(list(completions) + list(reference), normalize_embeddings=True, convert_to_numpy=True)
            n = len(completions)
            cos = [float(v[i] @ v[n + i]) for i in range(n)]
        return [score(c, r, p, not a.no_length, k) for p, c, r, k in zip(prompts, completions, reference, cos)]

    multimodal = hasattr(AutoConfig.from_pretrained(a.base), "vision_config")
    Model = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = Model.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=a.r, lora_alpha=a.r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
                      exclude_modules=r".*(visual|vision|lm_head).*" if multimodal else None)
    args = GRPOConfig(output_dir=str(out / "ckpt"), max_steps=10 if a.limit else a.steps, learning_rate=a.lr,
                      lr_scheduler_type="constant_with_warmup", warmup_steps=5, per_device_train_batch_size=a.batch * a.gens,
                      gradient_accumulation_steps=a.accum, num_generations=a.gens, max_completion_length=a.max_completion,
                      temperature=0.8, beta=a.beta, loss_type="dapo", scale_rewards="group", bf16=True, gradient_checkpointing=True, logging_steps=1,
                      save_strategy="steps", save_steps=50, save_total_limit=2, report_to=[], seed=a.seed,
                      use_vllm=a.use_vllm, vllm_mode="colocate" if a.use_vllm else None)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    import shutil
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    from sft_lora import complete_config  # noqa: E402  (same folder)
    if Path(a.base).is_dir():   # local SFT base: copy its config / processor / template files (never weights)
        for f in Path(a.base).iterdir():
            if f.suffix in (".json", ".jinja") and not f.name.endswith(".index.json") and not (out / f.name).exists():
                shutil.copy(f, out / f.name)
    else:
        complete_config(a.base, out)
    (out / "train_report.json").write_text(json.dumps({"args": vars(a), "rows": len(ds), "dropped_long": dropped, "minutes": round((time.time() - t0) / 60, 1),
                                                        "log": [r for r in trainer.state.log_history if "reward" in r]}, indent=1))
    print(f"done: {out} in {round((time.time() - t0) / 60, 1)} min")


if __name__ == "__main__":
    main()
