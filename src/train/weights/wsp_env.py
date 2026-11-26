"""Weight-space-programming environment for an LLM agent (cf. arXiv 2611.09382): load a model, edit its weights in Python, measure ROUGE-L on a FEEDBACK set of
train prompts (never dev/test), under a hard budget of generation evals. The agent only ever sees feedback-set numbers; dev is checked by us afterwards.

    from train.weights.wsp_env import Env
    env = Env("data/models/dpo_Vp_A2", feedback="data/wsp/feedback.jsonl", workdir="data/wsp/run1", budget=50)
    env.baseline()                       # counts as 1 eval
    sd = env.model.state_dict()          # edit in place, e.g. with torch.no_grad(): p.add_(...)
    env.eval(n=60)                       # greedy generation on the first n feedback prompts -> dict(rouge_l, len_ratio, n, left)
    env.save("data/wsp/run1/ckpt_a")     # writes a serve-able model dir
"""
import json
import statistics
from pathlib import Path


class Budget:
    def __init__(self, path: Path, total: int):
        self.path, self.total = Path(path), total
        if not self.path.exists():
            self.path.write_text("0")

    @property
    def used(self) -> int:
        return int(self.path.read_text())

    def spend(self) -> int:
        if self.used >= self.total:
            raise RuntimeError(f"eval budget exhausted ({self.total})")
        self.path.write_text(str(self.used + 1))
        return self.total - self.used


class Env:
    def __init__(self, base: str, feedback: str, workdir: str, budget: int = 50, device: str = "cuda"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.base, self.device = base, device
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.budget = Budget(self.work / "budget.txt", budget)
        self.rows = [json.loads(line) for line in open(feedback, encoding="utf-8")]
        self.tok = AutoTokenizer.from_pretrained(base)
        Model = AutoModelForImageTextToText if hasattr(AutoConfig.from_pretrained(base), "vision_config") else AutoModelForCausalLM
        self.model = Model.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.original = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}   # for env.reset()

    def reset(self) -> None:
        self.model.load_state_dict(self.original)

    def eval(self, n: int = 60, max_new_tokens: int = 400) -> dict:
        import torch
        from evaluate.metrics import rouge_l_score
        left = self.budget.spend()
        scores, ratios = [], []
        for r in self.rows[:n]:
            ids = self.tok(r["prompt"], return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            ans = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            scores.append(rouge_l_score(ans, r["reference"]))
            ratios.append(len(ans) / max(len(r["reference"]), 1))
        return {"rouge_l": round(statistics.mean(scores), 4), "len_ratio": round(statistics.mean(ratios), 3), "n": len(scores), "evals_left": left}

    def baseline(self, n: int = 60) -> dict:
        self.reset()
        return self.eval(n)

    def logprob(self, n: int = 60) -> float:
        """Mean per-token log-prob of the gold answer given the prompt (teacher forcing; free, does not use the budget)."""
        import torch
        tot = []
        for r in self.rows[:n]:
            p = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
            g = self.tok(r["reference"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            x = torch.tensor([p + g], device=self.device)
            with torch.no_grad():
                lp = torch.log_softmax(self.model(x).logits[0, len(p) - 1:-1].float(), -1)
            tot.append(lp.gather(1, torch.tensor(g, device=self.device)[:, None]).mean().item())
        return round(statistics.mean(tot), 4)

    def save(self, out: str) -> None:
        import shutil
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(o, safe_serialization=True)
        for f in Path(self.base).iterdir():
            if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and f.name != "model.safetensors.index.json" and not (o / f.name).exists():
                shutil.copy(f, o / f.name)
