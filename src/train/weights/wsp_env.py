"""Weight-space-programming environment for an LLM agent, following arXiv 2611.09382 (Hu et al., 2026): the agent edits the weights of a model directly (no gradients,
no training), sees the target ONLY through a budgeted forward tool, and diagnoses before it edits.

Forward tool (50 successful requests per search, a request may hold many prompts):
  G  generation  env.generate(rows, weights="edited"|"original"|"both")  -> per-item ROUGE-L / length ratio and, for "both", the paired difference with its standard error
  TF teacher-forced env.tf(rows, texts=None, topk=0, hidden_layers=(), mlp_layers=()) -> conditional log-probs of the continuation, top-k tokens, hidden states, MLP
     activations (no task score). Teacher forcing on the gold answer is the default continuation.
Python analyses (<= 300 per search): run in a persistent kernel by the agent; the harness counts `env.analysis()` calls and the brief caps them.
Original checkpoint stays unchanged: env.original (CPU copy) is never modified; env.reset() restores it; env.n_changed() counts differing bf16 values.
Rows are train-split feedback prompts (tail/mid), never dev/test. Sampling deviation from the paper: greedy, no thinking (our task has no reasoning trace), so the only
noise is across questions (paired SE).

    from train.weights.wsp_env import Env
    env = Env("data/models/dpo_Vp_A2", "data/wsp/feedback.jsonl", "data/wsp/run1")
    env.generate(range(0, 64), weights="original")
"""
import json
import math
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
            raise RuntimeError(f"budget exhausted ({self.total})")
        self.path.write_text(str(self.used + 1))
        return self.total - self.used


def paired_se(diffs: list[float]) -> float:
    return statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")


class Env:
    def __init__(self, base: str, feedback: str, workdir: str, forward_budget: int = 50, analysis_budget: int = 300, device: str = "cuda", batch: int = 16):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.base, self.device, self.batch = base, device, batch
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.forward = Budget(self.work / "forward_budget.txt", forward_budget)
        self.analyses = Budget(self.work / "analysis_budget.txt", analysis_budget)
        self.rows = [json.loads(line) for line in open(feedback, encoding="utf-8")]
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "left"
        Model = AutoModelForImageTextToText if hasattr(AutoConfig.from_pretrained(base), "vision_config") else AutoModelForCausalLM
        self.model = Model.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.original = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ---- bookkeeping
    def analysis(self) -> int:
        """Call once per Python analysis step; returns the analyses left."""
        return self.analyses.spend()

    def reset(self) -> None:
        self.model.load_state_dict(self.original)

    def n_changed(self) -> int:
        return sum(int((v.cpu() != self.original[k]).sum()) for k, v in self.model.state_dict().items())

    # ---- G: generation
    def _generate(self, idx: list[int], max_new_tokens: int) -> list[str]:
        import torch
        outs = []
        for i in range(0, len(idx), self.batch):
            part = [self.rows[j]["prompt"] for j in idx[i:i + self.batch]]
            enc = self.tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            outs += [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
        return outs

    def generate(self, rows, weights: str = "edited", max_new_tokens: int = 400) -> dict:
        """One G request. Answers of the ORIGINAL weights are cached per (row, max_new_tokens) for the life of the kernel, so "both"/"original" only generate the
        uncached rows and "both" costs about one generation pass of the edited model."""
        from evaluate.metrics import rouge_l_score
        idx = list(rows)
        left = self.forward.spend()
        cache = self.__dict__.setdefault("_orig_cache", {})
        res = {}
        if weights in ("original", "both"):
            todo = [j for j in idx if (j, max_new_tokens) not in cache]
            if todo:
                edited = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.load_state_dict(self.original)
                for j, a in zip(todo, self._generate(todo, max_new_tokens)):
                    cache[(j, max_new_tokens)] = a
                self.model.load_state_dict(edited)
            res["original"] = [cache[(j, max_new_tokens)] for j in idx]
        if weights in ("edited", "both"):
            res["edited"] = self._generate(idx, max_new_tokens)
        out = {"n": len(idx), "forward_left": left}
        scored = {}
        for name, ans in res.items():
            r = [rouge_l_score(a, self.rows[j]["reference"]) for a, j in zip(ans, idx)]
            ln = [len(a) / max(len(self.rows[j]["reference"]), 1) for a, j in zip(ans, idx)]
            scored[name] = r
            out[name] = {"rouge_l": round(statistics.mean(r), 4), "len_ratio": round(statistics.mean(ln), 3), "per_item": {"answers": ans, "rouge_l": r, "len_ratio": ln}}
        if weights == "both":
            d = [e - o for e, o in zip(scored["edited"], scored["original"])]
            out["paired_diff"] = {"mean": round(statistics.mean(d), 4), "se": round(paired_se(d), 4), "rescued": sum(x > 0.1 for x in d), "regressed": sum(x < -0.1 for x in d)}
        return out

    # ---- TF: teacher forcing
    def tf(self, rows, texts: list[str] | None = None, topk: int = 0, hidden_layers=(), mlp_layers=()) -> dict:
        """Teacher-forced scoring of a continuation per prompt (default: the gold answer; or `texts`, one per row). Returns mean/ per-token log-probs, optional top-k
        tokens and hidden states (residual after the given layers) / MLP outputs at the continuation positions. One forward request in total."""
        import torch
        idx = list(rows)
        left = self.forward.spend()
        layers = self._layers()
        hooks, mlp_cache = [], {}
        for L in mlp_layers:
            hooks.append(layers[L].mlp.register_forward_hook(lambda m, i, o, L=L: mlp_cache.__setitem__(L, o[0].detach().float().cpu())))
        out = []
        try:
            for n, j in enumerate(idx):
                p = self.tok(self.rows[j]["prompt"], add_special_tokens=False)["input_ids"]
                g = self.tok((texts[n] if texts else self.rows[j]["reference"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
                x = torch.tensor([p + g], device=self.device)
                with torch.no_grad():
                    o = self.model(x, output_hidden_states=bool(hidden_layers))
                lp = torch.log_softmax(o.logits[0, len(p) - 1:-1].float(), -1)
                tl = lp.gather(1, torch.tensor(g, device=self.device)[:, None])[:, 0]
                item = {"mean_logprob": round(tl.mean().item(), 4), "token_logprobs": tl.cpu().tolist(), "tokens": self.tok.convert_ids_to_tokens(g)}
                if topk:
                    v, ix = lp.topk(topk, -1)
                    item["topk"] = [[(self.tok.convert_ids_to_tokens([t])[0], round(s, 3)) for t, s in zip(r, vv)] for r, vv in zip(ix.cpu().tolist(), v.cpu().tolist())]
                if hidden_layers:
                    item["hidden"] = {L: o.hidden_states[L + 1][0, len(p):].float().cpu() for L in hidden_layers}
                if mlp_layers:
                    item["mlp"] = {L: mlp_cache[L][len(p):] for L in mlp_layers}
                out.append(item)
        finally:
            for h in hooks:
                h.remove()
        return {"items": out, "forward_left": left}

    def _layers(self):
        m = self.model
        for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
            obj = m
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("cannot find transformer layers")

    def save(self, out: str) -> None:
        import shutil
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(o, safe_serialization=True)
        for f in Path(self.base).iterdir():
            if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and f.name != "model.safetensors.index.json" and not (o / f.name).exists():
                shutil.copy(f, o / f.name)
